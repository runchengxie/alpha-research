"""Sample weighting and sequential bootstrap for overlapping financial labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd


WeightMode = Literal[
    "uniqueness",
    "time_decay",
    "uniqueness_time_decay",
    "return_attribution",
    "return_attribution_time_decay",
]


@dataclass(frozen=True)
class SampleWeightConfig:
    mode: WeightMode = "uniqueness_time_decay"
    uniqueness_power: float = 1.0
    time_decay_halflife: float | None = None
    min_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.uniqueness_power <= 0:
            raise ValueError("uniqueness_power must be > 0")
        if self.time_decay_halflife is not None and self.time_decay_halflife <= 0:
            raise ValueError("time_decay_halflife must be > 0")
        if self.min_weight < 0:
            raise ValueError("min_weight must be >= 0")


@dataclass(frozen=True)
class SampleWeightReceipt:
    schema_version: int
    mode: str
    event_count: int
    bar_count: int
    average_uniqueness: float
    effective_sample_size: float
    min_weight: float
    max_weight: float
    weight_concentration_hhi: float
    events_sha256: str
    config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_indicator_matrix(
    events: pd.DataFrame,
    *,
    bar_index: Sequence[object] | pd.Index | None = None,
    event_id_col: str = "event_id",
    start_col: str = "label_start",
    end_col: str = "label_end",
) -> pd.DataFrame:
    """Build a bar-by-event indicator matrix for label concurrency."""

    normalized = _normalize_events(
        events,
        event_id_col=event_id_col,
        start_col=start_col,
        end_col=end_col,
    )
    if normalized.empty:
        return pd.DataFrame(dtype=np.int8)
    if bar_index is None:
        start = normalized[start_col].min()
        end = normalized[end_col].max()
        bars = pd.date_range(start, end, freq="D")
    else:
        bars = pd.DatetimeIndex(pd.to_datetime(pd.Index(bar_index), errors="coerce")).dropna()
        bars = bars.drop_duplicates().sort_values()
    if bars.empty:
        raise ValueError("bar_index must contain at least one valid timestamp")

    matrix = np.zeros((len(bars), len(normalized)), dtype=np.int8)
    bar_values = bars.to_numpy()
    for column, event in enumerate(normalized.itertuples(index=False)):
        start = getattr(event, start_col)
        end = getattr(event, end_col)
        left = int(np.searchsorted(bar_values, np.datetime64(start), side="left"))
        right = int(np.searchsorted(bar_values, np.datetime64(end), side="right"))
        if left < right:
            matrix[left:right, column] = 1
    return pd.DataFrame(
        matrix,
        index=bars.rename("bar_time"),
        columns=pd.Index(normalized[event_id_col].tolist(), name=event_id_col),
    )


def event_concurrency(indicator: pd.DataFrame) -> pd.Series:
    """Return the number of active labels on every bar."""

    if indicator.empty:
        return pd.Series(dtype=float, name="concurrency")
    return indicator.sum(axis=1).astype(float).rename("concurrency")


def average_uniqueness(indicator: pd.DataFrame) -> pd.Series:
    """Compute average uniqueness for every event."""

    if indicator.empty:
        return pd.Series(dtype=float, name="average_uniqueness")
    concurrency = event_concurrency(indicator).replace(0.0, np.nan)
    uniqueness = indicator.div(concurrency, axis=0).where(indicator.astype(bool))
    result = uniqueness.mean(axis=0, skipna=True).fillna(0.0)
    result.name = "average_uniqueness"
    return result.astype(float)


def return_attribution_weights(
    indicator: pd.DataFrame,
    returns: pd.Series,
) -> pd.Series:
    """Weight events by absolute return attribution adjusted for concurrency."""

    if indicator.empty:
        return pd.Series(dtype=float, name="return_attribution")
    aligned = pd.to_numeric(returns, errors="coerce").reindex(indicator.index).fillna(0.0)
    concurrency = event_concurrency(indicator).replace(0.0, np.nan)
    per_bar = aligned.abs().div(concurrency).fillna(0.0)
    weights = indicator.mul(per_bar, axis=0).sum(axis=0)
    weights.name = "return_attribution"
    return weights.astype(float)


def time_decay_weights(
    event_end: pd.Series,
    *,
    halflife: float,
) -> pd.Series:
    """Generate mean-one exponential time-decay weights by event order."""

    if halflife <= 0:
        raise ValueError("halflife must be > 0")
    dates = pd.to_datetime(event_end, errors="coerce")
    if dates.isna().any():
        raise ValueError("event end dates must be datetime-like")
    order = dates.rank(method="dense").astype(float)
    ages = float(order.max()) - order
    values = np.power(0.5, ages / float(halflife))
    return pd.Series(values, index=event_end.index, dtype=float, name="time_decay")


def build_event_sample_weights(
    events: pd.DataFrame,
    *,
    config: SampleWeightConfig | None = None,
    bar_index: Sequence[object] | pd.Index | None = None,
    returns: pd.Series | None = None,
    event_id_col: str = "event_id",
    start_col: str = "label_start",
    end_col: str = "label_end",
) -> tuple[pd.DataFrame, SampleWeightReceipt]:
    """Build normalized event weights and a reproducibility receipt."""

    cfg = config or SampleWeightConfig()
    normalized = _normalize_events(
        events,
        event_id_col=event_id_col,
        start_col=start_col,
        end_col=end_col,
    )
    indicator = build_indicator_matrix(
        normalized,
        bar_index=bar_index,
        event_id_col=event_id_col,
        start_col=start_col,
        end_col=end_col,
    )
    raw_uniqueness = average_uniqueness(indicator).reindex(normalized[event_id_col]).fillna(0.0)
    raw_uniqueness.index = normalized.index
    uniqueness = raw_uniqueness.pow(cfg.uniqueness_power)

    mode = cfg.mode
    if mode.startswith("return_attribution"):
        if returns is None:
            raise ValueError(f"{mode} requires returns")
        base = return_attribution_weights(indicator, returns).reindex(
            normalized[event_id_col]
        ).fillna(0.0)
        base.index = normalized.index
    elif mode.startswith("uniqueness"):
        base = uniqueness
    elif mode == "time_decay":
        base = pd.Series(1.0, index=normalized.index, dtype=float)
    else:
        raise ValueError(f"Unsupported sample weight mode: {mode}")

    decay = pd.Series(1.0, index=normalized.index, dtype=float, name="time_decay")
    if mode.endswith("time_decay") or mode == "time_decay":
        if cfg.time_decay_halflife is None:
            raise ValueError(f"{mode} requires time_decay_halflife")
        decay = time_decay_weights(
            normalized[end_col],
            halflife=cfg.time_decay_halflife,
        )

    raw = pd.to_numeric(base, errors="coerce").fillna(0.0) * decay
    if cfg.min_weight > 0:
        raw = raw.clip(lower=cfg.min_weight)
    weights = _normalize_mean_one(raw)
    result = normalized.copy()
    result["average_uniqueness"] = raw_uniqueness.to_numpy(dtype=float)
    result["time_decay_weight"] = decay.to_numpy(dtype=float)
    result["sample_weight"] = weights.to_numpy(dtype=float)

    receipt = SampleWeightReceipt(
        schema_version=1,
        mode=mode,
        event_count=len(result),
        bar_count=len(indicator),
        average_uniqueness=float(result["average_uniqueness"].mean())
        if not result.empty
        else float("nan"),
        effective_sample_size=_effective_sample_size(weights),
        min_weight=float(weights.min()) if not weights.empty else float("nan"),
        max_weight=float(weights.max()) if not weights.empty else float("nan"),
        weight_concentration_hhi=_weight_hhi(weights),
        events_sha256=_events_hash(normalized, event_id_col, start_col, end_col),
        config=asdict(cfg),
    )
    return result, receipt


def sequential_bootstrap(
    indicator: pd.DataFrame,
    *,
    sample_length: int | None = None,
    random_state: int | np.random.Generator | None = None,
) -> list[object]:
    """Draw event IDs while favoring candidates that add unique information."""

    if indicator.empty:
        return []
    length = int(sample_length if sample_length is not None else indicator.shape[1])
    if length < 0:
        raise ValueError("sample_length must be >= 0")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    values = indicator.to_numpy(dtype=float)
    selected_counts = np.zeros(indicator.shape[0], dtype=float)
    selected: list[int] = []
    for _ in range(length):
        scores = np.empty(indicator.shape[1], dtype=float)
        for candidate in range(indicator.shape[1]):
            candidate_active = values[:, candidate] > 0
            if not bool(candidate_active.any()):
                scores[candidate] = 0.0
                continue
            concurrency = selected_counts[candidate_active] + 1.0
            scores[candidate] = float(np.mean(1.0 / concurrency))
        total = float(scores.sum())
        probabilities = (
            np.repeat(1.0 / len(scores), len(scores))
            if not np.isfinite(total) or total <= 0
            else scores / total
        )
        chosen = int(rng.choice(len(scores), p=probabilities))
        selected.append(chosen)
        selected_counts += values[:, chosen]
    return [indicator.columns[index] for index in selected]


def write_sample_weight_artifacts(
    weights: pd.DataFrame,
    receipt: SampleWeightReceipt,
    *,
    weights_path: str | Path,
    receipt_path: str | Path,
) -> None:
    """Persist weights and their receipt atomically at the file level."""

    weights_target = Path(weights_path)
    receipt_target = Path(receipt_path)
    weights_target.parent.mkdir(parents=True, exist_ok=True)
    receipt_target.parent.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(weights_target, index=False)
    receipt_target.write_text(
        json.dumps(receipt.to_dict(), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _normalize_events(
    events: pd.DataFrame,
    *,
    event_id_col: str,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    missing = [column for column in (start_col, end_col) if column not in events.columns]
    if missing:
        raise ValueError(f"events is missing required columns: {', '.join(missing)}")
    result = events.copy()
    if event_id_col not in result.columns:
        result[event_id_col] = np.arange(len(result), dtype=int)
    result[start_col] = pd.to_datetime(result[start_col], errors="coerce")
    result[end_col] = pd.to_datetime(result[end_col], errors="coerce")
    if result[[start_col, end_col]].isna().any().any():
        raise ValueError("event windows must contain valid timestamps")
    if bool((result[end_col] < result[start_col]).any()):
        raise ValueError("event end must be on or after event start")
    if bool(result[event_id_col].duplicated().any()):
        raise ValueError(f"{event_id_col} must be unique")
    return result.sort_values([start_col, end_col, event_id_col], kind="mergesort").reset_index(
        drop=True
    )


def _normalize_mean_one(values: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean = float(cleaned.mean()) if not cleaned.empty else float("nan")
    if not np.isfinite(mean) or mean <= 0:
        return pd.Series(1.0, index=cleaned.index, dtype=float)
    return (cleaned / mean).astype(float)


def _effective_sample_size(weights: pd.Series) -> float:
    values = pd.to_numeric(weights, errors="coerce").dropna().to_numpy(dtype=float)
    denominator = float(np.square(values).sum())
    return float(values.sum() ** 2 / denominator) if denominator > 0 else 0.0


def _weight_hhi(weights: pd.Series) -> float:
    values = pd.to_numeric(weights, errors="coerce").clip(lower=0).dropna()
    total = float(values.sum())
    if total <= 0:
        return float("nan")
    shares = values / total
    return float(np.square(shares).sum())


def _events_hash(
    events: pd.DataFrame,
    event_id_col: str,
    start_col: str,
    end_col: str,
) -> str:
    payload = events[[event_id_col, start_col, end_col]].to_csv(
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "SampleWeightConfig",
    "SampleWeightReceipt",
    "average_uniqueness",
    "build_event_sample_weights",
    "build_indicator_matrix",
    "event_concurrency",
    "return_attribution_weights",
    "sequential_bootstrap",
    "time_decay_weights",
    "write_sample_weight_artifacts",
]
