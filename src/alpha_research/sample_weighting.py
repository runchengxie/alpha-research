"""Sample weighting and sequential bootstrap for overlapping financial labels.

Panel event weights are computed per instrument. Labels on two different
symbols do not share underlying returns merely because their calendar windows
overlap.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

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
    group_count: int
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
    """Build a bar-by-event indicator matrix for a single return process.

    For multi-symbol panels use :func:`build_event_sample_weights`, which
    computes block-independent concurrency without allocating a giant dense
    panel matrix.
    """

    normalized = _normalize_events(
        events,
        event_id_col=event_id_col,
        start_col=start_col,
        end_col=end_col,
    )
    if normalized.empty:
        return pd.DataFrame(dtype=np.int8)
    bars = _resolve_bars(normalized, bar_index, start_col=start_col, end_col=end_col)
    matrix = np.zeros((len(bars), len(normalized)), dtype=np.int8)
    bar_values = bars.to_numpy(dtype="datetime64[ns]")
    for column, event in enumerate(normalized.itertuples(index=False)):
        start = pd.Timestamp(getattr(event, start_col)).to_datetime64()
        end = pd.Timestamp(getattr(event, end_col)).to_datetime64()
        left = int(np.searchsorted(bar_values, start, side="left"))
        right = int(np.searchsorted(bar_values, end, side="right"))
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
    """Compute average uniqueness for every event in an indicator matrix."""

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
    """Weight events by absolute bar-return attribution and concurrency."""

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
    bar_index: Sequence[object] | pd.Index | Mapping[object, Sequence[object]] | None = None,
    returns: pd.Series | None = None,
    event_id_col: str = "event_id",
    start_col: str = "label_start",
    end_col: str = "label_end",
    group_col: str | None = None,
) -> tuple[pd.DataFrame, SampleWeightReceipt]:
    """Build normalized event weights and a reproducibility receipt.

    When ``group_col`` is provided, concurrency is computed independently for
    each group. If omitted and a ``symbol`` column is present, grouping by
    symbol is enabled automatically.
    """

    cfg = config or SampleWeightConfig()
    normalized = _normalize_events(
        events,
        event_id_col=event_id_col,
        start_col=start_col,
        end_col=end_col,
    )
    resolved_group_col = group_col or ("symbol" if "symbol" in normalized.columns else None)
    if resolved_group_col is not None and resolved_group_col not in normalized.columns:
        raise ValueError(f"group column not found: {resolved_group_col}")

    uniqueness = pd.Series(0.0, index=normalized.index, dtype=float)
    attribution = pd.Series(0.0, index=normalized.index, dtype=float)
    total_bars = 0
    groups = (
        normalized.groupby(resolved_group_col, sort=False, dropna=False)
        if resolved_group_col is not None
        else [(None, normalized)]
    )
    group_count = 0
    for group_value, group in groups:
        group_count += 1
        group_bars = _group_bar_index(bar_index, group_value)
        bars = _resolve_bars(group, group_bars, start_col=start_col, end_col=end_col)
        total_bars += len(bars)
        group_uniqueness, group_attribution = _interval_weights(
            group,
            bars,
            returns=_group_returns(returns, group_value, resolved_group_col),
            start_col=start_col,
            end_col=end_col,
        )
        uniqueness.loc[group.index] = group_uniqueness
        if group_attribution is not None:
            attribution.loc[group.index] = group_attribution

    weighted_uniqueness = uniqueness.pow(cfg.uniqueness_power)
    mode = cfg.mode
    if mode.startswith("return_attribution"):
        if returns is None:
            raise ValueError(f"{mode} requires returns")
        base = attribution
    elif mode.startswith("uniqueness"):
        base = weighted_uniqueness
    elif mode == "time_decay":
        base = pd.Series(1.0, index=normalized.index, dtype=float)
    else:
        raise ValueError(f"Unsupported sample weight mode: {mode}")

    decay = pd.Series(1.0, index=normalized.index, dtype=float, name="time_decay")
    if mode.endswith("time_decay") or mode == "time_decay":
        if cfg.time_decay_halflife is None:
            raise ValueError(f"{mode} requires time_decay_halflife")
        decay = time_decay_weights(normalized[end_col], halflife=cfg.time_decay_halflife)

    raw = pd.to_numeric(base, errors="coerce").fillna(0.0) * decay
    if cfg.min_weight > 0:
        raw = raw.clip(lower=cfg.min_weight)
    weights = _normalize_mean_one(raw)
    result = normalized.copy()
    result["average_uniqueness"] = uniqueness.to_numpy(dtype=float)
    result["time_decay_weight"] = decay.to_numpy(dtype=float)
    result["sample_weight"] = weights.to_numpy(dtype=float)

    receipt = SampleWeightReceipt(
        schema_version=1,
        mode=mode,
        event_count=len(result),
        bar_count=total_bars,
        group_count=group_count,
        average_uniqueness=float(result["average_uniqueness"].mean())
        if not result.empty
        else float("nan"),
        effective_sample_size=_effective_sample_size(weights),
        min_weight=float(weights.min()) if not weights.empty else float("nan"),
        max_weight=float(weights.max()) if not weights.empty else float("nan"),
        weight_concentration_hhi=_weight_hhi(weights),
        events_sha256=_events_hash(
            normalized,
            event_id_col,
            start_col,
            end_col,
            resolved_group_col,
        ),
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
    """Persist weights and their receipt."""

    weights_target = Path(weights_path)
    receipt_target = Path(receipt_path)
    weights_target.parent.mkdir(parents=True, exist_ok=True)
    receipt_target.parent.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(weights_target, index=False)
    receipt_target.write_text(
        json.dumps(receipt.to_dict(), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _interval_weights(
    events: pd.DataFrame,
    bars: pd.DatetimeIndex,
    *,
    returns: pd.Series | None,
    start_col: str,
    end_col: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    bar_values = bars.to_numpy(dtype="datetime64[ns]")
    starts = pd.to_datetime(events[start_col]).to_numpy(dtype="datetime64[ns]")
    ends = pd.to_datetime(events[end_col]).to_numpy(dtype="datetime64[ns]")
    left = np.searchsorted(bar_values, starts, side="left")
    right = np.searchsorted(bar_values, ends, side="right") - 1
    valid = (left >= 0) & (right >= left) & (right < len(bars))
    difference = np.zeros(len(bars) + 1, dtype=float)
    for start, end, is_valid in zip(left, right, valid):
        if not is_valid:
            continue
        difference[int(start)] += 1.0
        difference[int(end) + 1] -= 1.0
    concurrency = np.cumsum(difference[:-1])
    inverse = np.divide(
        1.0,
        concurrency,
        out=np.zeros_like(concurrency),
        where=concurrency > 0,
    )
    inverse_prefix = np.concatenate([[0.0], np.cumsum(inverse)])
    uniqueness = np.zeros(len(events), dtype=float)
    for index, (start, end, is_valid) in enumerate(zip(left, right, valid)):
        if is_valid:
            count = int(end - start + 1)
            uniqueness[index] = (
                inverse_prefix[int(end) + 1] - inverse_prefix[int(start)]
            ) / count

    if returns is None:
        return uniqueness, None
    aligned_returns = pd.to_numeric(returns, errors="coerce").reindex(bars).fillna(0.0)
    attributed = aligned_returns.abs().to_numpy(dtype=float) * inverse
    attributed_prefix = np.concatenate([[0.0], np.cumsum(attributed)])
    attribution = np.zeros(len(events), dtype=float)
    for index, (start, end, is_valid) in enumerate(zip(left, right, valid)):
        if is_valid:
            attribution[index] = (
                attributed_prefix[int(end) + 1] - attributed_prefix[int(start)]
            )
    return uniqueness, attribution


def _resolve_bars(
    events: pd.DataFrame,
    bar_index: Sequence[object] | pd.Index | None,
    *,
    start_col: str,
    end_col: str,
) -> pd.DatetimeIndex:
    if bar_index is None:
        values = pd.concat([events[start_col], events[end_col]], ignore_index=True)
        bars = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce")).dropna()
    else:
        bars = pd.DatetimeIndex(pd.to_datetime(pd.Index(bar_index), errors="coerce")).dropna()
    bars = bars.drop_duplicates().sort_values()
    if bars.empty:
        raise ValueError("bar_index must contain at least one valid timestamp")
    return bars


def _group_bar_index(
    bar_index: Sequence[object] | pd.Index | Mapping[object, Sequence[object]] | None,
    group_value: object,
) -> Sequence[object] | pd.Index | None:
    if isinstance(bar_index, Mapping):
        return bar_index.get(group_value)
    return bar_index


def _group_returns(
    returns: pd.Series | None,
    group_value: object,
    group_col: str | None,
) -> pd.Series | None:
    if returns is None:
        return None
    if group_col is None:
        return returns
    if not isinstance(returns.index, pd.MultiIndex):
        raise ValueError("grouped return attribution requires a MultiIndex returns series")
    try:
        selected = returns.xs(group_value, level=0)
    except KeyError:
        return pd.Series(dtype=float)
    return selected


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
    group_col: str | None,
) -> str:
    columns = [event_id_col, start_col, end_col]
    if group_col is not None:
        columns.insert(1, group_col)
    payload = events[columns].to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S")
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
