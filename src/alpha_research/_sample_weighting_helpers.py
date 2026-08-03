"""Private computation helpers for AFML-style sample weighting."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd


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
        return cast(Sequence[object] | pd.Index | None, bar_index.get(group_value))
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
    for start, end, is_valid in zip(left, right, valid, strict=True):
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
    for index, (start, end, is_valid) in enumerate(zip(left, right, valid, strict=True)):
        if is_valid:
            count = int(end - start + 1)
            uniqueness[index] = (inverse_prefix[int(end) + 1] - inverse_prefix[int(start)]) / count

    if returns is None:
        return uniqueness, None
    aligned_returns = pd.to_numeric(returns, errors="coerce").reindex(bars).fillna(0.0)
    attributed = aligned_returns.abs().to_numpy(dtype=float) * inverse
    attributed_prefix = np.concatenate([[0.0], np.cumsum(attributed)])
    attribution = np.zeros(len(events), dtype=float)
    for index, (start, end, is_valid) in enumerate(zip(left, right, valid, strict=True)):
        if is_valid:
            attribution[index] = attributed_prefix[int(end) + 1] - attributed_prefix[int(start)]
    return uniqueness, attribution


def _grouped_interval_weights(
    events: pd.DataFrame,
    *,
    bar_index: Sequence[object] | pd.Index | Mapping[object, Sequence[object]] | None,
    returns: pd.Series | None,
    group_col: str | None,
    start_col: str,
    end_col: str,
) -> tuple[pd.Series, pd.Series, int, int]:
    uniqueness = pd.Series(0.0, index=events.index, dtype=float)
    attribution = pd.Series(0.0, index=events.index, dtype=float)
    groups = (
        events.groupby(group_col, sort=False, dropna=False)
        if group_col is not None
        else [(None, events)]
    )
    total_bars = 0
    group_count = 0
    for group_value, group in groups:
        group_count += 1
        group_bars = _group_bar_index(bar_index, group_value)
        bars = _resolve_bars(group, group_bars, start_col=start_col, end_col=end_col)
        total_bars += len(bars)
        group_uniqueness, group_attribution = _interval_weights(
            group,
            bars,
            returns=_group_returns(returns, group_value, group_col),
            start_col=start_col,
            end_col=end_col,
        )
        uniqueness.loc[group.index] = group_uniqueness
        if group_attribution is not None:
            attribution.loc[group.index] = group_attribution
    return uniqueness, attribution, total_bars, group_count
