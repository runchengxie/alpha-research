from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd


def _coerce_sample_weight_min(value: object) -> float:
    try:
        min_weight = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_weight_params.min_weight must be a number.") from exc
    if min_weight < 0:
        raise ValueError("sample_weight_params.min_weight must be >= 0.")
    return min_weight


def _time_decay_weights(
    data: pd.DataFrame,
    *,
    date_col: str,
    params: Mapping[str, object] | None,
) -> np.ndarray | None:
    if params is not None and not isinstance(params, Mapping):
        raise ValueError("sample_weight_params must be a mapping.")
    params_map = dict(params or {})
    halflife_raw = params_map.get("halflife", params_map.get("half_life"))
    decay_rate_raw = params_map.get("decay_rate", params_map.get("rate"))
    min_weight = _coerce_sample_weight_min(params_map.get("min_weight", 0.0))

    if halflife_raw is not None:
        decay_base = 0.5
        try:
            decay_scale = float(cast(Any, halflife_raw))
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_weight_params.halflife must be a number.") from exc
        if not np.isfinite(decay_scale) or decay_scale <= 0:
            raise ValueError("sample_weight_params.halflife must be > 0.")
    elif decay_rate_raw is not None:
        try:
            decay_base = float(cast(Any, decay_rate_raw))
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_weight_params.decay_rate must be a number.") from exc
        if not np.isfinite(decay_base) or decay_base <= 0 or decay_base > 1:
            raise ValueError("sample_weight_params.decay_rate must be in (0, 1].")
        decay_scale = 1.0
    else:
        raise ValueError(
            "exp_decay/time_decay sample_weight_mode requires either "
            "sample_weight_params.halflife or sample_weight_params.decay_rate."
        )

    date_values = pd.to_datetime(data[date_col], errors="coerce")
    if date_values.isna().any():
        raise ValueError(f"sample weights require valid dates in column: {date_col}")
    unique_dates = pd.Index(date_values.unique()).sort_values()
    if unique_dates.empty:
        return None
    unique_ages = float(len(unique_dates) - 1) - np.arange(len(unique_dates), dtype=float)
    unique_date_weights = np.power(decay_base, unique_ages / decay_scale)
    if min_weight > 0:
        unique_date_weights = np.maximum(unique_date_weights, min_weight)
    mean_weight = float(np.nanmean(unique_date_weights))
    if np.isfinite(mean_weight) and mean_weight > 0:
        unique_date_weights = unique_date_weights / mean_weight
    date_weight_map = pd.Series(unique_date_weights, index=unique_dates, dtype=float)
    date_weights = date_values.map(date_weight_map).to_numpy(dtype=float)
    counts = data.groupby(date_col, sort=False)[date_col].transform("count").to_numpy(dtype=float)
    return date_weights / counts


def select_train_window_dates(
    dates: np.ndarray | list[pd.Timestamp],
    *,
    mode: str | None = None,
    size: int | None = None,
    unit: str = "dates",
) -> np.ndarray:
    mode_text = str(mode or "full").strip().lower()
    if mode_text in {"", "full", "all", "expanding"}:
        return np.asarray(pd.to_datetime(dates).unique(), dtype="datetime64[ns]")
    if mode_text not in {"rolling", "recent"}:
        raise ValueError("train_window.mode must be one of: full, rolling.")
    if size is None:
        raise ValueError("train_window.size is required when train_window.mode=rolling.")
    try:
        size_value = int(size)
    except (TypeError, ValueError) as exc:
        raise ValueError("train_window.size must be a positive integer.") from exc
    if size_value <= 0:
        raise ValueError("train_window.size must be a positive integer.")

    date_index = pd.Index(pd.to_datetime(dates).unique()).sort_values()
    if date_index.empty:
        return np.array([], dtype="datetime64[ns]")

    unit_text = str(unit or "dates").strip().lower()
    if unit_text == "dates":
        return np.asarray(date_index[-size_value:], dtype="datetime64[ns]")
    if unit_text == "years":
        end_date = cast(pd.Timestamp, pd.Timestamp(date_index[-1]))
        cutoff = end_date - pd.DateOffset(years=size_value)
        selected = date_index[date_index >= cutoff]
        if selected.empty:
            selected = date_index[-1:]
        return np.asarray(selected, dtype="datetime64[ns]")
    raise ValueError("train_window.unit must be one of: dates, years.")


@dataclass(frozen=True)
class _LabelEventWindow:
    signal_date: pd.Timestamp
    label_start: pd.Timestamp
    label_end: pd.Timestamp


@dataclass(frozen=True)
class _CVDateSlices:
    sorted_data: pd.DataFrame
    dates: np.ndarray
    date_start_rows: np.ndarray
    date_end_rows: np.ndarray


def _date_key(date: object) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(cast(Any, date))).normalize()


def _as_date_tuple(dates: object) -> tuple[pd.Timestamp, ...]:
    values = pd.to_datetime(
        list(cast(Any, dates)) if not isinstance(dates, pd.Series) else dates, errors="coerce"
    )
    cleaned = [
        cast(pd.Timestamp, pd.Timestamp(date)).normalize() for date in values if not pd.isna(date)
    ]
    return tuple(pd.Index(cleaned).drop_duplicates().sort_values())


def _lookup_shifted_date(
    date: pd.Timestamp,
    all_dates: tuple[pd.Timestamp, ...],
    shift_days: int,
) -> pd.Timestamp | None:
    try:
        idx = all_dates.index(date)
    except ValueError:
        return None
    shifted_idx = idx + max(0, int(shift_days))
    if shifted_idx >= len(all_dates):
        return None
    return all_dates[shifted_idx]


def _build_label_event_windows(
    signal_dates: object,
    *,
    all_trade_dates: object | None,
    horizon_mode: str,
    horizon_days: int | None,
    shift_days: int,
    next_rebalance_map: Mapping[object, object] | None = None,
) -> tuple[dict[pd.Timestamp, _LabelEventWindow], str]:
    dates = _as_date_tuple(signal_dates)
    trade_dates = _as_date_tuple(all_trade_dates if all_trade_dates is not None else signal_dates)
    if not dates or not trade_dates:
        return {}, "fallback_gap"
    if horizon_days is None:
        return {}, "fallback_gap"

    mode = str(horizon_mode or "fixed").strip().lower()
    next_map = {
        _date_key(key): _date_key(value)
        for key, value in (next_rebalance_map or {}).items()
        if not pd.isna(pd.to_datetime(cast(Any, key), errors="coerce"))
        and not pd.isna(pd.to_datetime(cast(Any, value), errors="coerce"))
    }
    windows: dict[pd.Timestamp, _LabelEventWindow] = {}
    for signal_date in dates:
        label_start = _lookup_shifted_date(signal_date, trade_dates, shift_days)
        if label_start is None:
            continue
        if mode == "next_rebalance":
            exit_signal = next_map.get(signal_date)
            if exit_signal is None:
                continue
            label_end = _lookup_shifted_date(exit_signal, trade_dates, shift_days)
        else:
            label_end = _lookup_shifted_date(
                signal_date,
                trade_dates,
                int(horizon_days) + int(shift_days),
            )
        if label_end is None:
            continue
        if label_end < label_start:
            label_start, label_end = label_end, label_start
        windows[signal_date] = _LabelEventWindow(
            signal_date=signal_date,
            label_start=label_start,
            label_end=label_end,
        )
    return windows, "event_window" if len(windows) == len(dates) else "fallback_gap"


def _event_windows_overlap(left: _LabelEventWindow, right: _LabelEventWindow) -> bool:
    return left.label_start <= right.label_end and right.label_start <= left.label_end


def _apply_event_window_purge_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dates: np.ndarray,
    event_windows: dict[pd.Timestamp, _LabelEventWindow],
    *,
    embargo_days: int,
) -> tuple[np.ndarray, bool]:
    val_dates = [_date_key(dates[idx]) for idx in val_idx]
    test_windows = [event_windows[date] for date in val_dates if date in event_windows]
    if len(test_windows) != len(val_dates):
        return train_idx, False

    keep: list[int] = []
    embargo_delta = cast(pd.Timedelta, pd.Timedelta(days=max(0, int(embargo_days))))
    for idx in train_idx:
        train_date = _date_key(dates[idx])
        train_window = event_windows.get(train_date)
        if train_window is None:
            continue
        if any(_event_windows_overlap(train_window, test_window) for test_window in test_windows):
            continue
        if embargo_delta > pd.Timedelta(0) and any(
            test_window.label_end
            < train_window.signal_date
            <= test_window.label_end + embargo_delta
            for test_window in test_windows
        ):
            continue
        keep.append(int(idx))
    return np.asarray(keep, dtype=train_idx.dtype), True


def _prepare_cv_date_slices(data: pd.DataFrame, date_col: str) -> _CVDateSlices:
    sorted_data = data.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    date_values = sorted_data[date_col].to_numpy()
    if date_values.size == 0:
        empty = np.array([], dtype=int)
        return _CVDateSlices(
            sorted_data=sorted_data,
            dates=np.array([]),
            date_start_rows=empty,
            date_end_rows=empty,
        )

    dates, date_start_rows = np.unique(date_values, return_index=True)
    date_end_rows = np.empty_like(date_start_rows)
    if len(date_start_rows) > 1:
        date_end_rows[:-1] = date_start_rows[1:]
    date_end_rows[-1] = len(sorted_data)
    return _CVDateSlices(
        sorted_data=sorted_data,
        dates=dates,
        date_start_rows=date_start_rows,
        date_end_rows=date_end_rows,
    )


def _validate_cv_purge_mode(cv_purge_mode: str) -> str:
    purge_mode = str(cv_purge_mode or "gap").strip().lower()
    if purge_mode not in {"gap", "event_window"}:
        raise ValueError("cv_purge_mode must be one of: gap, event_window.")
    return purge_mode


def _windowed_cv_train_indices(
    train_idx: np.ndarray,
    *,
    dates: np.ndarray,
    train_window_mode: str | None,
    train_window_size: int | None,
    train_window_unit: str,
) -> np.ndarray | None:
    train_dates = select_train_window_dates(
        dates[train_idx],
        mode=train_window_mode,
        size=train_window_size,
        unit=train_window_unit,
    )
    if len(train_dates) == 0:
        return None
    train_start_date = pd.to_datetime(train_dates[0])
    train_idx = train_idx[pd.to_datetime(dates[train_idx]) >= train_start_date]
    if len(train_idx) == 0:
        return None
    return train_idx
