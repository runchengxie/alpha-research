from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .metrics import daily_ic_series
from .modeling import build_model, fit_model, resolve_model_spec
from .transform import apply_score_postprocess


def _coerce_sample_weight_min(value: object) -> float:
    try:
        min_weight = float(value)
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
            decay_scale = float(halflife_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_weight_params.halflife must be a number.") from exc
        if not np.isfinite(decay_scale) or decay_scale <= 0:
            raise ValueError("sample_weight_params.halflife must be > 0.")
    elif decay_rate_raw is not None:
        try:
            decay_base = float(decay_rate_raw)
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


def build_sample_weight(
    data: pd.DataFrame,
    mode: str | None,
    *,
    date_col: str = "trade_date",
    params: Mapping[str, object] | None = None,
) -> np.ndarray | None:
    if mode is None:
        return None
    mode_text = str(mode).strip().lower()
    if mode_text in {"", "none", "null"}:
        return None
    if mode_text in {"date_equal", "date"}:
        counts = data.groupby(date_col, sort=False)[date_col].transform("count")
        return (1.0 / counts).to_numpy()
    if mode_text in {"time_decay", "exp_decay", "exp"}:
        return _time_decay_weights(data, date_col=date_col, params=params)
    raise ValueError(f"Unsupported sample_weight_mode: {mode}")


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
        end_date = pd.Timestamp(date_index[-1])
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


@dataclass(frozen=True)
class _CVFitConfig:
    model_type: str
    model_params: Mapping[str, object]
    features: list[str]
    fit_target: str
    eval_target: str
    date_col: str
    signal_direction: float
    sample_weight_mode: str | None
    sample_weight_params: Mapping[str, object] | None
    score_postprocess_method: str
    score_postprocess_columns: list[str] | None
    score_postprocess_strength: float
    score_postprocess_min_obs: int | None


def _date_key(date: object) -> pd.Timestamp:
    return pd.Timestamp(date).normalize()


def _as_date_tuple(dates: object) -> tuple[pd.Timestamp, ...]:
    values = pd.to_datetime(
        list(dates) if not isinstance(dates, pd.Series) else dates, errors="coerce"
    )
    cleaned = [pd.Timestamp(date).normalize() for date in values if not pd.isna(date)]
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
        if not pd.isna(pd.to_datetime(key, errors="coerce"))
        and not pd.isna(pd.to_datetime(value, errors="coerce"))
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
    embargo_delta = pd.Timedelta(days=max(0, int(embargo_days)))
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


def _resolve_cv_model_spec(
    model_cfg: Mapping[str, object] | None,
    model_params: Mapping[str, object] | None,
) -> tuple[str, Mapping[str, object]]:
    if model_cfg is not None and model_params is not None:
        raise ValueError("Provide either model_cfg or model_params, not both.")
    if model_params is not None:
        return resolve_model_spec({"type": "xgb_regressor", "params": dict(model_params)})
    if model_cfg is None:
        return resolve_model_spec({})
    if "type" in model_cfg or "params" in model_cfg:
        return resolve_model_spec(model_cfg)
    return resolve_model_spec({"type": "xgb_regressor", "params": dict(model_cfg)})


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


def _event_window_state(
    *,
    dates: np.ndarray,
    purge_mode: str,
    all_trade_dates: object | None,
    label_horizon_mode: str,
    label_horizon_days: int | None,
    label_shift_days: int,
    next_rebalance_map: Mapping[object, object] | None,
) -> tuple[dict[pd.Timestamp, _LabelEventWindow], str]:
    if purge_mode != "event_window":
        return {}, "fallback_gap"
    return _build_label_event_windows(
        dates,
        all_trade_dates=all_trade_dates,
        horizon_mode=label_horizon_mode,
        horizon_days=label_horizon_days,
        shift_days=label_shift_days,
        next_rebalance_map=next_rebalance_map,
    )


def _purged_cv_train_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    dates: np.ndarray,
    purge_mode: str,
    event_window_status: str,
    event_windows: dict[pd.Timestamp, _LabelEventWindow],
    embargo_days: int,
    gap: int,
) -> np.ndarray | None:
    used_event_window = False
    if purge_mode == "event_window" and event_window_status == "event_window":
        train_idx, used_event_window = _apply_event_window_purge_indices(
            train_idx,
            val_idx,
            dates,
            event_windows,
            embargo_days=embargo_days,
        )
        if len(train_idx) == 0:
            return None
    if not used_event_window and gap > 0:
        cutoff = val_idx[0] - gap
        train_idx = train_idx[train_idx < cutoff]
        if len(train_idx) == 0:
            return None
    return train_idx


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


def _cv_frame_for_date_indices(
    slices: _CVDateSlices,
    date_idx: np.ndarray,
    *,
    copy: bool = False,
) -> pd.DataFrame:
    start = slices.date_start_rows[date_idx[0]]
    end = slices.date_end_rows[date_idx[-1]]
    frame = slices.sorted_data.iloc[start:end]
    return frame.copy() if copy else frame


def _score_cv_fold(
    slices: _CVDateSlices,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: _CVFitConfig,
) -> float:
    tr_df = _cv_frame_for_date_indices(slices, train_idx)
    va_df = _cv_frame_for_date_indices(slices, val_idx, copy=True)

    model = build_model(config.model_type, config.model_params)
    sample_weight = build_sample_weight(
        tr_df,
        config.sample_weight_mode,
        date_col=config.date_col,
        params=config.sample_weight_params,
    )
    fit_model(
        model,
        config.model_type,
        tr_df,
        features=config.features,
        target_col=config.fit_target,
        sample_weight=sample_weight,
        date_col=config.date_col,
    )
    va_df["pred"] = model.predict(va_df[config.features])
    va_df["pred"] = apply_score_postprocess(
        va_df,
        "pred",
        method=config.score_postprocess_method,
        columns=config.score_postprocess_columns or [],
        strength=config.score_postprocess_strength,
        min_obs=config.score_postprocess_min_obs,
    )
    if config.signal_direction != 1.0:
        va_df["pred"] = va_df["pred"] * config.signal_direction

    ic_input = (
        va_df
        if config.date_col == "trade_date"
        else va_df.rename(columns={config.date_col: "trade_date"})
    )
    ic_values = daily_ic_series(ic_input, config.eval_target, "pred")
    return float(ic_values.mean()) if not ic_values.empty else np.nan


def time_series_cv_ic(
    data: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_splits: int,
    embargo_days: int,
    purge_days: int,
    model_cfg: Mapping[str, object] | None = None,
    signal_direction: float = 1.0,
    sample_weight_mode: str | None = None,
    sample_weight_params: Mapping[str, object] | None = None,
    date_col: str = "trade_date",
    *,
    model_params: Mapping[str, object] | None = None,
    train_window_mode: str | None = None,
    train_window_size: int | None = None,
    train_window_unit: str = "dates",
    fit_target_col: str | None = None,
    eval_target_col: str | None = None,
    score_postprocess_method: str = "none",
    score_postprocess_columns: list[str] | None = None,
    score_postprocess_strength: float = 1.0,
    score_postprocess_min_obs: int | None = None,
    cv_purge_mode: str = "gap",
    label_horizon_mode: str = "fixed",
    label_horizon_days: int | None = None,
    label_shift_days: int = 0,
    all_trade_dates: object | None = None,
    next_rebalance_map: Mapping[object, object] | None = None,
):
    resolved_type, resolved_params = _resolve_cv_model_spec(model_cfg, model_params)
    fit_target = fit_target_col or target_col
    eval_target = eval_target_col or target_col
    slices = _prepare_cv_date_slices(data, date_col)
    if slices.dates.size == 0:
        return []

    gap = max(int(embargo_days), int(purge_days))
    purge_mode = _validate_cv_purge_mode(cv_purge_mode)
    event_windows, event_window_status = _event_window_state(
        dates=slices.dates,
        purge_mode=purge_mode,
        all_trade_dates=all_trade_dates,
        label_horizon_mode=label_horizon_mode,
        label_horizon_days=label_horizon_days,
        label_shift_days=label_shift_days,
        next_rebalance_map=next_rebalance_map,
    )
    fit_config = _CVFitConfig(
        model_type=resolved_type,
        model_params=resolved_params,
        features=features,
        fit_target=fit_target,
        eval_target=eval_target,
        date_col=date_col,
        signal_direction=signal_direction,
        sample_weight_mode=sample_weight_mode,
        sample_weight_params=sample_weight_params,
        score_postprocess_method=score_postprocess_method,
        score_postprocess_columns=score_postprocess_columns,
        score_postprocess_strength=score_postprocess_strength,
        score_postprocess_min_obs=score_postprocess_min_obs,
    )

    scores = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(slices.dates):
        train_idx = _purged_cv_train_indices(
            train_idx,
            val_idx,
            dates=slices.dates,
            purge_mode=purge_mode,
            event_window_status=event_window_status,
            event_windows=event_windows,
            embargo_days=embargo_days,
            gap=gap,
        )
        if train_idx is None:
            continue
        train_idx = _windowed_cv_train_indices(
            train_idx,
            dates=slices.dates,
            train_window_mode=train_window_mode,
            train_window_size=train_window_size,
            train_window_unit=train_window_unit,
        )
        if train_idx is None:
            continue
        scores.append(_score_cv_fold(slices, train_idx, val_idx, fit_config))
    return scores
