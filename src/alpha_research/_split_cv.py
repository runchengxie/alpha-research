from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ._split_sample_weight import build_sample_weight
from ._split_windows import (
    _apply_event_window_purge_indices,
    _build_label_event_windows,
    _CVDateSlices,
    _LabelEventWindow,
    _prepare_cv_date_slices,
    _validate_cv_purge_mode,
    _windowed_cv_train_indices,
)
from .backends import NativeTrainerBackend, TrainerBackend, TrainerFitRequest
from .metrics import daily_ic_series
from .modeling import resolve_model_spec
from .transform import apply_score_postprocess


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
    trainer_backend: TrainerBackend


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

    sample_weight = build_sample_weight(
        tr_df,
        config.sample_weight_mode,
        date_col=config.date_col,
        params=config.sample_weight_params,
    )
    handle = config.trainer_backend.fit(
        TrainerFitRequest(
            frame=tr_df,
            model_type=config.model_type,
            model_params=config.model_params,
            features=tuple(config.features),
            target_col=config.fit_target,
            sample_weight=sample_weight,
            date_col=config.date_col,
        )
    )
    va_df["pred"] = config.trainer_backend.predict(
        handle,
        va_df,
        features=config.features,
    )
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
    trainer_backend: TrainerBackend | None = None,
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
    effective_weight_params = dict(sample_weight_params or {})
    if sample_weight_mode and str(sample_weight_mode).lower().startswith("uniqueness"):
        effective_weight_params.setdefault("all_trade_dates", all_trade_dates)
        effective_weight_params.setdefault("label_horizon_mode", label_horizon_mode)
        effective_weight_params.setdefault("label_horizon_days", label_horizon_days)
        effective_weight_params.setdefault("label_shift_days", label_shift_days)
        effective_weight_params.setdefault("next_rebalance_map", next_rebalance_map)
    fit_config = _CVFitConfig(
        model_type=resolved_type,
        model_params=resolved_params,
        features=features,
        fit_target=fit_target,
        eval_target=eval_target,
        date_col=date_col,
        signal_direction=signal_direction,
        sample_weight_mode=sample_weight_mode,
        sample_weight_params=effective_weight_params,
        score_postprocess_method=score_postprocess_method,
        score_postprocess_columns=score_postprocess_columns,
        score_postprocess_strength=score_postprocess_strength,
        score_postprocess_min_obs=score_postprocess_min_obs,
        trainer_backend=trainer_backend or NativeTrainerBackend(),
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
