from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ._split_windows import (
    _build_label_event_windows,
    _coerce_sample_weight_min,
    _time_decay_weights,
)
from .sample_weighting import SampleWeightConfig, build_event_sample_weights


def _event_sample_weights(
    data: pd.DataFrame,
    *,
    mode: str,
    date_col: str,
    params: Mapping[str, object] | None,
) -> np.ndarray | None:
    if params is not None and not isinstance(params, Mapping):
        raise ValueError("sample_weight_params must be a mapping.")
    params_map = dict(params or {})
    events = data.copy()
    events["_sample_event_id"] = np.arange(len(events), dtype=int)
    start_col = str(params_map.get("start_col", "label_start"))
    end_col = str(params_map.get("end_col", "label_end"))
    all_trade_dates = params_map.get("all_trade_dates")

    if start_col not in events.columns or end_col not in events.columns:
        horizon_days_raw = params_map.get("label_horizon_days")
        horizon_days = None if horizon_days_raw is None else int(horizon_days_raw)
        event_windows, status = _build_label_event_windows(
            events[date_col],
            all_trade_dates=(all_trade_dates if all_trade_dates is not None else events[date_col]),
            horizon_mode=str(params_map.get("label_horizon_mode", "fixed")),
            horizon_days=horizon_days,
            shift_days=int(params_map.get("label_shift_days", 0)),
            next_rebalance_map=(
                params_map.get("next_rebalance_map")
                if isinstance(params_map.get("next_rebalance_map"), Mapping)
                else None
            ),
        )
        normalized_dates = pd.to_datetime(events[date_col], errors="coerce").dt.normalize()
        events[start_col] = normalized_dates.map(
            {date: window.label_start for date, window in event_windows.items()}
        )
        events[end_col] = normalized_dates.map(
            {date: window.label_end for date, window in event_windows.items()}
        )
        if status != "event_window" or events[[start_col, end_col]].isna().any().any():
            if bool(params_map.get("allow_event_window_fallback", False)):
                return _time_decay_weights(data, date_col=date_col, params=params_map)
            raise ValueError(
                "uniqueness sample weights require complete label event windows; "
                "provide label_start/label_end or label horizon parameters with all_trade_dates."
            )

    halflife_raw = params_map.get("halflife", params_map.get("half_life"))
    config = SampleWeightConfig(
        mode=(
            "uniqueness_time_decay"
            if mode in {"uniqueness_time_decay", "uniqueness_decay"}
            else "uniqueness"
        ),
        uniqueness_power=float(params_map.get("uniqueness_power", 1.0)),
        time_decay_halflife=(float(halflife_raw) if halflife_raw is not None else None),
        min_weight=_coerce_sample_weight_min(params_map.get("min_weight", 0.0)),
    )
    group_col_raw = params_map.get("group_col", params_map.get("symbol_col", "symbol"))
    group_col = (
        str(group_col_raw) if group_col_raw and str(group_col_raw) in events.columns else None
    )
    weighted, _receipt = build_event_sample_weights(
        events,
        config=config,
        bar_index=(all_trade_dates if all_trade_dates is not None else events[date_col]),
        event_id_col="_sample_event_id",
        start_col=start_col,
        end_col=end_col,
        group_col=group_col,
    )
    by_id = weighted.set_index("_sample_event_id")["sample_weight"]
    weights = by_id.reindex(events["_sample_event_id"]).to_numpy(dtype=float)

    if bool(params_map.get("equalize_dates", True)):
        series = pd.Series(weights, index=events.index, dtype=float)
        dates = pd.to_datetime(events[date_col], errors="coerce")
        date_totals = series.groupby(dates).transform("sum")
        series = series.div(date_totals.replace(0.0, np.nan)).fillna(0.0)
        mean = float(series.mean())
        if np.isfinite(mean) and mean > 0:
            series = series / mean
        weights = series.to_numpy(dtype=float)
    return weights


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
    if mode_text in {"uniqueness", "uniqueness_time_decay", "uniqueness_decay"}:
        return _event_sample_weights(
            data,
            mode=mode_text,
            date_col=date_col,
            params=params,
        )
    raise ValueError(f"Unsupported sample_weight_mode: {mode}")
