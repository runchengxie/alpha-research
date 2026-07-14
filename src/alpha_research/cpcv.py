from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmarking import build_benchmark_series
from .date_slices import _apply_model_train_window, _slice_trade_dates
from .metrics import (
    daily_ic_series,
    estimate_turnover,
    quantile_returns,
    summarize_active_returns,
    summarize_ic,
    topk_positive_ratio,
)
from .modeling import build_model, fit_model
from .rebalance_calendar import get_rebalance_dates
from .return_metrics import summarize_period_returns
from .split import build_sample_weight, time_series_cv_ic
from .transform import apply_score_postprocess

logger = logging.getLogger("alpha_research")


@dataclass(frozen=True)
class LabelEventWindow:
    signal_date: pd.Timestamp
    label_start: pd.Timestamp
    label_end: pd.Timestamp


@dataclass(frozen=True)
class CPCVSplit:
    split_id: int
    test_groups: tuple[int, ...]
    train_groups: tuple[int, ...]
    train_dates_raw: tuple[pd.Timestamp, ...]
    train_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]
    purged_train_dates: tuple[pd.Timestamp, ...]
    embargoed_train_dates: tuple[pd.Timestamp, ...]
    purge_mode: str
    status: str = "ok"


@dataclass(frozen=True)
class _SplitFrames:
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class _SplitEvalMetrics:
    scored: pd.DataFrame
    ic_stats: dict[str, Any]
    pearson_ic_stats: dict[str, Any]
    long_short: float
    turnover_mean: float
    topk_positive_ratio: dict[str, Any]


@dataclass(frozen=True)
class _SplitBacktestMetrics:
    bt_stats: dict[str, Any] | None
    active_stats: dict[str, Any] | None
    net_series: pd.Series
    gross_series: pd.Series
    turnover_series: pd.Series
    benchmark_series: pd.Series
    active_series: pd.Series
    period_info: list[dict[str, Any]]


def _date_key(date: Any) -> pd.Timestamp:
    return pd.Timestamp(date).normalize()


def _as_date_tuple(dates: Any) -> tuple[pd.Timestamp, ...]:
    values = pd.to_datetime(
        list(dates) if not isinstance(dates, pd.Series) else dates, errors="coerce"
    )
    cleaned = [pd.Timestamp(date).normalize() for date in values if not pd.isna(date)]
    return tuple(pd.Index(cleaned).drop_duplicates().sort_values())


def _format_date(date: Any) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def _format_dates(dates: tuple[pd.Timestamp, ...] | list[pd.Timestamp]) -> str:
    return "|".join(_format_date(date) for date in dates)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return _format_date(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def assign_cpcv_groups(dates: Any, n_groups: int) -> dict[int, tuple[pd.Timestamp, ...]]:
    date_values = _as_date_tuple(dates)
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2.")
    if len(date_values) < n_groups:
        raise ValueError("n_groups cannot exceed the number of eligible dates.")

    groups: dict[int, tuple[pd.Timestamp, ...]] = {}
    base_size, extra = divmod(len(date_values), n_groups)
    cursor = 0
    for group_id in range(n_groups):
        size = base_size + (1 if group_id < extra else 0)
        groups[group_id] = date_values[cursor : cursor + size]
        cursor += size
    return groups


def expected_cpcv_path_count(n_groups: int, test_groups: int) -> int:
    if test_groups < 1 or test_groups >= n_groups:
        raise ValueError("test_groups must satisfy 1 <= test_groups < n_groups.")
    return comb(n_groups - 1, test_groups - 1)


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


def build_label_event_windows(
    signal_dates: Any,
    *,
    all_trade_dates: Any,
    horizon_mode: str,
    horizon_days: int,
    shift_days: int,
    next_rebalance_map: dict[Any, Any] | None = None,
) -> tuple[dict[pd.Timestamp, LabelEventWindow], str]:
    dates = _as_date_tuple(signal_dates)
    trade_dates = _as_date_tuple(all_trade_dates)
    if not dates or not trade_dates:
        return {}, "fallback_gap"

    mode = str(horizon_mode or "fixed").strip().lower()
    next_map = {
        _date_key(key): _date_key(value)
        for key, value in (next_rebalance_map or {}).items()
        if not pd.isna(pd.to_datetime(key, errors="coerce"))
        and not pd.isna(pd.to_datetime(value, errors="coerce"))
    }
    windows: dict[pd.Timestamp, LabelEventWindow] = {}
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
                signal_date, trade_dates, int(horizon_days) + int(shift_days)
            )
        if label_end is None:
            continue
        if label_end < label_start:
            label_start, label_end = label_end, label_start
        windows[signal_date] = LabelEventWindow(
            signal_date=signal_date,
            label_start=label_start,
            label_end=label_end,
        )
    purge_mode = "event_window" if len(windows) == len(dates) else "fallback_gap"
    return windows, purge_mode


def _intervals_overlap(left: LabelEventWindow, right: LabelEventWindow) -> bool:
    return left.label_start <= right.label_end and right.label_start <= left.label_end


def _apply_event_purge(
    train_dates: tuple[pd.Timestamp, ...],
    test_dates: tuple[pd.Timestamp, ...],
    event_windows: dict[pd.Timestamp, LabelEventWindow],
    *,
    embargo_days: int,
) -> tuple[tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...], str]:
    test_windows = [event_windows[date] for date in test_dates if date in event_windows]
    if len(test_windows) != len(test_dates):
        return train_dates, (), (), "fallback_gap"

    purged: list[pd.Timestamp] = []
    embargoed: list[pd.Timestamp] = []
    kept: list[pd.Timestamp] = []
    embargo_delta = pd.Timedelta(days=max(0, int(embargo_days)))
    for train_date in train_dates:
        train_window = event_windows.get(train_date)
        if train_window is None:
            purged.append(train_date)
            continue
        if any(_intervals_overlap(train_window, test_window) for test_window in test_windows):
            purged.append(train_date)
            continue
        if embargo_delta > pd.Timedelta(0) and any(
            test_window.label_end
            < train_window.signal_date
            <= test_window.label_end + embargo_delta
            for test_window in test_windows
        ):
            embargoed.append(train_date)
            continue
        kept.append(train_date)
    return tuple(kept), tuple(purged), tuple(embargoed), "event_window"


def _apply_gap_purge(
    train_dates: tuple[pd.Timestamp, ...],
    test_dates: tuple[pd.Timestamp, ...],
    all_dates: tuple[pd.Timestamp, ...],
    *,
    gap_steps: int,
) -> tuple[tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...]]:
    if gap_steps <= 0:
        return train_dates, (), ()
    positions = {date: idx for idx, date in enumerate(all_dates)}
    test_positions = [positions[date] for date in test_dates if date in positions]
    if not test_positions:
        return train_dates, (), ()
    min_test = min(test_positions)
    max_test = max(test_positions)
    purged: list[pd.Timestamp] = []
    kept: list[pd.Timestamp] = []
    for train_date in train_dates:
        pos = positions.get(train_date)
        if pos is None:
            purged.append(train_date)
            continue
        if min_test - gap_steps <= pos <= max_test + gap_steps:
            purged.append(train_date)
            continue
        kept.append(train_date)
    return tuple(kept), tuple(purged), ()


def build_cpcv_splits(
    dates: Any,
    *,
    n_groups: int,
    test_groups: int,
    event_windows: dict[pd.Timestamp, LabelEventWindow] | None = None,
    embargo_days: int = 0,
    fallback_gap_steps: int = 0,
    min_train_dates: int = 1,
    min_test_dates: int = 1,
) -> tuple[dict[int, tuple[pd.Timestamp, ...]], list[CPCVSplit]]:
    if test_groups < 1 or test_groups >= n_groups:
        raise ValueError("test_groups must satisfy 1 <= test_groups < n_groups.")
    all_dates = _as_date_tuple(dates)
    groups = assign_cpcv_groups(all_dates, n_groups)
    group_ids = tuple(groups)
    splits: list[CPCVSplit] = []
    for split_id, test_group_tuple in enumerate(combinations(group_ids, test_groups), start=1):
        train_group_tuple = tuple(group for group in group_ids if group not in test_group_tuple)
        test_dates = tuple(date for group in test_group_tuple for date in groups[group])
        train_dates_raw = tuple(date for group in train_group_tuple for date in groups[group])
        if event_windows:
            train_dates, purged, embargoed, purge_mode = _apply_event_purge(
                train_dates_raw,
                test_dates,
                event_windows,
                embargo_days=embargo_days,
            )
            if purge_mode == "fallback_gap":
                train_dates, purged, embargoed = _apply_gap_purge(
                    train_dates_raw,
                    test_dates,
                    all_dates,
                    gap_steps=fallback_gap_steps,
                )
        else:
            train_dates, purged, embargoed = _apply_gap_purge(
                train_dates_raw,
                test_dates,
                all_dates,
                gap_steps=fallback_gap_steps,
            )
            purge_mode = "fallback_gap" if fallback_gap_steps > 0 else "none"
        status = (
            "ok"
            if len(train_dates) >= min_train_dates and len(test_dates) >= min_test_dates
            else "insufficient_data"
        )
        splits.append(
            CPCVSplit(
                split_id=split_id,
                test_groups=tuple(int(group) for group in test_group_tuple),
                train_groups=tuple(int(group) for group in train_group_tuple),
                train_dates_raw=train_dates_raw,
                train_dates=tuple(sorted(train_dates)),
                test_dates=tuple(sorted(test_dates)),
                purged_train_dates=tuple(sorted(purged)),
                embargoed_train_dates=tuple(sorted(embargoed)),
                purge_mode=purge_mode,
                status=status,
            )
        )
    return groups, splits


def build_cpcv_paths(
    valid_splits: list[CPCVSplit],
    *,
    n_groups: int,
    test_groups: int,
) -> list[list[CPCVSplit]]:
    path_count = expected_cpcv_path_count(n_groups, test_groups)
    paths: list[list[CPCVSplit]] = [[] for _ in range(path_count)]
    covered: list[set[int]] = [set() for _ in range(path_count)]
    for split in sorted(valid_splits, key=lambda item: (item.test_groups, item.split_id)):
        for path_idx, group_set in enumerate(covered):
            if all(group not in group_set for group in split.test_groups):
                paths[path_idx].append(split)
                group_set.update(split.test_groups)
                break
    return paths


def _series_stat(values: list[float], op: str) -> float | None:
    clean = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)], dtype=float
    )
    if clean.size == 0:
        return None
    if op == "mean":
        return float(np.mean(clean))
    if op == "median":
        return float(np.median(clean))
    if op == "p25":
        return float(np.percentile(clean, 25))
    if op == "p10":
        return float(np.percentile(clean, 10))
    if op == "min":
        return float(np.min(clean))
    if op == "positive_ratio":
        return float(np.mean(clean > 0))
    return None


def _summarize_cpcv(path_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    sharpe_values = [float(row["sharpe"]) for row in path_metrics if pd.notna(row.get("sharpe"))]
    ic_values = [float(row["ic_mean"]) for row in path_metrics if pd.notna(row.get("ic_mean"))]
    long_short_values = [
        float(row["long_short"]) for row in path_metrics if pd.notna(row.get("long_short"))
    ]
    drawdown_values = [
        abs(float(row["max_drawdown"])) for row in path_metrics if pd.notna(row.get("max_drawdown"))
    ]
    turnover_values = [
        float(row["avg_turnover"]) for row in path_metrics if pd.notna(row.get("avg_turnover"))
    ]
    cost_values = [
        float(row["avg_cost_drag"]) for row in path_metrics if pd.notna(row.get("avg_cost_drag"))
    ]
    return {
        "valid_path_count": len(path_metrics),
        "sharpe_mean": _series_stat(sharpe_values, "mean"),
        "sharpe_median": _series_stat(sharpe_values, "median"),
        "sharpe_p25": _series_stat(sharpe_values, "p25"),
        "sharpe_p10": _series_stat(sharpe_values, "p10"),
        "sharpe_min": _series_stat(sharpe_values, "min"),
        "positive_sharpe_ratio": _series_stat(sharpe_values, "positive_ratio"),
        "ic_median": _series_stat(ic_values, "median"),
        "long_short_median": _series_stat(long_short_values, "median"),
        "max_drawdown_p10": _series_stat(drawdown_values, "p10"),
        "turnover_median": _series_stat(turnover_values, "median"),
        "cost_drag_median": _series_stat(cost_values, "median"),
    }


def _collapse_series_by_date(series: pd.Series) -> pd.Series:
    if series.empty or series.index.is_unique:
        return series
    return series.groupby(level=0).mean().sort_index()


def _frame_for_dates(request_data: Any, dates: tuple[pd.Timestamp, ...]) -> pd.DataFrame:
    return _slice_trade_dates(
        request_data.df_model_sorted,
        request_data.all_date_start_rows,
        request_data.all_date_end_rows,
        request_data.all_date_to_pos,
        dates,
    )


def _score_frame(
    frame: pd.DataFrame,
    model: Any,
    *,
    features: list[str],
    signal_direction: float,
    backtest_signal_direction: float,
    score_postprocess_method: str,
    score_postprocess_columns: list[str] | None,
    score_postprocess_strength: float,
    score_postprocess_min_obs: int | None,
) -> pd.DataFrame:
    scored = frame.copy()
    scored["pred"] = model.predict(scored[features])
    if score_postprocess_method != "none":
        scored["pred"] = apply_score_postprocess(
            scored,
            "pred",
            method=score_postprocess_method,
            columns=score_postprocess_columns or [],
            strength=score_postprocess_strength,
            min_obs=score_postprocess_min_obs,
        )
    scored["signal_eval"] = scored["pred"] * signal_direction
    scored["signal_backtest"] = scored["pred"] * backtest_signal_direction
    return scored


def _sample_rebalance_frame(
    frame: pd.DataFrame,
    *,
    frequency: str,
    valid_dates: set[pd.Timestamp] | None = None,
    allowed_dates: tuple[pd.Timestamp, ...] | None = None,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    if frame.empty:
        return frame.copy(), []
    dates = sorted(pd.to_datetime(frame["trade_date"].unique()))
    rebalance_dates = get_rebalance_dates(dates, frequency)
    if valid_dates:
        rebalance_dates = [date for date in rebalance_dates if date in valid_dates]
    if allowed_dates is not None:
        allowed = set(allowed_dates)
        rebalance_dates = [date for date in rebalance_dates if date in allowed]
    sampled = frame[frame["trade_date"].isin(rebalance_dates)].copy()
    return sampled, rebalance_dates


def _prepare_split_frames(request: Any, split: CPCVSplit) -> _SplitFrames | None:
    data = request.data
    model_settings = request.model
    train_dates = _apply_model_train_window(
        split.train_dates,
        label=f"cpcv split {split.split_id}",
        train_window_mode=model_settings.train_window_mode,
        train_window_size=model_settings.train_window_size,
        train_window_unit=model_settings.train_window_unit or "dates",
    )
    train_df = _frame_for_dates(data, tuple(pd.to_datetime(train_dates)))
    test_df = _frame_for_dates(data, split.test_dates)
    if train_df.empty or test_df.empty:
        return None
    return _SplitFrames(train=train_df, test=test_df)


def _resolve_cv_signal_direction(request: Any, train_df: pd.DataFrame) -> float:
    data = request.data
    feature_target = request.feature_target
    model_settings = request.model
    signal_settings = request.signal
    period_settings = request.period
    backtest_settings = request.backtest
    direction = float(signal_settings.signal_direction)
    if signal_settings.signal_direction_mode != "cv_ic":
        return direction
    cv_scores = time_series_cv_ic(
        train_df,
        feature_target.features,
        feature_target.target,
        model_settings.n_splits,
        model_settings.embargo_steps,
        model_settings.purge_steps,
        model_settings.model_cfg,
        1.0,
        sample_weight_mode=model_settings.sample_weight_mode,
        sample_weight_params=model_settings.sample_weight_params,
        train_window_mode=model_settings.train_window_mode,
        train_window_size=model_settings.train_window_size,
        train_window_unit=model_settings.train_window_unit or "dates",
        fit_target_col=feature_target.train_target,
        cv_purge_mode=model_settings.cv_purge_mode,
        label_horizon_mode=period_settings.label_horizon_mode,
        label_horizon_days=int(period_settings.label_horizon_effective),
        label_shift_days=backtest_settings.label_shift_days,
        all_trade_dates=data.all_dates,
    )
    if not cv_scores:
        return direction
    cv_mean = float(np.nanmean(cv_scores))
    if np.isfinite(cv_mean) and cv_mean != 0 and abs(cv_mean) >= signal_settings.min_abs_ic_to_flip:
        return float(np.sign(cv_mean))
    return direction


def _fit_split_model(request: Any, train_df: pd.DataFrame) -> Any:
    feature_target = request.feature_target
    model_settings = request.model
    model = build_model(model_settings.model_type, model_settings.model_params)
    sample_weight = build_sample_weight(
        train_df,
        model_settings.sample_weight_mode,
        params=model_settings.sample_weight_params,
    )
    fit_model(
        model,
        model_settings.model_type,
        train_df,
        features=feature_target.features,
        target_col=feature_target.train_target,
        sample_weight=sample_weight,
    )
    return model


def _resolve_train_ic_signal_direction(
    request: Any,
    train_df: pd.DataFrame,
    model: Any,
    direction: float,
) -> float:
    feature_target = request.feature_target
    signal_settings = request.signal
    if signal_settings.signal_direction_mode == "train_ic":
        train_eval = _score_frame(
            train_df,
            model,
            features=feature_target.features,
            signal_direction=1.0,
            backtest_signal_direction=1.0,
            score_postprocess_method=signal_settings.score_postprocess_method,
            score_postprocess_columns=signal_settings.score_postprocess_columns,
            score_postprocess_strength=signal_settings.score_postprocess_strength,
            score_postprocess_min_obs=signal_settings.score_postprocess_min_obs,
        )
        train_ic = summarize_ic(daily_ic_series(train_eval, feature_target.target, "pred"))
        raw_mean = train_ic.get("mean", np.nan)
        return float(np.sign(raw_mean)) if np.isfinite(raw_mean) and raw_mean != 0 else 1.0
    return direction


def _backtest_direction(request: Any, direction: float) -> float:
    backtest_settings = request.backtest
    if backtest_settings.backtest_signal_direction_raw is None:
        return direction
    return float(backtest_settings.backtest_signal_direction_raw)


def _score_with_request(
    frame: pd.DataFrame,
    model: Any,
    request: Any,
    *,
    direction: float,
    backtest_direction: float,
) -> pd.DataFrame:
    feature_target = request.feature_target
    signal_settings = request.signal
    return _score_frame(
        frame,
        model,
        features=feature_target.features,
        signal_direction=direction,
        backtest_signal_direction=backtest_direction,
        score_postprocess_method=signal_settings.score_postprocess_method,
        score_postprocess_columns=signal_settings.score_postprocess_columns,
        score_postprocess_strength=signal_settings.score_postprocess_strength,
        score_postprocess_min_obs=signal_settings.score_postprocess_min_obs,
    )


def _evaluate_split_eval(
    request: Any,
    split: CPCVSplit,
    test_df: pd.DataFrame,
    model: Any,
    *,
    direction: float,
    backtest_direction: float,
) -> _SplitEvalMetrics:
    data = request.data
    feature_target = request.feature_target
    period_settings = request.period
    scored_test = _score_with_request(
        test_df,
        model,
        request,
        direction=direction,
        backtest_direction=backtest_direction,
    )
    allowed_dates = split.test_dates if period_settings.sample_on_rebalance_dates else None
    eval_df, eval_rebalance_dates = _sample_rebalance_frame(
        scored_test,
        frequency=period_settings.rebalance_frequency,
        valid_dates=data.valid_dates_set,
        allowed_dates=allowed_dates,
    )
    ic_stats = summarize_ic(daily_ic_series(eval_df, feature_target.target, "signal_eval"))
    pearson_ic_stats = summarize_ic(
        daily_ic_series(eval_df, feature_target.target, "signal_eval", method="pearson")
    )
    quantile_ts = quantile_returns(
        eval_df,
        "signal_eval",
        feature_target.target,
        period_settings.n_quantiles,
    )
    quantile_mean = quantile_ts.mean() if not quantile_ts.empty else pd.Series(dtype=float)
    long_short = (
        float(quantile_mean.iloc[-1] - quantile_mean.iloc[0]) if not quantile_mean.empty else np.nan
    )
    k = min(period_settings.top_k, eval_df["symbol"].nunique()) if not eval_df.empty else 0
    turnover = pd.Series(dtype=float, name="turnover")
    if k > 0 and eval_rebalance_dates:
        turnover = estimate_turnover(
            eval_df,
            "signal_eval",
            k,
            eval_rebalance_dates,
            buffer_exit=period_settings.eval_buffer_exit,
            buffer_entry=period_settings.eval_buffer_entry,
        )
    return _SplitEvalMetrics(
        scored=eval_df,
        ic_stats=ic_stats,
        pearson_ic_stats=pearson_ic_stats,
        long_short=long_short,
        turnover_mean=float(turnover.mean()) if not turnover.empty else np.nan,
        topk_positive_ratio=topk_positive_ratio(
            eval_df,
            "signal_eval",
            feature_target.target,
            k,
        ),
    )


def _empty_split_backtest_metrics() -> _SplitBacktestMetrics:
    return _SplitBacktestMetrics(
        bt_stats=None,
        active_stats=None,
        net_series=pd.Series(dtype=float, name="net_return"),
        gross_series=pd.Series(dtype=float, name="gross_return"),
        turnover_series=pd.Series(dtype=float, name="turnover"),
        benchmark_series=pd.Series(dtype=float, name="benchmark_return"),
        active_series=pd.Series(dtype=float, name="active_return"),
        period_info=[],
    )


def _evaluate_split_backtest(
    request: Any,
    split: CPCVSplit,
    model: Any,
    *,
    direction: float,
    backtest_direction: float,
) -> _SplitBacktestMetrics:
    data = request.data
    feature_target = request.feature_target
    backtest_settings = request.backtest
    services = request.services
    result = _empty_split_backtest_metrics()
    if not backtest_settings.backtest_enabled:
        return result

    test_start = min(split.test_dates)
    test_end = max(split.test_dates)
    test_full = data.df_full[
        (data.df_full["trade_date"] >= test_start) & (data.df_full["trade_date"] <= test_end)
    ].copy()
    if test_full.empty:
        return result

    scored_full = _score_with_request(
        test_full,
        model,
        request,
        direction=direction,
        backtest_direction=backtest_direction,
    )
    bt_rebalance_dates = get_rebalance_dates(
        sorted(scored_full["trade_date"].unique()),
        backtest_settings.backtest_rebalance_frequency,
    )
    if data.valid_dates_set:
        bt_rebalance_dates = [date for date in bt_rebalance_dates if date in data.valid_dates_set]
    try:
        bt_result = services.backtest_topk_fn(
            scored_full,
            pred_col="signal_backtest",
            price_col=feature_target.price_col,
            rebalance_dates=bt_rebalance_dates,
            top_k=backtest_settings.backtest_top_k,
            shift_days=backtest_settings.label_shift_days,
            cost_bps=backtest_settings.backtest_cost_bps_effective,
            trading_days_per_year=backtest_settings.backtest_trading_days_per_year,
            exit_mode=backtest_settings.backtest_exit_mode,
            exit_horizon_days=backtest_settings.backtest_exit_horizon_days,
            long_only=backtest_settings.backtest_long_only,
            short_k=backtest_settings.backtest_short_k,
            weighting=backtest_settings.backtest_weighting,
            buffer_exit=backtest_settings.backtest_buffer_exit,
            buffer_entry=backtest_settings.backtest_buffer_entry,
            group_col=backtest_settings.backtest_group_col
            if backtest_settings.backtest_group_col in scored_full.columns
            else None,
            max_names_per_group=backtest_settings.backtest_max_names_per_group,
            tradable_col=backtest_settings.backtest_tradable_col
            if backtest_settings.backtest_tradable_col in data.backtest_pricing_df.columns
            else None,
            exit_price_policy=backtest_settings.backtest_exit_price_policy,
            exit_fallback_policy=backtest_settings.backtest_exit_fallback_policy,
            execution=backtest_settings.execution_model,
            pricing_data=data.backtest_pricing_df,
        )
    except ValueError:
        return result
    if bt_result is None:
        return result

    bt_stats, net_series, gross_series, turnover_series, period_info = bt_result
    benchmark_series, _benchmark_periods = build_benchmark_series(
        data.benchmark_df,
        backtest_settings.execution_model.entry_policy.price_col,
        backtest_settings.execution_model.exit_policy.price_col,
        period_info,
        benchmark_return_series=data.benchmark_return_series,
    )
    active_stats = None
    active_series = result.active_series
    if not benchmark_series.empty:
        active_stats, active_series = summarize_active_returns(
            net_series,
            benchmark_series,
            bt_stats.get("periods_per_year", np.nan),
        )
    return _SplitBacktestMetrics(
        bt_stats=bt_stats,
        active_stats=active_stats,
        net_series=net_series,
        gross_series=gross_series,
        turnover_series=turnover_series,
        benchmark_series=benchmark_series,
        active_series=active_series,
        period_info=period_info,
    )


def _evaluate_split(context: dict[str, Any], split: CPCVSplit) -> dict[str, Any]:
    request = context["train_eval_request"]
    if split.status != "ok":
        return {"status": split.status, "split": split}

    frames = _prepare_split_frames(request, split)
    if frames is None:
        return {"status": "insufficient_data", "split": split}

    direction = _resolve_cv_signal_direction(request, frames.train)
    model = _fit_split_model(request, frames.train)
    direction = _resolve_train_ic_signal_direction(request, frames.train, model, direction)
    backtest_direction = _backtest_direction(request, direction)
    eval_metrics = _evaluate_split_eval(
        request,
        split,
        frames.test,
        model,
        direction=direction,
        backtest_direction=backtest_direction,
    )
    backtest_metrics = _evaluate_split_backtest(
        request,
        split,
        model,
        direction=direction,
        backtest_direction=backtest_direction,
    )

    return {
        "status": "ok",
        "split": split,
        "direction": direction,
        "eval_scored": eval_metrics.scored,
        "ic_stats": eval_metrics.ic_stats,
        "pearson_ic_stats": eval_metrics.pearson_ic_stats,
        "long_short": eval_metrics.long_short,
        "turnover_mean": eval_metrics.turnover_mean,
        "topk_positive_ratio": eval_metrics.topk_positive_ratio,
        "bt_stats": backtest_metrics.bt_stats,
        "active_stats": backtest_metrics.active_stats,
        "net_series": backtest_metrics.net_series,
        "gross_series": backtest_metrics.gross_series,
        "turnover_series": backtest_metrics.turnover_series,
        "benchmark_series": backtest_metrics.benchmark_series,
        "active_series": backtest_metrics.active_series,
        "period_info": backtest_metrics.period_info,
    }


def _split_to_row(split: CPCVSplit) -> dict[str, Any]:
    return {
        "split_id": split.split_id,
        "test_groups": "|".join(str(group) for group in split.test_groups),
        "train_groups": "|".join(str(group) for group in split.train_groups),
        "train_start": _format_date(split.train_dates[0]) if split.train_dates else None,
        "train_end": _format_date(split.train_dates[-1]) if split.train_dates else None,
        "test_start": _format_date(split.test_dates[0]) if split.test_dates else None,
        "test_end": _format_date(split.test_dates[-1]) if split.test_dates else None,
        "train_dates_raw": len(split.train_dates_raw),
        "train_dates": len(split.train_dates),
        "test_dates": len(split.test_dates),
        "purged_train_dates": len(split.purged_train_dates),
        "embargoed_train_dates": len(split.embargoed_train_dates),
        "purge_mode": split.purge_mode,
        "status": split.status,
    }


def _path_metric_row(
    path_id: int,
    split_results: list[dict[str, Any]],
    *,
    target_col: str,
    n_quantiles: int,
    trading_days_per_year: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not split_results:
        return None, []
    split_ids = [int(item["split"].split_id) for item in split_results]
    test_dates = sorted({date for item in split_results for date in item["split"].test_dates})
    eval_frames = [item["eval_scored"] for item in split_results if not item["eval_scored"].empty]
    eval_frame = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    net_series = _collapse_series_by_date(
        pd.concat([item["net_series"] for item in split_results]).sort_index()
    )
    gross_series = _collapse_series_by_date(
        pd.concat([item["gross_series"] for item in split_results]).sort_index()
    )
    turnover_series = _collapse_series_by_date(
        pd.concat([item["turnover_series"] for item in split_results]).sort_index()
    )
    benchmark_series = _collapse_series_by_date(
        pd.concat([item["benchmark_series"] for item in split_results]).sort_index()
    )
    active_series = _collapse_series_by_date(
        pd.concat([item["active_series"] for item in split_results]).sort_index()
    )
    period_info = [period for item in split_results for period in item["period_info"]]
    period_info = sorted(period_info, key=lambda item: item.get("exit_date"))
    stats = summarize_period_returns(net_series, period_info, trading_days_per_year)

    if not eval_frame.empty:
        if target_col not in eval_frame.columns:
            candidates = [col for col in eval_frame.columns if col.endswith("return")]
            target_col = candidates[0] if candidates else "future_return"
        ic_stats = summarize_ic(daily_ic_series(eval_frame, target_col, "signal_eval"))
        q = quantile_returns(eval_frame, "signal_eval", target_col, n_quantiles)
        q_mean = q.mean() if not q.empty else pd.Series(dtype=float)
        long_short = float(q_mean.iloc[-1] - q_mean.iloc[0]) if not q_mean.empty else np.nan
    else:
        ic_stats = {}
        long_short = np.nan
    active_stats = None
    if not active_series.empty and not benchmark_series.empty:
        active_stats, _ = summarize_active_returns(
            net_series,
            benchmark_series,
            stats.get("periods_per_year", np.nan),
        )
    row = {
        "path_id": path_id,
        "split_ids": "|".join(str(split_id) for split_id in split_ids),
        "test_start": _format_date(test_dates[0]) if test_dates else None,
        "test_end": _format_date(test_dates[-1]) if test_dates else None,
        "observation_count": len(net_series) if not net_series.empty else len(eval_frame),
        "sharpe": stats.get("sharpe"),
        "total_return": stats.get("total_return"),
        "ann_return": stats.get("ann_return"),
        "ann_vol": stats.get("ann_vol"),
        "max_drawdown": stats.get("max_drawdown"),
        "ic_mean": ic_stats.get("mean"),
        "ic_ir": ic_stats.get("ir"),
        "long_short": long_short,
        "avg_turnover": float(turnover_series.mean()) if not turnover_series.empty else np.nan,
        "avg_cost_drag": np.nan,
        "active_total_return": (active_stats or {}).get("active_total_return"),
        "information_ratio": (active_stats or {}).get("information_ratio"),
        "tracking_error": (active_stats or {}).get("tracking_error"),
    }
    split_costs = [
        item["bt_stats"].get("avg_cost_drag")
        for item in split_results
        if item.get("bt_stats") and pd.notna(item["bt_stats"].get("avg_cost_drag"))
    ]
    if split_costs:
        row["avg_cost_drag"] = float(np.mean(split_costs))

    return_rows: list[dict[str, Any]] = []
    index_values = sorted(
        set(net_series.index).union(gross_series.index).union(benchmark_series.index)
    )
    for date in index_values:
        net_value = net_series.get(date, np.nan)
        gross_value = gross_series.get(date, np.nan)
        benchmark_value = benchmark_series.get(date, np.nan)
        return_rows.append(
            {
                "path_id": path_id,
                "date": _format_date(date),
                "net_return": float(net_value) if pd.notna(net_value) else np.nan,
                "gross_return": float(gross_value) if pd.notna(gross_value) else np.nan,
                "benchmark_return": float(benchmark_value) if pd.notna(benchmark_value) else np.nan,
                "active_return": float(net_value - benchmark_value)
                if pd.notna(net_value) and pd.notna(benchmark_value)
                else np.nan,
            }
        )
    return row, return_rows


def run_cpcv_audit(
    context: dict[str, Any],
    *,
    n_groups: int,
    test_groups: int,
    embargo_days: int | None,
    include_final_oos: bool,
    out_dir: Path,
) -> dict[str, Any]:
    from .cpcv_audit import run_cpcv_audit as _run_cpcv_audit

    return _run_cpcv_audit(
        context,
        n_groups=n_groups,
        test_groups=test_groups,
        embargo_days=embargo_days,
        include_final_oos=include_final_oos,
        out_dir=out_dir,
    )


def _default_out_dir(config_ref: str | Path | None) -> Path:
    tag = "default" if config_ref is None else Path(str(config_ref)).stem.replace(".", "_")
    return Path("artifacts") / "reports" / f"cpcv_{tag}"


def add_cpcv_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        default=None,
        help="Pipeline config path, artifact_cpcv config path, or built-in name.",
    )
    parser.add_argument(
        "--n-groups", type=int, default=8, help="Number of chronological CPCV groups."
    )
    parser.add_argument(
        "--test-groups", type=int, default=2, help="Number of groups tested per split."
    )
    parser.add_argument(
        "--embargo-days", type=int, default=None, help="Optional CPCV embargo days override."
    )
    parser.add_argument("--out", default=None, help="Output report directory.")
    parser.add_argument(
        "--include-final-oos",
        action="store_true",
        help="Include final OOS dates in the CPCV audit instead of reserving them.",
    )
    parser.add_argument(
        "--fail-on-quality",
        choices=["none", "info", "warning", "error"],
        default=None,
        help="Optional quality gate threshold forwarded to pipeline preparation.",
    )
    parser.add_argument("--artifacts-root", default=None, help="Optional artifacts root override.")
    return parser


def run(args: argparse.Namespace) -> int:
    from .artifact_cpcv import is_artifact_cpcv_config, run_artifact_cpcv

    out_dir = Path(args.out).expanduser() if args.out else _default_out_dir(args.config)
    if is_artifact_cpcv_config(args.config):
        summary = run_artifact_cpcv(
            args.config,
            n_groups=args.n_groups,
            test_groups=args.test_groups,
            embargo_days=args.embargo_days,
            out_dir=out_dir,
        )
        print(
            json.dumps(
                _to_jsonable({"output_dir": str(out_dir), **summary}),
                ensure_ascii=True,
                indent=2,
            )
        )
        return 0

    prepare_research_context = getattr(args, "prepare_research_context", None)
    if prepare_research_context is None:
        raise SystemExit(
            "Pipeline-backed CPCV requires a prepare_research_context adapter. "
            "Use the strategy CLI provided by strategy-pipeline or an artifact_cpcv config."
        )
    context = prepare_research_context(
        args.config,
        fail_on_quality=args.fail_on_quality,
        artifacts_root=args.artifacts_root,
    )
    summary = run_cpcv_audit(
        context,
        n_groups=args.n_groups,
        test_groups=args.test_groups,
        embargo_days=args.embargo_days,
        include_final_oos=bool(args.include_final_oos),
        out_dir=out_dir,
    )
    print(
        json.dumps(
            _to_jsonable({"output_dir": str(out_dir), **summary}), ensure_ascii=True, indent=2
        )
    )
    return 0
