"""CPCV per-split model fit, scoring, and evaluation (private helpers).

Re-exported from ``alpha_research.cpcv`` so existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._cpcv_groups import CPCVSplit
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
from .split import build_sample_weight, time_series_cv_ic
from .transform import apply_score_postprocess


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
