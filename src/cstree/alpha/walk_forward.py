from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from ..contracts.rebalance import _sample_rebalance_frame
from ..metrics import (
    daily_ic_series,
    estimate_turnover,
    hit_rate,
    quantile_returns,
    regression_error_metrics,
    summarize_ic,
    topk_positive_ratio,
)
from ..pipeline.dates import (
    _apply_model_train_window,
    _slice_trade_dates,
)
from .evaluation import _permutation_test_ic, _postprocess_pred_column
from .modeling import build_model, feature_importance_frame, fit_model
from .split import build_sample_weight, time_series_cv_ic

logger = logging.getLogger("cstree")


def _prepare_walk_forward_data(
    window_meta: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> tuple[int, Any, Any, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    window_id = int(window_meta["window"])
    train_dates = _apply_model_train_window(
        window_meta["train_dates"],
        label=f"walk_forward window {window_id}",
        train_window_mode=context["train_window_mode"],
        train_window_size=context["train_window_size"],
        train_window_unit=context["train_window_unit"],
    )
    test_dates = window_meta["test_dates"]
    train_df_w = _slice_trade_dates(
        context["df_model_sorted"],
        context["all_date_start_rows"],
        context["all_date_end_rows"],
        context["all_date_to_pos"],
        train_dates,
    )
    test_df_w = _slice_trade_dates(
        context["df_model_sorted"],
        context["all_date_start_rows"],
        context["all_date_end_rows"],
        context["all_date_to_pos"],
        test_dates,
    )
    result = {
        "window": window_id,
        "train_start": pd.to_datetime(train_dates[0]).strftime("%Y-%m-%d")
        if len(train_dates)
        else pd.to_datetime(window_meta["train_start"]).strftime("%Y-%m-%d"),
        "train_end": pd.to_datetime(train_dates[-1]).strftime("%Y-%m-%d")
        if len(train_dates)
        else pd.to_datetime(window_meta["train_end"]).strftime("%Y-%m-%d"),
        "test_start": pd.to_datetime(window_meta["test_start"]).strftime("%Y-%m-%d"),
        "test_end": pd.to_datetime(window_meta["test_end"]).strftime("%Y-%m-%d"),
        "status": "ok",
    }
    return window_id, train_dates, test_dates, train_df_w, test_df_w, result


def _resolve_walk_forward_cv_direction(
    train_df_w: pd.DataFrame,
    *,
    context: Mapping[str, Any],
    direction: float,
) -> tuple[float, dict[str, Any] | None]:
    if context["signal_direction_mode"] != "cv_ic":
        return direction, None

    cv_scores_w = time_series_cv_ic(
        train_df_w,
        context["features"],
        context["target"],
        context["n_splits"],
        context["embargo_steps"],
        context["purge_steps"],
        context["model_cfg"],
        1.0,
        sample_weight_mode=context["sample_weight_mode"],
        sample_weight_params=context["sample_weight_params"],
        train_window_mode=context["train_window_mode"],
        train_window_size=context["train_window_size"],
        train_window_unit=context["train_window_unit"],
        fit_target_col=context["train_target"],
        cv_purge_mode=context.get("cv_purge_mode", "gap"),
        label_horizon_mode=context.get("label_horizon_mode", "fixed"),
        label_horizon_days=context.get("label_horizon_days"),
        label_shift_days=context.get("label_shift_days", 0),
        all_trade_dates=context.get("all_dates"),
    )
    if not cv_scores_w:
        return direction, None

    cv_mean = float(np.nanmean(cv_scores_w))
    cv_std = float(np.nanstd(cv_scores_w))
    if np.isfinite(cv_mean) and cv_mean != 0 and abs(cv_mean) >= context["min_abs_ic_to_flip"]:
        direction = float(np.sign(cv_mean))
    return direction, {
        "mean": cv_mean,
        "std": cv_std,
        "scores": [float(score) for score in cv_scores_w],
    }


def _fit_walk_forward_model(
    train_df_w: pd.DataFrame,
    *,
    context: Mapping[str, Any],
) -> tuple[Any, pd.DataFrame, str]:
    model_w = build_model(context["model_type"], context["model_params"])
    train_weights_w = build_sample_weight(
        train_df_w,
        context["sample_weight_mode"],
        params=context["sample_weight_params"],
    )
    fit_model(
        model_w,
        context["model_type"],
        train_df_w,
        features=context["features"],
        target_col=context["train_target"],
        sample_weight=train_weights_w,
    )
    importance_df_w, importance_source_w = feature_importance_frame(
        model_w,
        context["features"],
    )
    return model_w, importance_df_w, importance_source_w


def _walk_forward_importance_rows(
    importance_df_w: pd.DataFrame,
    *,
    window_id: int,
    result: Mapping[str, Any],
    importance_source_w: str,
) -> list[dict[str, Any]]:
    importance_rows: list[dict[str, Any]] = []
    if importance_df_w.empty:
        return importance_rows
    for _, row in importance_df_w.iterrows():
        importance_rows.append(
            {
                "window": window_id,
                "train_start": result["train_start"],
                "train_end": result["train_end"],
                "test_start": result["test_start"],
                "test_end": result["test_end"],
                "feature": str(row["feature"]),
                "importance": float(row["importance"]),
                "importance_source": importance_source_w,
            }
        )
    return importance_rows


def _score_walk_forward_frame(
    frame: pd.DataFrame,
    *,
    model_w: Any,
    context: Mapping[str, Any],
) -> pd.DataFrame:
    scored = frame.copy()
    scored["pred"] = model_w.predict(scored[context["features"]])
    _postprocess_pred_column(
        scored,
        "pred",
        method=context["score_postprocess_method"],
        columns=context["score_postprocess_columns"],
        strength=context["score_postprocess_strength"],
        min_obs=context["score_postprocess_min_obs"],
    )
    return scored


def _resolve_walk_forward_train_direction(
    train_eval: pd.DataFrame,
    *,
    context: Mapping[str, Any],
    direction: float,
) -> tuple[float, dict[str, Any] | None]:
    if context["signal_direction_mode"] != "train_ic":
        return direction, None

    train_ic_raw = daily_ic_series(train_eval, context["target"], "pred")
    train_ic_raw_stats = summarize_ic(train_ic_raw)
    raw_mean = train_ic_raw_stats.get("mean", np.nan)
    if np.isfinite(raw_mean) and raw_mean != 0:
        direction = float(np.sign(raw_mean))
    else:
        direction = 1.0
    return direction, train_ic_raw_stats


def _apply_walk_forward_train_signal(
    train_eval: pd.DataFrame,
    *,
    direction: float,
) -> str:
    if direction == 1.0:
        return "pred"
    train_eval["signal"] = train_eval["pred"] * direction
    return "signal"


def _sample_walk_forward_eval_frame(
    test_eval: pd.DataFrame,
    test_dates: Any,
    *,
    context: Mapping[str, Any],
    direction: float,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    test_eval["signal_eval"] = test_eval["pred"] * direction
    eval_allowed_dates_w = test_dates if context["sample_on_rebalance_dates"] else None
    return _sample_rebalance_frame(
        test_eval,
        frequency=context["rebalance_frequency"],
        valid_dates=context["valid_dates_set"],
        allowed_dates=eval_allowed_dates_w,
    )


def _walk_forward_eval_metrics(
    eval_df_w: pd.DataFrame,
    *,
    context: Mapping[str, Any],
    signal_col_w: str,
) -> dict[str, Any]:
    target = context["target"]
    target_values = cast(pd.Series, eval_df_w[target])
    signal_values = cast(pd.Series, eval_df_w[signal_col_w])
    return {
        "test_ic": summarize_ic(daily_ic_series(eval_df_w, target, signal_col_w)),
        "test_pearson_ic": summarize_ic(
            daily_ic_series(eval_df_w, target, signal_col_w, method="pearson")
        ),
        "error_metrics": regression_error_metrics(target_values, signal_values),
        "hit_rate": hit_rate(target_values, signal_values),
    }


def _walk_forward_permutation_stats(
    train_df_w: pd.DataFrame,
    test_df_w: pd.DataFrame,
    rebalance_dates_w: list[pd.Timestamp],
    *,
    context: Mapping[str, Any],
    direction: float,
) -> dict[str, Any] | None:
    if not context["wf_perm_test_enabled"]:
        return None

    perm_scores = _permutation_test_ic(
        train_df_w,
        test_df_w,
        context["wf_perm_test_runs"],
        context["wf_perm_test_seed"],
        direction,
        model_type=context["model_type"],
        model_params=context["model_params"],
        features=context["features"],
        fit_target_col=context["train_target"],
        target_col=context["target"],
        sample_weight_mode=context["sample_weight_mode"],
        sample_weight_params=context["sample_weight_params"],
        eval_dates=rebalance_dates_w,
        score_postprocess_method=context["score_postprocess_method"],
        score_postprocess_columns=context["score_postprocess_columns"],
        score_postprocess_strength=context["score_postprocess_strength"],
        score_postprocess_min_obs=context["score_postprocess_min_obs"],
    )
    if not perm_scores:
        return None
    return {
        "mean": float(np.nanmean(perm_scores)),
        "std": float(np.nanstd(perm_scores)),
        "scores": [float(score) for score in perm_scores],
        "runs": int(len(perm_scores)),
    }


def _walk_forward_portfolio_metrics(
    eval_df_w: pd.DataFrame,
    rebalance_dates_w: list[pd.Timestamp],
    *,
    context: Mapping[str, Any],
    signal_col_w: str,
) -> dict[str, Any]:
    quantile_ts_w = quantile_returns(
        eval_df_w,
        signal_col_w,
        context["target"],
        context["n_quantiles"],
    )
    quantile_mean_w = (
        cast(pd.Series, quantile_ts_w.mean()) if not quantile_ts_w.empty else pd.Series(dtype=float)
    )
    long_short_w = (
        float(quantile_mean_w.iloc[-1] - quantile_mean_w.iloc[0])
        if not quantile_mean_w.empty
        else None
    )

    top_k = int(context["top_k"])
    symbol_count = int(cast(pd.Series, eval_df_w["symbol"]).nunique())
    k_w = min(top_k, symbol_count)
    if k_w > 0 and rebalance_dates_w:
        turnover_series_w = estimate_turnover(
            eval_df_w,
            signal_col_w,
            k_w,
            rebalance_dates_w,
            buffer_exit=context["eval_buffer_exit"],
            buffer_entry=context["eval_buffer_entry"],
        )
    else:
        turnover_series_w = pd.Series(dtype=float, name="turnover")
    turnover_mean_w = float(turnover_series_w.mean()) if not turnover_series_w.empty else None

    return {
        "long_short": long_short_w,
        "turnover_mean": turnover_mean_w,
        "topk_positive_ratio": topk_positive_ratio(
            eval_df_w,
            signal_col_w,
            context["target"],
            k_w,
        ),
    }


def _walk_forward_feature_importance_top(
    importance_df_w: pd.DataFrame,
    *,
    wf_feature_top_k: int,
) -> list[dict[str, Any]]:
    return [
        {"feature": str(item["feature"]), "importance": float(item["importance"])}
        for _, item in importance_df_w.head(wf_feature_top_k).iterrows()
    ]


def _evaluate_injected_walk_forward_backtest(
    window_meta: dict,
    *,
    model_w: Any,
    direction: float,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    if not context["wf_backtest_enabled"]:
        return None, None, None

    evaluator = context.get("walk_forward_backtest_fn")
    backtest_topk_fn = context.get("backtest_topk_fn")
    if evaluator is None or backtest_topk_fn is None:
        raise SystemExit(
            "eval.walk_forward.backtest requires injected walk_forward_backtest_fn "
            "and backtest_topk_fn from the strategy pipeline."
        )
    return evaluator(
        window_meta,
        model_w=model_w,
        direction=direction,
        context=context,
        valid_dates_set=context["valid_dates_set"],
        backtest_topk_fn=backtest_topk_fn,
    )


def _update_walk_forward_result(
    result: dict[str, Any],
    *,
    context: Mapping[str, Any],
    direction: float,
    cv_stats: dict[str, Any] | None,
    train_ic_stats: dict[str, Any],
    train_ic_raw_stats: dict[str, Any] | None,
    eval_metrics: dict[str, Any],
    portfolio_metrics: dict[str, Any],
    backtest_stats: tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None],
    perm_stats_w: dict[str, Any] | None,
    importance_source_w: str,
    importance_df_w: pd.DataFrame,
) -> None:
    bt_stats_w, bt_benchmark_stats_w, bt_active_stats_w = backtest_stats
    result.update(
        {
            "signal_direction": direction,
            "signal_direction_mode": context["signal_direction_mode"],
            "cv_ic": cv_stats,
            "train_ic": train_ic_stats if context["report_train_ic"] else None,
            "train_ic_raw": train_ic_raw_stats,
            **eval_metrics,
            **portfolio_metrics,
            "backtest": {
                "stats": bt_stats_w,
                "benchmark": bt_benchmark_stats_w,
                "active": bt_active_stats_w,
            }
            if context["wf_backtest_enabled"]
            else None,
            "permutation_test": perm_stats_w,
            "feature_importance_source": importance_source_w,
            "feature_importance_top": _walk_forward_feature_importance_top(
                importance_df_w,
                wf_feature_top_k=context["wf_feature_top_k"],
            ),
        }
    )


def _evaluate_walk_forward_window(
    window_meta: dict,
    *,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    (
        window_id,
        _,
        test_dates,
        train_df_w,
        test_df_w,
        result,
    ) = _prepare_walk_forward_data(window_meta, context=context)
    if train_df_w.empty or test_df_w.empty:
        result["status"] = "insufficient_data"
        return result, []

    direction, cv_stats = _resolve_walk_forward_cv_direction(
        train_df_w,
        context=context,
        direction=context["signal_direction"],
    )
    model_w, importance_df_w, importance_source_w = _fit_walk_forward_model(
        train_df_w,
        context=context,
    )
    importance_rows = _walk_forward_importance_rows(
        importance_df_w,
        window_id=window_id,
        result=result,
        importance_source_w=importance_source_w,
    )

    train_eval = _score_walk_forward_frame(train_df_w, model_w=model_w, context=context)
    direction, train_ic_raw_stats = _resolve_walk_forward_train_direction(
        train_eval,
        context=context,
        direction=direction,
    )
    train_signal_col = _apply_walk_forward_train_signal(train_eval, direction=direction)
    train_ic_stats = {}
    if context["report_train_ic"]:
        train_ic_stats = summarize_ic(
            daily_ic_series(train_eval, context["target"], train_signal_col)
        )

    test_eval = _score_walk_forward_frame(test_df_w, model_w=model_w, context=context)
    signal_col_w = "signal_eval"
    eval_df_w, rebalance_dates_w = _sample_walk_forward_eval_frame(
        test_eval,
        test_dates,
        context=context,
        direction=direction,
    )
    eval_metrics = _walk_forward_eval_metrics(
        eval_df_w,
        context=context,
        signal_col_w=signal_col_w,
    )
    perm_stats_w = _walk_forward_permutation_stats(
        train_df_w,
        test_df_w,
        rebalance_dates_w,
        context=context,
        direction=direction,
    )
    portfolio_metrics = _walk_forward_portfolio_metrics(
        eval_df_w,
        rebalance_dates_w,
        context=context,
        signal_col_w=signal_col_w,
    )
    bt_stats_w, bt_benchmark_stats_w, bt_active_stats_w = _evaluate_injected_walk_forward_backtest(
        window_meta,
        model_w=model_w,
        direction=direction,
        context=context,
    )

    _update_walk_forward_result(
        result,
        context=context,
        direction=direction,
        cv_stats=cv_stats,
        train_ic_stats=train_ic_stats,
        train_ic_raw_stats=train_ic_raw_stats,
        eval_metrics=eval_metrics,
        portfolio_metrics=portfolio_metrics,
        backtest_stats=(bt_stats_w, bt_benchmark_stats_w, bt_active_stats_w),
        perm_stats_w=perm_stats_w,
        importance_source_w=importance_source_w,
        importance_df_w=importance_df_w,
    )
    return result, importance_rows
