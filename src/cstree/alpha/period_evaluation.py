from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from ..contracts.rebalance import _sample_rebalance_frame
from .evaluation import (
    _permutation_test_ic,
    _postprocess_pred_column,
    _record_primary_period_metrics,
    _record_quantile_turnover_bucket_metrics,
)
from .freshness_overlay import apply_freshness_overlay
from .metrics import bucket_ic_summary
from .rebalance_calendar import estimate_rebalance_gap, get_rebalance_dates

logger = logging.getLogger("cstree")


def _score_period_frame(
    frame: pd.DataFrame,
    model_eval: Any,
    *,
    features: list[str],
    score_postprocess_method: str,
    score_postprocess_columns: list[str] | None,
    score_postprocess_strength: float,
    score_postprocess_min_obs: int | None,
    signal_direction: float,
    backtest_signal_direction: float,
    label_prefix: str,
    freshness_overlay: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    scored = frame.copy()
    scored["pred"] = model_eval.predict(scored[features])
    _postprocess_pred_column(
        scored,
        "pred",
        method=score_postprocess_method,
        columns=score_postprocess_columns or [],
        strength=score_postprocess_strength,
        min_obs=score_postprocess_min_obs,
    )
    scored["signal_eval"] = scored["pred"] * signal_direction
    scored["signal_backtest"] = scored["pred"] * backtest_signal_direction
    scored, overlay_meta = apply_freshness_overlay(
        scored,
        score_col="signal_backtest",
        cfg=freshness_overlay,
    )
    if overlay_meta.get("enabled"):
        logger.info(
            "%sFreshness overlay applied: %s lambda=%s output_col=%s",
            label_prefix,
            overlay_meta.get("name"),
            overlay_meta.get("lambda"),
            overlay_meta.get("output_col"),
        )
    if signal_direction != 1.0:
        logger.info("%sSignal direction applied to ranking: %s", label_prefix, signal_direction)
    return scored


def _freshness_overlay_audit_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    suffix = "_freshness_volume_rank"
    for column in frame.columns:
        if not column.endswith(suffix):
            continue
        base_col = f"{column[: -len(suffix)]}_base"
        cols.extend([base_col, column])
    return [column for column in dict.fromkeys(cols) if column in frame.columns]


def _build_scored_data(
    eval_df_full: pd.DataFrame,
    *,
    price_col: str,
    target: str,
    price_passthrough_cols: list[str],
    passthrough_cols: list[str],
    bucket_cols: list[str],
    feature_cols: list[str] | None = None,
    backtest_tradable_col: str | None,
) -> pd.DataFrame:
    freshness_audit_cols = _freshness_overlay_audit_columns(eval_df_full)
    scored_cols = [
        "trade_date",
        "symbol",
        price_col,
        target,
        "pred",
        "signal_eval",
        "signal_backtest",
    ]
    scored_cols.extend(freshness_audit_cols)
    scored_cols.extend(price_passthrough_cols)
    scored_cols.extend(passthrough_cols)
    scored_cols.extend(bucket_cols)
    scored_cols.extend(feature_cols or [])
    scored_cols = [
        column for column in dict.fromkeys(scored_cols) if column in eval_df_full.columns
    ]
    if backtest_tradable_col and backtest_tradable_col in eval_df_full.columns:
        scored_cols.append(backtest_tradable_col)
    return cast(pd.DataFrame, eval_df_full.loc[:, scored_cols].copy())


def _record_period_permutation(
    result: dict[str, Any],
    *,
    context: Mapping[str, Any],
    label_prefix: str,
    signal_direction: float,
    rebalance_dates_eval: list[pd.Timestamp],
    perm_train_df: Optional[pd.DataFrame],
    perm_test_df: Optional[pd.DataFrame],
) -> None:
    if perm_train_df is None or perm_test_df is None:
        raise SystemExit("Permutation test requested but data was not provided.")
    logger.info("%sPermutation test (shuffle train labels within date) ...", label_prefix)
    perm_scores = _permutation_test_ic(
        perm_train_df,
        perm_test_df,
        context["perm_test_runs"],
        context["perm_test_seed"],
        signal_direction,
        model_type=context["model_type"],
        model_params=context["model_params"],
        features=context["features"],
        fit_target_col=context["train_target"],
        target_col=context["target"],
        sample_weight_mode=context["sample_weight_mode"],
        sample_weight_params=context["sample_weight_params"],
        eval_dates=rebalance_dates_eval,
        score_postprocess_method=context["score_postprocess_method"],
        score_postprocess_columns=context["score_postprocess_columns"],
        score_postprocess_strength=context["score_postprocess_strength"],
        score_postprocess_min_obs=context["score_postprocess_min_obs"],
    )
    if not perm_scores:
        return
    perm_mean = np.nanmean(perm_scores)
    perm_std = np.nanstd(perm_scores)
    logger.info(
        "%sPermutation IC: mean=%.4f, std=%.4f, runs=%s",
        label_prefix,
        perm_mean,
        perm_std,
        len(perm_scores),
    )
    logger.info("%sPermutation ICs: %s", label_prefix, [f"{s:.4f}" for s in perm_scores])
    result["perm_stats"] = {
        "mean": float(perm_mean),
        "std": float(perm_std),
        "scores": [float(score) for score in perm_scores],
        "runs": int(len(perm_scores)),
    }


def _warn_label_rebalance_gap(
    *,
    eval_df_full: pd.DataFrame,
    context: Mapping[str, Any],
    label_prefix: str,
) -> None:
    trade_dates_sorted_full = sorted(eval_df_full["trade_date"].unique())
    rebalance_dates_full = get_rebalance_dates(
        trade_dates_sorted_full,
        context["rebalance_frequency"],
    )
    rebalance_gap = estimate_rebalance_gap(trade_dates_sorted_full, rebalance_dates_full)
    if (
        context["backtest_exit_mode"] == "rebalance"
        and np.isfinite(rebalance_gap)
        and context["label_horizon_mode"] == "fixed"
    ):
        gap_diff = abs(rebalance_gap - context["label_horizon_effective"])
        if gap_diff >= max(3.0, rebalance_gap * 0.25):
            logger.warning(
                "%sLabel horizon (%s days) differs from rebalance gap (median %.1f days).",
                label_prefix,
                context["label_horizon_effective"],
                rebalance_gap,
            )


def _score_and_record_period_eval_metrics(
    result: dict[str, Any],
    *,
    test_df_full: pd.DataFrame,
    model_eval: Any,
    test_dates: np.ndarray,
    context: Mapping[str, Any],
    label_prefix: str,
    run_perm_test: bool,
    perm_train_df: Optional[pd.DataFrame],
    perm_test_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    eval_df_full = _score_period_frame(
        test_df_full,
        model_eval,
        features=context["features"],
        score_postprocess_method=context["score_postprocess_method"],
        score_postprocess_columns=context["score_postprocess_columns"],
        score_postprocess_strength=context["score_postprocess_strength"],
        score_postprocess_min_obs=context["score_postprocess_min_obs"],
        signal_direction=context["signal_direction"],
        backtest_signal_direction=context["backtest_signal_direction"],
        freshness_overlay=context.get("freshness_overlay"),
        label_prefix=label_prefix,
    )

    eval_allowed_dates = test_dates if context["sample_on_rebalance_dates"] else None
    eval_df, rebalance_dates_eval = _sample_rebalance_frame(
        eval_df_full,
        frequency=context["rebalance_frequency"],
        valid_dates=context["valid_dates_set"],
        allowed_dates=eval_allowed_dates,
    )
    result["eval_rebalance_dates"] = rebalance_dates_eval

    signal_col = "signal_eval"
    _record_primary_period_metrics(
        result,
        eval_df,
        target=context["target"],
        signal_col=signal_col,
        label_prefix=label_prefix,
    )
    if run_perm_test:
        _record_period_permutation(
            result,
            context=context,
            label_prefix=label_prefix,
            signal_direction=context["signal_direction"],
            rebalance_dates_eval=rebalance_dates_eval,
            perm_train_df=perm_train_df,
            perm_test_df=perm_test_df,
        )
    _warn_label_rebalance_gap(
        eval_df_full=eval_df_full,
        context=context,
        label_prefix=label_prefix,
    )
    _record_quantile_turnover_bucket_metrics(
        result,
        eval_df,
        target=context["target"],
        signal_col=signal_col,
        label_prefix=label_prefix,
        n_quantiles=context["n_quantiles"],
        top_k=context["top_k"],
        rebalance_dates_eval=rebalance_dates_eval,
        eval_buffer_exit=context["eval_buffer_exit"],
        eval_buffer_entry=context["eval_buffer_entry"],
        transaction_cost_bps=context["transaction_cost_bps"],
        bucket_ic_enabled=context["bucket_ic_enabled"],
        bucket_ic_schemes=context["bucket_ic_schemes"],
        bucket_ic_method=context["bucket_ic_method"],
        bucket_ic_min_count=context["bucket_ic_min_count"],
        bucket_ic_summary_fn=context.get("bucket_ic_summary_fn", bucket_ic_summary),
    )
    return eval_df_full
