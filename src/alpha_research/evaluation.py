from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from .metrics import (
    assign_daily_quantile_bucket,
    daily_ic_series,
    estimate_turnover,
    hit_rate,
    quantile_returns,
    regression_error_metrics,
    summarize_ic,
    topk_positive_ratio,
)
from .modeling import build_model, fit_model
from .split import build_sample_weight
from .transform import apply_score_postprocess

logger = logging.getLogger("alpha_research")


def _postprocess_pred_column(
    frame: pd.DataFrame,
    pred_col: str,
    *,
    method: str,
    columns: list[str],
    strength: float,
    min_obs: int | None,
) -> None:
    if method == "none":
        return
    frame[pred_col] = apply_score_postprocess(
        frame,
        pred_col,
        method=method,
        columns=columns,
        strength=strength,
        min_obs=min_obs,
    )


def _permute_target_within_date(
    data: pd.DataFrame,
    target_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    def _permute(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group[target_col] = rng.permutation(cast(pd.Series, group[target_col]).to_numpy())
        return group

    return data.groupby("trade_date", group_keys=False, sort=False).apply(_permute)


def _permutation_test_ic(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    n_runs: int,
    seed: int | None,
    signal_direction: float,
    *,
    model_type: str,
    model_params: Mapping[str, Any],
    features: list[str],
    fit_target_col: str,
    target_col: str,
    sample_weight_mode: str,
    sample_weight_params: Mapping[str, Any],
    eval_dates: list[pd.Timestamp] | None = None,
    score_postprocess_method: str = "none",
    score_postprocess_columns: list[str] | None = None,
    score_postprocess_strength: float = 1.0,
    score_postprocess_min_obs: int | None = None,
) -> list[float]:
    scores = []
    eval_date_values = sorted(set(pd.to_datetime(eval_dates))) if eval_dates else None
    if eval_date_values:
        from .date_slices import _build_trade_date_slices, _slice_trade_dates

        (
            test_data_sorted,
            _,
            test_date_start_rows,
            test_date_end_rows,
            test_date_to_pos,
        ) = _build_trade_date_slices(test_data)
        eval_test_data = _slice_trade_dates(
            test_data_sorted,
            test_date_start_rows,
            test_date_end_rows,
            test_date_to_pos,
            eval_date_values,
        )
    else:
        eval_test_data = test_data
    for idx in range(n_runs):
        run_seed = None if seed is None else seed + idx
        rng = np.random.default_rng(run_seed)
        perm_train = _permute_target_within_date(train_data, fit_target_col, rng)

        perm_model = build_model(model_type, model_params)
        perm_weights = build_sample_weight(
            perm_train,
            sample_weight_mode,
            params=sample_weight_params,
        )
        fit_model(
            perm_model,
            model_type,
            perm_train,
            features=features,
            target_col=fit_target_col,
            sample_weight=perm_weights,
        )

        perm_test = eval_test_data.copy()
        perm_test["pred"] = perm_model.predict(perm_test[features])
        _postprocess_pred_column(
            perm_test,
            "pred",
            method=score_postprocess_method,
            columns=score_postprocess_columns or [],
            strength=score_postprocess_strength,
            min_obs=score_postprocess_min_obs,
        )
        if signal_direction != 1.0:
            perm_test["pred"] = perm_test["pred"] * signal_direction

        ic_values = daily_ic_series(perm_test, target_col, "pred")
        scores.append(float(ic_values.mean()) if not ic_values.empty else np.nan)
    return scores


def _record_primary_period_metrics(
    result: dict[str, Any],
    eval_df: pd.DataFrame,
    *,
    target: str,
    signal_col: str,
    label_prefix: str,
) -> None:
    ic_series = daily_ic_series(eval_df, target, signal_col)
    ic_stats = summarize_ic(ic_series)
    logger.info(
        "%sRebalance-date IC: mean=%.4f, std=%.4f, IR=%.2f, t=%.2f, p=%.4f (n=%s)",
        label_prefix,
        ic_stats["mean"],
        ic_stats["std"],
        ic_stats["ir"],
        ic_stats["t_stat"],
        ic_stats["p_value"],
        ic_stats["n"],
    )
    result["ic_series"] = ic_series
    result["ic_stats"] = ic_stats

    pearson_ic_series = daily_ic_series(eval_df, target, signal_col, method="pearson")
    pearson_ic_stats = summarize_ic(pearson_ic_series)
    logger.info(
        "%sRebalance-date Pearson IC: mean=%.4f, std=%.4f, IR=%.2f, t=%.2f, p=%.4f (n=%s)",
        label_prefix,
        pearson_ic_stats["mean"],
        pearson_ic_stats["std"],
        pearson_ic_stats["ir"],
        pearson_ic_stats["t_stat"],
        pearson_ic_stats["p_value"],
        pearson_ic_stats["n"],
    )
    result["pearson_ic_series"] = pearson_ic_series
    result["pearson_ic_stats"] = pearson_ic_stats

    target_values = cast(pd.Series, eval_df[target])
    signal_values = cast(pd.Series, eval_df[signal_col])
    error_metrics = regression_error_metrics(target_values, signal_values)
    result["error_metrics"] = error_metrics
    if error_metrics and error_metrics.get("n", 0) > 0:
        logger.info(
            "%sError metrics: MAE=%.6f, RMSE=%.6f, R2=%.4f (n=%s)",
            label_prefix,
            error_metrics.get("mae", np.nan),
            error_metrics.get("rmse", np.nan),
            error_metrics.get("r2", np.nan),
            error_metrics.get("n", 0),
        )

    hit_stats = hit_rate(target_values, signal_values)
    result["hit_rate"] = hit_stats
    if hit_stats and hit_stats.get("n", 0) > 0:
        logger.info(
            "%sHit rate: %.2f%% (n=%s)",
            label_prefix,
            hit_stats.get("hit_rate", np.nan) * 100,
            hit_stats.get("n", 0),
        )


def _record_quantile_return_metrics(
    result: dict[str, Any],
    eval_df: pd.DataFrame,
    *,
    target: str,
    signal_col: str,
    label_prefix: str,
    n_quantiles: int,
) -> None:
    quantile_ts = quantile_returns(eval_df, signal_col, target, n_quantiles)
    quantile_mean = quantile_ts.mean() if not quantile_ts.empty else pd.Series(dtype=float)
    result["quantile_ts"] = quantile_ts
    result["quantile_mean"] = quantile_mean
    if not quantile_mean.empty:
        for q_idx, value in quantile_mean.items():
            q_number = int(cast(Any, q_idx)) + 1
            logger.info("%sQ%s mean return: %.4f%%", label_prefix, q_number, value * 100)
        long_short = quantile_mean.iloc[-1] - quantile_mean.iloc[0]
        logger.info("%sLong-short (Q%s-Q1): %.4f%%", label_prefix, n_quantiles, long_short * 100)
    else:
        logger.info(
            "%sQuantile returns not available - insufficient symbols per date.",
            label_prefix,
        )


def _record_turnover_topk_metrics(
    result: dict[str, Any],
    eval_df: pd.DataFrame,
    *,
    target: str,
    signal_col: str,
    label_prefix: str,
    top_k: int,
    rebalance_dates_eval: list[pd.Timestamp],
    eval_buffer_exit: int,
    eval_buffer_entry: int,
    transaction_cost_bps: float,
) -> None:
    symbol_count = int(cast(pd.Series, eval_df["symbol"]).nunique()) if not eval_df.empty else 0
    k = min(top_k, symbol_count)
    if k > 0 and rebalance_dates_eval:
        turnover_series = estimate_turnover(
            eval_df,
            signal_col,
            k,
            rebalance_dates_eval,
            buffer_exit=eval_buffer_exit,
            buffer_entry=eval_buffer_entry,
        )
    else:
        turnover_series = pd.Series(dtype=float, name="turnover")
    result["turnover_series"] = turnover_series
    if not turnover_series.empty:
        turnover = turnover_series.mean()
        cost_drag = 2 * (transaction_cost_bps / 10000.0) * turnover
        logger.info(
            "%sTop-%s turnover per rebalance: %.2f%% (n=%s)",
            label_prefix,
            k,
            turnover * 100,
            len(turnover_series),
        )
        logger.info(
            "%sApprox cost drag per rebalance: %.2f%% at %s bps per side",
            label_prefix,
            cost_drag * 100,
            transaction_cost_bps,
        )

    topk_stats = topk_positive_ratio(eval_df, signal_col, target, k)
    result["topk_positive_ratio"] = topk_stats
    if topk_stats and topk_stats.get("n_dates", 0) > 0:
        logger.info(
            "%sTop-%s positive ratio: %.2f%% (n=%s)",
            label_prefix,
            k,
            topk_stats.get("topk_positive_ratio", np.nan) * 100,
            topk_stats.get("n_dates", 0),
        )


def _record_bucket_ic_metrics(
    result: dict[str, Any],
    eval_df: pd.DataFrame,
    *,
    target: str,
    signal_col: str,
    bucket_ic_enabled: bool,
    bucket_ic_schemes: list[dict[str, Any]],
    bucket_ic_method: str,
    bucket_ic_min_count: int,
    bucket_ic_summary_fn: Any,
) -> None:
    if not bucket_ic_enabled or not bucket_ic_schemes:
        return
    bucket_frames = []
    for scheme in bucket_ic_schemes:
        col = scheme["column"]
        if col not in eval_df.columns:
            continue
        bucket_type = str(scheme.get("type", "category")).strip().lower()
        if bucket_type not in {"category", "quantile"}:
            bucket_type = "category"
        data_for_bucket = eval_df.copy()
        bucket_col = col
        if bucket_type == "quantile":
            n_bins = int(scheme.get("n_bins") or 0)
            if n_bins < 2:
                continue
            bucket_col = f"bucket_{scheme['name']}"
            data_for_bucket[bucket_col] = assign_daily_quantile_bucket(data_for_bucket, col, n_bins)
        summary_df = bucket_ic_summary_fn(
            data_for_bucket,
            target,
            signal_col,
            bucket_col,
            method=bucket_ic_method,
            min_count=bucket_ic_min_count,
        )
        if not summary_df.empty:
            summary_df.insert(0, "scheme", scheme["name"])
            summary_df.insert(1, "type", bucket_type)
            if bucket_type == "quantile":
                summary_df.insert(2, "n_bins", int(scheme.get("n_bins") or 0))
            summary_df["method"] = bucket_ic_method
            bucket_frames.append(summary_df)
    if bucket_frames:
        bucket_df = pd.concat(bucket_frames, ignore_index=True)
        result["bucket_ic"] = bucket_df.to_dict(orient="records")


def _record_quantile_turnover_bucket_metrics(
    result: dict[str, Any],
    eval_df: pd.DataFrame,
    *,
    target: str,
    signal_col: str,
    label_prefix: str,
    n_quantiles: int,
    top_k: int,
    rebalance_dates_eval: list[pd.Timestamp],
    eval_buffer_exit: int,
    eval_buffer_entry: int,
    transaction_cost_bps: float,
    bucket_ic_enabled: bool,
    bucket_ic_schemes: list[dict[str, Any]],
    bucket_ic_method: str,
    bucket_ic_min_count: int,
    bucket_ic_summary_fn: Any,
) -> None:
    _record_quantile_return_metrics(
        result,
        eval_df,
        target=target,
        signal_col=signal_col,
        label_prefix=label_prefix,
        n_quantiles=n_quantiles,
    )
    _record_turnover_topk_metrics(
        result,
        eval_df,
        target=target,
        signal_col=signal_col,
        label_prefix=label_prefix,
        top_k=top_k,
        rebalance_dates_eval=rebalance_dates_eval,
        eval_buffer_exit=eval_buffer_exit,
        eval_buffer_entry=eval_buffer_entry,
        transaction_cost_bps=transaction_cost_bps,
    )
    _record_bucket_ic_metrics(
        result,
        eval_df,
        target=target,
        signal_col=signal_col,
        bucket_ic_enabled=bucket_ic_enabled,
        bucket_ic_schemes=bucket_ic_schemes,
        bucket_ic_method=bucket_ic_method,
        bucket_ic_min_count=bucket_ic_min_count,
        bucket_ic_summary_fn=bucket_ic_summary_fn,
    )
