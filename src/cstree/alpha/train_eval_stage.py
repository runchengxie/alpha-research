from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..pipeline.contracts import TrainEvalData, TrainEvalRequest
from ..pipeline.dates import build_walk_forward_windows
from ..pipeline.eval import _evaluate_period
from ..pipeline.live import _prepare_live_snapshot
from ..pipeline.stats import (
    _compute_rolling_ic,
    _compute_rolling_sharpe,
    _latest_rolling_stats,
    build_recency_diagnostics,
)
from ..pipeline.support import (
    _annotate_positions_window,
    _summarize_walk_forward_feature_stability,
)
from ..pipeline.train_eval_request_builder import (
    train_eval_request_from_kwargs as _build_train_eval_request_from_kwargs,
)
from ..pipeline.train_eval_result import (
    build_train_eval_stage_result as _build_train_eval_stage_result,
)
from .modeling import feature_importance_frame
from .split import time_series_cv_ic
from .train_eval_fit import (
    _TrainFitResult,
    fit_model_and_score_train as _fit_model_and_score_train,
)
from .walk_forward import _evaluate_walk_forward_window

logger = logging.getLogger("cstree")

_PIT_METADATA_COLUMNS = {"report_period", "disclosure_date", "available_date"}
_PREFERRED_INDUSTRY_COLUMNS = (
    "industry_name",
    "first_industry_name",
    "second_industry_name",
    "third_industry_name",
)


def _industry_exposure_columns(data: TrainEvalData) -> list[str]:
    preferred = [col for col in _PREFERRED_INDUSTRY_COLUMNS if col in data.industry_keep_columns]
    remaining_keep = [col for col in data.industry_keep_columns if col not in set(preferred)]
    fallback = [col for col in data.passthrough_cols if col not in _PIT_METADATA_COLUMNS]
    return list(dict.fromkeys(preferred + remaining_keep + fallback))


def run_train_eval_stage(
    *,
    request: TrainEvalRequest | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if request is not None:
        if kwargs:
            raise TypeError("Pass either request or keyword stage fields, not both.")
        return _run_train_eval_stage_impl(request)
    return _run_train_eval_stage_impl(_train_eval_request_from_kwargs(kwargs))


def _train_eval_request_from_kwargs(kwargs: dict[str, Any]) -> TrainEvalRequest:
    return _build_train_eval_request_from_kwargs(kwargs)


def _build_period_eval_context(
    request: TrainEvalRequest,
    *,
    live_state: dict[str, Any],
    updated_signal_direction: float,
    backtest_signal_direction: float,
) -> dict[str, Any]:
    data = request.data
    feature_target = request.feature_target
    model_settings = request.model
    signal_settings = request.signal
    live_settings = request.live
    backtest_settings = request.backtest
    period_settings = request.period
    services = request.services

    return {
        "features": feature_target.features,
        "target": feature_target.target,
        "signal_direction": updated_signal_direction,
        "backtest_signal_direction": backtest_signal_direction,
        "sample_on_rebalance_dates": period_settings.sample_on_rebalance_dates,
        "score_postprocess_method": signal_settings.score_postprocess_method,
        "score_postprocess_columns": signal_settings.score_postprocess_columns,
        "score_postprocess_strength": signal_settings.score_postprocess_strength,
        "score_postprocess_min_obs": signal_settings.score_postprocess_min_obs,
        "rebalance_frequency": period_settings.rebalance_frequency,
        "valid_dates_set": data.valid_dates_set,
        "perm_test_runs": period_settings.perm_test_runs,
        "perm_test_seed": period_settings.perm_test_seed,
        "model_type": model_settings.model_type,
        "model_params": model_settings.model_params,
        "train_target": feature_target.train_target,
        "sample_weight_mode": model_settings.sample_weight_mode,
        "sample_weight_params": model_settings.sample_weight_params,
        "label_horizon_mode": period_settings.label_horizon_mode,
        "label_horizon_effective": period_settings.label_horizon_effective,
        "n_quantiles": period_settings.n_quantiles,
        "top_k": period_settings.top_k,
        "eval_buffer_exit": period_settings.eval_buffer_exit,
        "eval_buffer_entry": period_settings.eval_buffer_entry,
        "transaction_cost_bps": period_settings.transaction_cost_bps,
        "bucket_ic_enabled": period_settings.bucket_ic_enabled,
        "bucket_ic_schemes": period_settings.bucket_ic_schemes,
        "bucket_ic_method": period_settings.bucket_ic_method,
        "bucket_ic_min_count": period_settings.bucket_ic_min_count,
        "backtest_rebalance_frequency": backtest_settings.backtest_rebalance_frequency,
        "backtest_enabled": backtest_settings.backtest_enabled,
        "live_enabled": live_settings.live_enabled,
        "backtest_top_k": backtest_settings.backtest_top_k,
        "label_shift_days": backtest_settings.label_shift_days,
        "backtest_weighting": backtest_settings.backtest_weighting,
        "backtest_buffer_exit": backtest_settings.backtest_buffer_exit,
        "backtest_buffer_entry": backtest_settings.backtest_buffer_entry,
        "backtest_long_only": backtest_settings.backtest_long_only,
        "backtest_short_k": backtest_settings.backtest_short_k,
        "backtest_tradable_col": backtest_settings.backtest_tradable_col,
        "backtest_group_col": backtest_settings.backtest_group_col,
        "backtest_max_names_per_group": backtest_settings.backtest_max_names_per_group,
        "backtest_liquidity_floor_col": backtest_settings.backtest_liquidity_floor_col,
        "backtest_liquidity_floor_quantile": backtest_settings.backtest_liquidity_floor_quantile,
        "backtest_weighting_liquidity_col": backtest_settings.backtest_weighting_liquidity_col,
        "backtest_max_turnover_per_rebalance": (
            backtest_settings.backtest_max_turnover_per_rebalance
        ),
        "backtest_selection_tiebreak_col": backtest_settings.backtest_selection_tiebreak_col,
        "backtest_selection_score_bucket_size": (
            backtest_settings.backtest_selection_score_bucket_size
        ),
        "backtest_selection_score_margin": backtest_settings.backtest_selection_score_margin,
        "backtest_selection_score_margin_rank_limit": (
            backtest_settings.backtest_selection_score_margin_rank_limit
        ),
        "post_buffer_exposure_repair": backtest_settings.backtest_post_buffer_exposure_repair,
        "cash_gross_overlay": backtest_settings.backtest_cash_gross_overlay,
        "freshness_overlay": backtest_settings.backtest_freshness_overlay,
        "backtest_preserve_gross_exposure": backtest_settings.backtest_preserve_gross_exposure,
        "execution_model": backtest_settings.execution_model,
        "execution_sim_config": backtest_settings.execution_sim_config,
        "positions_by_rebalance_live": live_state["positions_by_rebalance_live"],
        "backtest_cost_bps_effective": backtest_settings.backtest_cost_bps_effective,
        "backtest_trading_days_per_year": backtest_settings.backtest_trading_days_per_year,
        "backtest_exit_mode": backtest_settings.backtest_exit_mode,
        "backtest_exit_horizon_days": backtest_settings.backtest_exit_horizon_days,
        "backtest_pricing_df": data.backtest_pricing_df,
        "backtest_exit_price_policy": backtest_settings.backtest_exit_price_policy,
        "backtest_exit_fallback_policy": backtest_settings.backtest_exit_fallback_policy,
        "benchmark_df": data.benchmark_df,
        "benchmark_return_series": data.benchmark_return_series,
        "exposure_source_df": data.df_full,
        "industry_source_df": data.industry_source_df,
        "fundamentals_mcap_col": feature_target.fundamentals_mcap_col,
        "industry_columns": _industry_exposure_columns(data),
        "price_col": feature_target.price_col,
        "price_passthrough_cols": data.price_passthrough_cols,
        "passthrough_cols": data.passthrough_cols,
        "bucket_cols": data.bucket_cols,
        "backtest_topk_fn": services.backtest_topk_fn,
        "bucket_ic_summary_fn": services.bucket_ic_summary_fn,
    }


def _build_walk_forward_context(
    request: TrainEvalRequest,
    *,
    updated_signal_direction: float,
) -> dict[str, Any]:
    data = request.data
    feature_target = request.feature_target
    model_settings = request.model
    signal_settings = request.signal
    backtest_settings = request.backtest
    period_settings = request.period
    walk_forward_settings = request.walk_forward
    services = request.services

    return {
        "df_model_sorted": data.df_model_sorted,
        "all_dates": data.all_dates,
        "all_date_start_rows": data.all_date_start_rows,
        "all_date_end_rows": data.all_date_end_rows,
        "all_date_to_pos": data.all_date_to_pos,
        "train_window_mode": model_settings.train_window_mode,
        "train_window_size": model_settings.train_window_size,
        "train_window_unit": model_settings.train_window_unit,
        "signal_direction": updated_signal_direction,
        "signal_direction_mode": signal_settings.signal_direction_mode,
        "features": feature_target.features,
        "target": feature_target.target,
        "n_splits": model_settings.n_splits,
        "embargo_steps": model_settings.embargo_steps,
        "purge_steps": model_settings.purge_steps,
        "cv_purge_mode": model_settings.cv_purge_mode,
        "model_cfg": model_settings.model_cfg,
        "min_abs_ic_to_flip": signal_settings.min_abs_ic_to_flip,
        "sample_weight_mode": model_settings.sample_weight_mode,
        "sample_weight_params": model_settings.sample_weight_params,
        "train_target": feature_target.train_target,
        "model_type": model_settings.model_type,
        "model_params": model_settings.model_params,
        "report_train_ic": signal_settings.report_train_ic,
        "sample_on_rebalance_dates": period_settings.sample_on_rebalance_dates,
        "score_postprocess_method": signal_settings.score_postprocess_method,
        "score_postprocess_columns": signal_settings.score_postprocess_columns,
        "score_postprocess_strength": signal_settings.score_postprocess_strength,
        "score_postprocess_min_obs": signal_settings.score_postprocess_min_obs,
        "rebalance_frequency": period_settings.rebalance_frequency,
        "valid_dates_set": data.valid_dates_set,
        "wf_perm_test_enabled": walk_forward_settings.wf_perm_test_enabled,
        "wf_perm_test_runs": walk_forward_settings.wf_perm_test_runs,
        "wf_perm_test_seed": walk_forward_settings.wf_perm_test_seed,
        "n_quantiles": period_settings.n_quantiles,
        "top_k": period_settings.top_k,
        "eval_buffer_exit": period_settings.eval_buffer_exit,
        "eval_buffer_entry": period_settings.eval_buffer_entry,
        "wf_backtest_enabled": walk_forward_settings.wf_backtest_enabled,
        "backtest_signal_direction_raw": backtest_settings.backtest_signal_direction_raw,
        "df_full": data.df_full,
        "price_col": feature_target.price_col,
        "backtest_rebalance_frequency": backtest_settings.backtest_rebalance_frequency,
        "label_horizon_mode": period_settings.label_horizon_mode,
        "label_horizon_days": int(period_settings.label_horizon_effective),
        "label_shift_days": backtest_settings.label_shift_days,
        "backtest_cost_bps_effective": backtest_settings.backtest_cost_bps_effective,
        "backtest_trading_days_per_year": backtest_settings.backtest_trading_days_per_year,
        "backtest_exit_mode": backtest_settings.backtest_exit_mode,
        "backtest_exit_horizon_days": backtest_settings.backtest_exit_horizon_days,
        "backtest_long_only": backtest_settings.backtest_long_only,
        "backtest_short_k": backtest_settings.backtest_short_k,
        "backtest_buffer_exit": backtest_settings.backtest_buffer_exit,
        "backtest_buffer_entry": backtest_settings.backtest_buffer_entry,
        "backtest_group_col": backtest_settings.backtest_group_col,
        "backtest_max_names_per_group": backtest_settings.backtest_max_names_per_group,
        "backtest_liquidity_floor_col": backtest_settings.backtest_liquidity_floor_col,
        "backtest_liquidity_floor_quantile": backtest_settings.backtest_liquidity_floor_quantile,
        "backtest_weighting": backtest_settings.backtest_weighting,
        "backtest_weighting_liquidity_col": backtest_settings.backtest_weighting_liquidity_col,
        "backtest_max_turnover_per_rebalance": (
            backtest_settings.backtest_max_turnover_per_rebalance
        ),
        "backtest_selection_tiebreak_col": backtest_settings.backtest_selection_tiebreak_col,
        "backtest_selection_score_bucket_size": (
            backtest_settings.backtest_selection_score_bucket_size
        ),
        "backtest_selection_score_margin": backtest_settings.backtest_selection_score_margin,
        "backtest_selection_score_margin_rank_limit": (
            backtest_settings.backtest_selection_score_margin_rank_limit
        ),
        "post_buffer_exposure_repair": backtest_settings.backtest_post_buffer_exposure_repair,
        "cash_gross_overlay": backtest_settings.backtest_cash_gross_overlay,
        "freshness_overlay": backtest_settings.backtest_freshness_overlay,
        "backtest_preserve_gross_exposure": backtest_settings.backtest_preserve_gross_exposure,
        "backtest_tradable_col": backtest_settings.backtest_tradable_col,
        "backtest_exit_price_policy": backtest_settings.backtest_exit_price_policy,
        "backtest_exit_fallback_policy": backtest_settings.backtest_exit_fallback_policy,
        "execution_model": backtest_settings.execution_model,
        "execution_sim_config": backtest_settings.execution_sim_config,
        "backtest_pricing_df": data.backtest_pricing_df,
        "benchmark_df": data.benchmark_df,
        "benchmark_return_series": data.benchmark_return_series,
        "backtest_top_k": backtest_settings.backtest_top_k,
        "wf_feature_top_k": walk_forward_settings.wf_feature_top_k,
        "backtest_topk_fn": services.backtest_topk_fn,
    }


def _run_walk_forward_evaluation(
    request: TrainEvalRequest,
    *,
    updated_signal_direction: float,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    walk_forward_settings = request.walk_forward
    walk_forward_results: list[dict] = []
    walk_forward_importance_rows: list[dict[str, Any]] = []
    if walk_forward_settings.wf_enabled:
        walk_forward_context = _build_walk_forward_context(
            request,
            updated_signal_direction=updated_signal_direction,
        )
        try:
            walk_forward_test_size = float(walk_forward_settings.wf_test_size)
        except (TypeError, ValueError):
            walk_forward_test_size = None
        windows = build_walk_forward_windows(
            request.data.all_dates,
            walk_forward_test_size,
            walk_forward_settings.wf_n_windows,
            walk_forward_settings.wf_step_size,
            walk_forward_settings.effective_gap_steps,
            walk_forward_settings.wf_anchor_end,
        )
        if not windows:
            logger.info("Walk-forward evaluation skipped: insufficient windows.")
        else:
            if len(windows) < walk_forward_settings.wf_n_windows:
                logger.warning(
                    "Walk-forward requested %s windows but only %s fit "
                    "(test_size=%s, step_size=%s, anchor_end=%s). "
                    "Reduce eval.test_size / eval.walk_forward.test_size, "
                    "set a smaller eval.walk_forward.step_size, or lower n_windows.",
                    walk_forward_settings.wf_n_windows,
                    len(windows),
                    walk_forward_test_size,
                    walk_forward_settings.wf_step_size,
                    walk_forward_settings.wf_anchor_end,
                )
            logger.info("Walk-forward evaluation: %s windows.", len(windows))
            for window_meta in windows:
                window_result, window_importance_rows = _evaluate_walk_forward_window(
                    window_meta,
                    context=walk_forward_context,
                )
                walk_forward_results.append(window_result)
                walk_forward_importance_rows.extend(window_importance_rows)

    walk_forward_importance_df = pd.DataFrame(walk_forward_importance_rows)
    walk_forward_feature_stability_df = _summarize_walk_forward_feature_stability(
        walk_forward_importance_df,
        walk_forward_settings.wf_feature_top_k,
    )
    return walk_forward_results, walk_forward_importance_df, walk_forward_feature_stability_df


def _run_cv_and_fit_model(request: TrainEvalRequest) -> _TrainFitResult:
    data = request.data
    feature_target = request.feature_target
    model_settings = request.model
    signal_settings = request.signal

    logger.info("Time-series cross-validation (IC) ...")
    cv_scores_raw = time_series_cv_ic(
        data.train_df,
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
        train_window_unit=model_settings.train_window_unit,
        fit_target_col=feature_target.train_target,
        score_postprocess_method=signal_settings.score_postprocess_method,
        score_postprocess_columns=signal_settings.score_postprocess_columns,
        score_postprocess_strength=signal_settings.score_postprocess_strength,
        score_postprocess_min_obs=signal_settings.score_postprocess_min_obs,
        cv_purge_mode=model_settings.cv_purge_mode,
        label_horizon_mode=request.period.label_horizon_mode,
        label_horizon_days=int(request.period.label_horizon_effective),
        label_shift_days=request.backtest.label_shift_days,
        all_trade_dates=data.all_dates,
    )
    if cv_scores_raw:
        logger.info(
            "CV IC (raw): mean=%.4f, std=%.4f",
            np.nanmean(cv_scores_raw),
            np.nanstd(cv_scores_raw),
        )
        logger.info("CV fold ICs (raw): %s", [f"{s:.4f}" for s in cv_scores_raw])
    else:
        logger.info("CV IC not available - insufficient data after embargo/purge.")

    updated_signal_direction = signal_settings.signal_direction
    if signal_settings.signal_direction_mode == "cv_ic" and cv_scores_raw:
        cv_mean = float(np.nanmean(cv_scores_raw))
        if (
            np.isfinite(cv_mean)
            and cv_mean != 0
            and abs(cv_mean) >= signal_settings.min_abs_ic_to_flip
        ):
            updated_signal_direction = float(np.sign(cv_mean))
            logger.info("Signal direction set from CV IC: %s", updated_signal_direction)
        else:
            logger.info(
                "CV IC mean below threshold (|mean| < %.4f); keeping signal direction: %s",
                signal_settings.min_abs_ic_to_flip,
                updated_signal_direction,
            )

    fit_state = _fit_model_and_score_train(
        data.train_df,
        feature_target=feature_target,
        model_settings=model_settings,
        signal_settings=signal_settings,
        cv_scores_raw=cv_scores_raw,
    )
    return _TrainFitResult(
        model=fit_state.model,
        train_eval_df=fit_state.train_eval_df,
        updated_signal_direction=fit_state.updated_signal_direction,
        train_signal_col=fit_state.train_signal_col,
        train_ic_raw_stats=fit_state.train_ic_raw_stats,
        train_ic_series=fit_state.train_ic_series,
        train_ic_stats=fit_state.train_ic_stats,
        train_pearson_ic_series=fit_state.train_pearson_ic_series,
        train_pearson_ic_stats=fit_state.train_pearson_ic_stats,
        cv_scores_raw=cv_scores_raw,
        cv_scores_adj=fit_state.cv_scores_adj,
    )


def _test_window_full_data(data: TrainEvalData) -> pd.DataFrame:
    test_start = pd.to_datetime(data.test_dates[0])
    test_end = pd.to_datetime(data.test_dates[-1])
    test_df_full = data.df_full[
        (data.df_full["trade_date"] >= test_start) & (data.df_full["trade_date"] <= test_end)
    ].copy()
    if test_df_full.empty:
        raise SystemExit("Not enough test data after applying the split window.")
    return test_df_full


def _prepare_stage_live_state(
    request: TrainEvalRequest,
    model: Any,
    *,
    updated_signal_direction: float,
) -> dict[str, Any]:
    feature_target = request.feature_target
    model_settings = request.model
    signal_settings = request.signal
    live_settings = request.live
    backtest_settings = request.backtest
    return _prepare_live_snapshot(
        request.data.df_features,
        model,
        context={
            "live_enabled": live_settings.live_enabled,
            "live_as_of_token": live_settings.live_as_of,
            "live_signal_asof_token": live_settings.live_signal_asof,
            "live_entry_date_token": live_settings.live_entry_date,
            "market": live_settings.market,
            "provider": live_settings.provider,
            "target": feature_target.target,
            "live_train_mode": live_settings.live_train_mode,
            "model_type": model_settings.model_type,
            "model_params": model_settings.model_params,
            "train_window_mode": model_settings.train_window_mode,
            "train_window_size": model_settings.train_window_size,
            "train_window_unit": model_settings.train_window_unit,
            "sample_weight_mode": model_settings.sample_weight_mode,
            "sample_weight_params": model_settings.sample_weight_params,
            "train_target": feature_target.train_target,
            "features": feature_target.features,
            "signal_direction": updated_signal_direction,
            "score_postprocess_method": signal_settings.score_postprocess_method,
            "score_postprocess_columns": signal_settings.score_postprocess_columns,
            "score_postprocess_strength": signal_settings.score_postprocess_strength,
            "score_postprocess_min_obs": signal_settings.score_postprocess_min_obs,
            "backtest_rebalance_frequency": backtest_settings.backtest_rebalance_frequency,
            "min_symbols_per_date": live_settings.min_symbols_per_date,
            "price_col": feature_target.price_col,
            "backtest_top_k": backtest_settings.backtest_top_k,
            "label_shift_days": backtest_settings.label_shift_days,
            "backtest_weighting": backtest_settings.backtest_weighting,
            "backtest_buffer_exit": backtest_settings.backtest_buffer_exit,
            "backtest_buffer_entry": backtest_settings.backtest_buffer_entry,
            "backtest_long_only": backtest_settings.backtest_long_only,
            "backtest_short_k": backtest_settings.backtest_short_k,
            "backtest_tradable_col": backtest_settings.backtest_tradable_col,
            "backtest_group_col": backtest_settings.backtest_group_col,
            "backtest_max_names_per_group": backtest_settings.backtest_max_names_per_group,
            "backtest_liquidity_floor_col": backtest_settings.backtest_liquidity_floor_col,
            "backtest_liquidity_floor_quantile": (
                backtest_settings.backtest_liquidity_floor_quantile
            ),
            "backtest_weighting_liquidity_col": backtest_settings.backtest_weighting_liquidity_col,
            "backtest_max_turnover_per_rebalance": (
                backtest_settings.backtest_max_turnover_per_rebalance
            ),
            "backtest_selection_tiebreak_col": (backtest_settings.backtest_selection_tiebreak_col),
            "backtest_selection_score_bucket_size": (
                backtest_settings.backtest_selection_score_bucket_size
            ),
            "backtest_selection_score_margin": (backtest_settings.backtest_selection_score_margin),
            "backtest_selection_score_margin_rank_limit": (
                backtest_settings.backtest_selection_score_margin_rank_limit
            ),
            "post_buffer_exposure_repair": backtest_settings.backtest_post_buffer_exposure_repair,
            "cash_gross_overlay": backtest_settings.backtest_cash_gross_overlay,
            "freshness_overlay": backtest_settings.backtest_freshness_overlay,
            "backtest_pricing_df": request.data.backtest_pricing_df,
            "benchmark_df": request.data.benchmark_df,
            "benchmark_return_series": request.data.benchmark_return_series,
            "industry_source_df": request.data.industry_source_df,
            "industry_columns": _industry_exposure_columns(request.data),
            "fundamentals_mcap_col": feature_target.fundamentals_mcap_col,
            "execution_model": backtest_settings.execution_model,
        },
    )


def _cv_stats(
    cv_scores_raw: list[float],
    cv_scores_adj: list[float] | None,
    updated_signal_direction: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[float] | None]:
    if not cv_scores_raw:
        return None, None, cv_scores_adj
    cv_stats_raw = {
        "mean": float(np.nanmean(cv_scores_raw)),
        "std": float(np.nanstd(cv_scores_raw)),
        "scores": [float(score) for score in cv_scores_raw],
    }
    if cv_scores_adj is None:
        cv_scores_adj = [float(score) * updated_signal_direction for score in cv_scores_raw]
    cv_stats = {
        "mean": float(np.nanmean(cv_scores_adj)),
        "std": float(np.nanstd(cv_scores_adj)),
        "scores": [float(score) for score in cv_scores_adj],
    }
    return cv_stats_raw, cv_stats, cv_scores_adj


def _require_live_positions_if_needed(
    *,
    live_enabled: bool,
    backtest_enabled: bool,
    live_positions_ready: bool,
) -> None:
    if live_enabled and not backtest_enabled and not live_positions_ready:
        raise SystemExit(
            "live.enabled=true but no live positions were generated; "
            "refusing to fall back to backtest holdings."
        )


def _evaluate_main_period(
    request: TrainEvalRequest,
    model: Any,
    *,
    live_state: dict[str, Any],
    updated_signal_direction: float,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    data = request.data
    backtest_settings = request.backtest
    period_settings = request.period
    backtest_signal_direction = (
        updated_signal_direction
        if backtest_settings.backtest_signal_direction_raw is None
        else backtest_settings.backtest_signal_direction_raw
    )
    period_eval_context = _build_period_eval_context(
        request,
        live_state=live_state,
        updated_signal_direction=updated_signal_direction,
        backtest_signal_direction=backtest_signal_direction,
    )
    test_df_full = _test_window_full_data(data)
    eval_main = _evaluate_period(
        "Test",
        model,
        test_df_full,
        data.test_dates,
        context=period_eval_context,
        run_perm_test=period_settings.perm_test_enabled,
        perm_train_df=data.train_df,
        perm_test_df=data.test_df,
        allow_live_fallback=True,
    )
    return backtest_signal_direction, period_eval_context, eval_main


def _rolling_eval_summaries(
    eval_main: dict[str, Any],
    rolling_windows_months: list[int],
) -> tuple[dict[str, Any], float, dict[str, Any], dict[str, Any], dict[str, Any]]:
    rolling_ic_results, rolling_ic_obs_per_year = _compute_rolling_ic(
        eval_main["ic_series"], rolling_windows_months
    )
    rolling_ic_latest = {
        label: _latest_rolling_stats(frame, ["ic_mean", "ic_ir"])
        for label, frame in rolling_ic_results.items()
    }
    rolling_sharpe_results = {}
    rolling_sharpe_latest = {}
    if eval_main["bt_stats"] is not None and not eval_main["bt_net_series"].empty:
        periods_per_year = eval_main["bt_stats"].get("periods_per_year", np.nan)
        rolling_sharpe_results = _compute_rolling_sharpe(
            eval_main["bt_net_series"], rolling_windows_months, periods_per_year
        )
        rolling_sharpe_latest = {
            label: _latest_rolling_stats(frame, ["mean", "std", "sharpe"])
            for label, frame in rolling_sharpe_results.items()
        }
    return (
        rolling_ic_results,
        rolling_ic_obs_per_year,
        rolling_ic_latest,
        rolling_sharpe_results,
        rolling_sharpe_latest,
    )


def _recency_eval_diagnostics(
    eval_main: dict[str, Any],
    recency_windows: list[str],
) -> pd.DataFrame:
    periods_per_year = np.nan
    if eval_main["bt_stats"] is not None:
        periods_per_year = eval_main["bt_stats"].get("periods_per_year", np.nan)
    return build_recency_diagnostics(
        window_labels=recency_windows,
        ic_series=eval_main["ic_series"],
        returns=eval_main["bt_net_series"],
        active_returns=eval_main["bt_active_series"],
        turnover=eval_main["bt_turnover_series"],
        periods_per_year=periods_per_year,
    )


def _annotated_stage_positions(
    eval_main: dict[str, Any],
    live_state: dict[str, Any],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    positions_by_rebalance = eval_main["positions_by_rebalance"]
    positions_by_rebalance_live = live_state["positions_by_rebalance_live"]
    if positions_by_rebalance is not None and not positions_by_rebalance.empty:
        positions_by_rebalance = _annotate_positions_window(positions_by_rebalance)
    if positions_by_rebalance_live is not None and not positions_by_rebalance_live.empty:
        positions_by_rebalance_live = _annotate_positions_window(positions_by_rebalance_live)
    return positions_by_rebalance, positions_by_rebalance_live


def _prediction_diagnostics(
    eval_scored_data: pd.DataFrame | None,
) -> tuple[int | None, bool | None]:
    if eval_scored_data is None or eval_scored_data.empty or "pred" not in eval_scored_data.columns:
        return None, None
    pred_nunique = int(eval_scored_data["pred"].nunique(dropna=True))
    return pred_nunique, pred_nunique <= 1


def _importance_diagnostics(importance_df: pd.DataFrame) -> tuple[int | None, bool | None]:
    if importance_df.empty or "importance" not in importance_df.columns:
        return None, None
    importance_values = pd.to_numeric(importance_df["importance"], errors="coerce").fillna(0.0)
    feature_importance_nonzero = int((importance_values.abs() > 0.0).sum())
    return feature_importance_nonzero, feature_importance_nonzero == 0


def _feature_importance_diagnostics(
    model: Any,
    features: list[str],
    eval_scored_data: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str, int | None, bool | None, int | None, bool | None]:
    logger.info("Feature importance:")
    importance_df, importance_source = feature_importance_frame(model, features)
    logger.info("Feature importance source: %s", importance_source)
    for _, row in importance_df.iterrows():
        logger.info("  %-20s: %.4f", row["feature"], float(row["importance"]))

    pred_nunique, constant_prediction = _prediction_diagnostics(eval_scored_data)
    feature_importance_nonzero, zero_feature_importance = _importance_diagnostics(importance_df)
    return (
        importance_df,
        importance_source,
        pred_nunique,
        constant_prediction,
        feature_importance_nonzero,
        zero_feature_importance,
    )


def _run_train_eval_stage_impl(request: TrainEvalRequest) -> dict[str, Any]:
    data = request.data
    feature_target = request.feature_target
    live_settings = request.live
    backtest_settings = request.backtest
    period_settings = request.period

    features = feature_target.features
    live_enabled = live_settings.live_enabled
    backtest_enabled = backtest_settings.backtest_enabled
    rolling_windows_months = period_settings.rolling_windows_months
    recency_windows = period_settings.recency_windows

    fit_state = _run_cv_and_fit_model(request)
    model = fit_state.model
    updated_signal_direction = fit_state.updated_signal_direction
    train_ic_raw_stats = fit_state.train_ic_raw_stats
    train_ic_series = fit_state.train_ic_series
    train_ic_stats = fit_state.train_ic_stats
    train_pearson_ic_series = fit_state.train_pearson_ic_series
    train_pearson_ic_stats = fit_state.train_pearson_ic_stats
    cv_scores_raw = fit_state.cv_scores_raw or []
    cv_scores_adj = fit_state.cv_scores_adj

    logger.info("Evaluating model on train/test sets ...")
    live_state = _prepare_stage_live_state(
        request,
        model,
        updated_signal_direction=updated_signal_direction,
    )
    live_positions_ready = bool(live_state["live_positions_ready"])
    _require_live_positions_if_needed(
        live_enabled=live_enabled,
        backtest_enabled=backtest_enabled,
        live_positions_ready=live_positions_ready,
    )

    backtest_signal_direction, period_eval_context, eval_main = _evaluate_main_period(
        request,
        model,
        live_state=live_state,
        updated_signal_direction=updated_signal_direction,
    )

    (
        rolling_ic_results,
        rolling_ic_obs_per_year,
        rolling_ic_latest,
        rolling_sharpe_results,
        rolling_sharpe_latest,
    ) = _rolling_eval_summaries(eval_main, rolling_windows_months)
    recency_diagnostics = _recency_eval_diagnostics(eval_main, recency_windows)
    positions_by_rebalance, positions_by_rebalance_live = _annotated_stage_positions(
        eval_main,
        live_state,
    )

    cv_stats_raw, cv_stats, cv_scores_adj = _cv_stats(
        cv_scores_raw,
        cv_scores_adj,
        updated_signal_direction,
    )

    (
        walk_forward_results,
        walk_forward_importance_df,
        walk_forward_feature_stability_df,
    ) = _run_walk_forward_evaluation(
        request,
        updated_signal_direction=updated_signal_direction,
    )

    eval_scored_data = eval_main["scored_data"]
    (
        importance_df,
        importance_source,
        pred_nunique,
        constant_prediction,
        feature_importance_nonzero,
        zero_feature_importance,
    ) = _feature_importance_diagnostics(model, features, eval_scored_data)

    return _build_train_eval_stage_result(locals())
