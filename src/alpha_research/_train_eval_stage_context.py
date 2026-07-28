"""Train/eval stage: period and walk-forward context assembly.

Private helpers that build the period-eval and walk-forward contexts from a
:class:`TrainEvalRequest`, resolve industry-exposure columns, and compute the
test-window full data frame. Split out of the historical single-file
:mod:`alpha_research.train_eval_stage` implementation to keep individual files
smaller while preserving the exact public/private symbol surface.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .train_eval_contracts import TrainEvalData, TrainEvalRequest

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
        "walk_forward_backtest_fn": services.walk_forward_backtest_fn,
    }


def _test_window_full_data(data: TrainEvalData) -> pd.DataFrame:
    test_start = pd.to_datetime(data.test_dates[0])
    test_end = pd.to_datetime(data.test_dates[-1])
    test_df_full = data.df_full[
        (data.df_full["trade_date"] >= test_start) & (data.df_full["trade_date"] <= test_end)
    ].copy()
    if test_df_full.empty:
        raise SystemExit("Not enough test data after applying the split window.")
    return test_df_full
