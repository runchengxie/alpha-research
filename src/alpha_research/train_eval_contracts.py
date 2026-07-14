from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .backends import (
    ExperimentRecorder,
    NativeTrainerBackend,
    NullExperimentRecorder,
    TrainerBackend,
)


@dataclass(frozen=True)
class TrainEvalData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    test_dates: np.ndarray
    df_features: pd.DataFrame
    df_full: pd.DataFrame
    df_model_sorted: pd.DataFrame
    all_dates: np.ndarray
    all_date_start_rows: np.ndarray
    all_date_end_rows: np.ndarray
    all_date_to_pos: dict[pd.Timestamp, int]
    valid_dates_set: set[pd.Timestamp]
    backtest_pricing_df: pd.DataFrame
    benchmark_df: pd.DataFrame | None
    benchmark_return_series: pd.Series
    industry_source_df: pd.DataFrame
    passthrough_cols: list[str]
    industry_keep_columns: list[str]
    price_passthrough_cols: list[str]
    bucket_cols: list[str]


@dataclass(frozen=True)
class TrainEvalFeatureTarget:
    features: list[str]
    target: str
    train_target: str
    price_col: str
    fundamentals_mcap_col: str


@dataclass(frozen=True)
class TrainEvalModelSettings:
    model_type: str
    model_params: dict[str, Any]
    model_cfg: dict[str, Any]
    sample_weight_mode: str
    sample_weight_params: dict[str, Any]
    n_splits: int
    embargo_steps: int
    purge_steps: int
    cv_purge_mode: str
    train_window_mode: str
    train_window_size: int | None
    train_window_unit: str | None


@dataclass(frozen=True)
class TrainEvalSignalSettings:
    signal_direction_mode: str
    signal_direction: float
    min_abs_ic_to_flip: float
    score_postprocess_method: str
    score_postprocess_columns: list[str] | None
    score_postprocess_strength: float
    score_postprocess_min_obs: int
    report_train_ic: bool


@dataclass(frozen=True)
class TrainEvalLiveSettings:
    live_enabled: bool
    live_as_of: str | None
    market: str
    provider: str
    live_train_mode: str
    min_symbols_per_date: int
    live_signal_asof: str | None = None
    live_entry_date: str | None = None


@dataclass(frozen=True)
class TrainEvalBacktestSettings:
    backtest_top_k: int
    label_shift_days: int
    backtest_weighting: str
    backtest_buffer_exit: int
    backtest_buffer_entry: int
    backtest_long_only: bool
    backtest_short_k: int | None
    backtest_tradable_col: str | None
    backtest_group_col: str | None
    backtest_max_names_per_group: int | None
    backtest_liquidity_floor_col: str | None
    backtest_liquidity_floor_quantile: float | None
    backtest_weighting_liquidity_col: str
    backtest_max_turnover_per_rebalance: float | None
    backtest_post_buffer_exposure_repair: dict[str, Any]
    backtest_cash_gross_overlay: dict[str, Any]
    backtest_freshness_overlay: dict[str, Any]
    backtest_preserve_gross_exposure: bool
    execution_model: Any
    execution_sim_config: Any
    backtest_rebalance_frequency: str
    backtest_enabled: bool
    backtest_signal_direction_raw: float | None
    backtest_cost_bps_effective: float
    backtest_trading_days_per_year: int
    backtest_exit_mode: str
    backtest_exit_horizon_days: int
    backtest_exit_price_policy: str
    backtest_exit_fallback_policy: str
    backtest_selection_tiebreak_col: str | None = None
    backtest_selection_score_bucket_size: float | None = None
    backtest_selection_score_margin: float | None = None
    backtest_selection_score_margin_rank_limit: int | None = None


@dataclass(frozen=True)
class TrainEvalPeriodSettings:
    rebalance_frequency: str
    sample_on_rebalance_dates: bool
    perm_test_enabled: bool
    perm_test_runs: int
    perm_test_seed: int | None
    label_horizon_mode: str
    label_horizon_effective: int | float
    n_quantiles: int
    top_k: int
    eval_buffer_exit: int
    eval_buffer_entry: int
    transaction_cost_bps: float
    bucket_ic_enabled: bool
    bucket_ic_schemes: list[dict[str, Any]]
    bucket_ic_method: str
    bucket_ic_min_count: int
    rolling_windows_months: list[int]
    recency_windows: list[str]


@dataclass(frozen=True)
class TrainEvalWalkForwardSettings:
    wf_enabled: bool
    wf_n_windows: int
    wf_test_size: float | int | None
    wf_step_size: float | int | None
    effective_gap_steps: int
    wf_anchor_end: bool
    wf_feature_top_k: int
    wf_backtest_enabled: bool
    wf_perm_test_enabled: bool
    wf_perm_test_runs: int
    wf_perm_test_seed: int | None


@dataclass(frozen=True)
class TrainEvalServices:
    backtest_topk_fn: Any
    bucket_ic_summary_fn: Any
    walk_forward_backtest_fn: Any
    period_eval_fn: Any
    live_snapshot_fn: Any
    trainer_backend: TrainerBackend = field(default_factory=NativeTrainerBackend)
    experiment_recorder: ExperimentRecorder = field(default_factory=NullExperimentRecorder)


def _data_window_kwargs(data: TrainEvalData) -> dict[str, Any]:
    return {
        "train_df": data.train_df,
        "test_df": data.test_df,
        "test_dates": data.test_dates,
        "df_features": data.df_features,
        "df_full": data.df_full,
        "df_model_sorted": data.df_model_sorted,
        "all_dates": data.all_dates,
        "all_date_start_rows": data.all_date_start_rows,
        "all_date_end_rows": data.all_date_end_rows,
        "all_date_to_pos": data.all_date_to_pos,
    }


def _feature_model_kwargs(
    feature_target: TrainEvalFeatureTarget,
    model: TrainEvalModelSettings,
) -> dict[str, Any]:
    return {
        "features": feature_target.features,
        "target": feature_target.target,
        "train_target": feature_target.train_target,
        "model_type": model.model_type,
        "model_params": model.model_params,
        "model_cfg": model.model_cfg,
        "sample_weight_mode": model.sample_weight_mode,
        "sample_weight_params": model.sample_weight_params,
        "n_splits": model.n_splits,
        "embargo_steps": model.embargo_steps,
        "purge_steps": model.purge_steps,
        "cv_purge_mode": model.cv_purge_mode,
        "train_window_mode": model.train_window_mode,
        "train_window_size": model.train_window_size,
        "train_window_unit": model.train_window_unit,
    }


def _signal_live_kwargs(
    signal: TrainEvalSignalSettings,
    live: TrainEvalLiveSettings,
) -> dict[str, Any]:
    return {
        "signal_direction_mode": signal.signal_direction_mode,
        "signal_direction": signal.signal_direction,
        "min_abs_ic_to_flip": signal.min_abs_ic_to_flip,
        "score_postprocess_method": signal.score_postprocess_method,
        "score_postprocess_columns": signal.score_postprocess_columns,
        "score_postprocess_strength": signal.score_postprocess_strength,
        "score_postprocess_min_obs": signal.score_postprocess_min_obs,
        "report_train_ic": signal.report_train_ic,
        "live_enabled": live.live_enabled,
        "live_as_of": live.live_as_of,
        "live_signal_asof": live.live_signal_asof,
        "live_entry_date": live.live_entry_date,
        "market": live.market,
        "provider": live.provider,
        "live_train_mode": live.live_train_mode,
        "min_symbols_per_date": live.min_symbols_per_date,
    }


def _backtest_selection_kwargs(
    feature_target: TrainEvalFeatureTarget,
    backtest: TrainEvalBacktestSettings,
) -> dict[str, Any]:
    return {
        "price_col": feature_target.price_col,
        "backtest_top_k": backtest.backtest_top_k,
        "label_shift_days": backtest.label_shift_days,
        "backtest_weighting": backtest.backtest_weighting,
        "backtest_buffer_exit": backtest.backtest_buffer_exit,
        "backtest_buffer_entry": backtest.backtest_buffer_entry,
        "backtest_long_only": backtest.backtest_long_only,
        "backtest_short_k": backtest.backtest_short_k,
        "backtest_tradable_col": backtest.backtest_tradable_col,
        "backtest_group_col": backtest.backtest_group_col,
        "backtest_max_names_per_group": backtest.backtest_max_names_per_group,
        "backtest_liquidity_floor_col": backtest.backtest_liquidity_floor_col,
        "backtest_liquidity_floor_quantile": backtest.backtest_liquidity_floor_quantile,
        "backtest_weighting_liquidity_col": backtest.backtest_weighting_liquidity_col,
        "backtest_max_turnover_per_rebalance": backtest.backtest_max_turnover_per_rebalance,
        "backtest_selection_tiebreak_col": backtest.backtest_selection_tiebreak_col,
        "backtest_selection_score_bucket_size": backtest.backtest_selection_score_bucket_size,
        "backtest_selection_score_margin": backtest.backtest_selection_score_margin,
        "backtest_selection_score_margin_rank_limit": (
            backtest.backtest_selection_score_margin_rank_limit
        ),
        "post_buffer_exposure_repair": backtest.backtest_post_buffer_exposure_repair,
        "cash_gross_overlay": backtest.backtest_cash_gross_overlay,
        "freshness_overlay": backtest.backtest_freshness_overlay,
        "backtest_preserve_gross_exposure": backtest.backtest_preserve_gross_exposure,
        "execution_model": backtest.execution_model,
        "execution_sim_config": backtest.execution_sim_config,
    }


def _period_evaluation_kwargs(
    data: TrainEvalData,
    period: TrainEvalPeriodSettings,
) -> dict[str, Any]:
    return {
        "rebalance_frequency": period.rebalance_frequency,
        "sample_on_rebalance_dates": period.sample_on_rebalance_dates,
        "valid_dates_set": data.valid_dates_set,
        "perm_test_enabled": period.perm_test_enabled,
        "perm_test_runs": period.perm_test_runs,
        "perm_test_seed": period.perm_test_seed,
        "label_horizon_mode": period.label_horizon_mode,
        "label_horizon_effective": period.label_horizon_effective,
        "n_quantiles": period.n_quantiles,
        "top_k": period.top_k,
        "eval_buffer_exit": period.eval_buffer_exit,
        "eval_buffer_entry": period.eval_buffer_entry,
        "transaction_cost_bps": period.transaction_cost_bps,
        "bucket_ic_enabled": period.bucket_ic_enabled,
        "bucket_ic_schemes": period.bucket_ic_schemes,
        "bucket_ic_method": period.bucket_ic_method,
        "bucket_ic_min_count": period.bucket_ic_min_count,
    }


def _backtest_reporting_kwargs(
    data: TrainEvalData,
    feature_target: TrainEvalFeatureTarget,
    backtest: TrainEvalBacktestSettings,
    services: TrainEvalServices,
) -> dict[str, Any]:
    return {
        "backtest_rebalance_frequency": backtest.backtest_rebalance_frequency,
        "backtest_enabled": backtest.backtest_enabled,
        "backtest_signal_direction_raw": backtest.backtest_signal_direction_raw,
        "backtest_cost_bps_effective": backtest.backtest_cost_bps_effective,
        "backtest_trading_days_per_year": backtest.backtest_trading_days_per_year,
        "backtest_exit_mode": backtest.backtest_exit_mode,
        "backtest_exit_horizon_days": backtest.backtest_exit_horizon_days,
        "backtest_pricing_df": data.backtest_pricing_df,
        "backtest_exit_price_policy": backtest.backtest_exit_price_policy,
        "backtest_exit_fallback_policy": backtest.backtest_exit_fallback_policy,
        "benchmark_df": data.benchmark_df,
        "benchmark_return_series": data.benchmark_return_series,
        "industry_source_df": data.industry_source_df,
        "fundamentals_mcap_col": feature_target.fundamentals_mcap_col,
        "passthrough_cols": data.passthrough_cols,
        "industry_keep_columns": data.industry_keep_columns,
        "price_passthrough_cols": data.price_passthrough_cols,
        "bucket_cols": data.bucket_cols,
        "backtest_topk_fn": services.backtest_topk_fn,
        "bucket_ic_summary_fn": services.bucket_ic_summary_fn,
        "walk_forward_backtest_fn": services.walk_forward_backtest_fn,
        "period_eval_fn": services.period_eval_fn,
        "live_snapshot_fn": services.live_snapshot_fn,
        "trainer_backend": services.trainer_backend,
        "experiment_recorder": services.experiment_recorder,
    }


def _walk_forward_kwargs(
    period: TrainEvalPeriodSettings,
    walk_forward: TrainEvalWalkForwardSettings,
) -> dict[str, Any]:
    return {
        "rolling_windows_months": period.rolling_windows_months,
        "recency_windows": period.recency_windows,
        "wf_enabled": walk_forward.wf_enabled,
        "wf_n_windows": walk_forward.wf_n_windows,
        "wf_test_size": walk_forward.wf_test_size,
        "wf_step_size": walk_forward.wf_step_size,
        "effective_gap_steps": walk_forward.effective_gap_steps,
        "wf_anchor_end": walk_forward.wf_anchor_end,
        "wf_feature_top_k": walk_forward.wf_feature_top_k,
        "wf_backtest_enabled": walk_forward.wf_backtest_enabled,
        "wf_perm_test_enabled": walk_forward.wf_perm_test_enabled,
        "wf_perm_test_runs": walk_forward.wf_perm_test_runs,
        "wf_perm_test_seed": walk_forward.wf_perm_test_seed,
    }


@dataclass(frozen=True)
class TrainEvalRequest:
    data: TrainEvalData
    feature_target: TrainEvalFeatureTarget
    model: TrainEvalModelSettings
    signal: TrainEvalSignalSettings
    live: TrainEvalLiveSettings
    backtest: TrainEvalBacktestSettings
    period: TrainEvalPeriodSettings
    walk_forward: TrainEvalWalkForwardSettings
    services: TrainEvalServices

    def to_kwargs(self) -> dict[str, Any]:
        return {
            **_data_window_kwargs(self.data),
            **_feature_model_kwargs(self.feature_target, self.model),
            **_signal_live_kwargs(self.signal, self.live),
            **_backtest_selection_kwargs(self.feature_target, self.backtest),
            **_period_evaluation_kwargs(self.data, self.period),
            **_backtest_reporting_kwargs(
                self.data,
                self.feature_target,
                self.backtest,
                self.services,
            ),
            **_walk_forward_kwargs(self.period, self.walk_forward),
        }
