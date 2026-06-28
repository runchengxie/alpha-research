from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .train_eval_contracts import (
    TrainEvalBacktestSettings,
    TrainEvalData,
    TrainEvalFeatureTarget,
    TrainEvalLiveSettings,
    TrainEvalModelSettings,
    TrainEvalPeriodSettings,
    TrainEvalRequest,
    TrainEvalServices,
    TrainEvalSignalSettings,
    TrainEvalWalkForwardSettings,
)


@dataclass
class _TrainEvalKwargReader:
    values: dict[str, Any]
    used: set[str]

    def required(self, name: str) -> Any:
        self.used.add(name)
        try:
            return self.values[name]
        except KeyError as exc:
            raise TypeError(f"Missing train/eval stage field: {name}") from exc

    def optional(self, name: str, default: Any) -> Any:
        self.used.add(name)
        return self.values.get(name, default)


def train_eval_request_from_kwargs(kwargs: dict[str, Any]) -> TrainEvalRequest:
    reader = _TrainEvalKwargReader(values=kwargs, used=set())
    request = TrainEvalRequest(
        data=_train_eval_data_from_kwargs(reader),
        feature_target=_feature_target_from_kwargs(reader),
        model=_model_settings_from_kwargs(reader),
        signal=_signal_settings_from_kwargs(reader),
        live=_live_settings_from_kwargs(reader),
        backtest=_backtest_settings_from_kwargs(reader),
        period=_period_settings_from_kwargs(reader),
        walk_forward=_walk_forward_settings_from_kwargs(reader),
        services=_services_from_kwargs(reader),
    )
    unknown = sorted(set(kwargs) - reader.used)
    if unknown:
        raise TypeError(f"Unexpected train/eval stage fields: {', '.join(unknown)}")
    return request


def _train_eval_data_from_kwargs(reader: _TrainEvalKwargReader) -> TrainEvalData:
    get = reader.required
    return TrainEvalData(
        train_df=get("train_df"),
        test_df=get("test_df"),
        test_dates=get("test_dates"),
        df_features=get("df_features"),
        df_full=get("df_full"),
        df_model_sorted=get("df_model_sorted"),
        all_dates=get("all_dates"),
        all_date_start_rows=get("all_date_start_rows"),
        all_date_end_rows=get("all_date_end_rows"),
        all_date_to_pos=get("all_date_to_pos"),
        valid_dates_set=get("valid_dates_set"),
        backtest_pricing_df=get("backtest_pricing_df"),
        benchmark_df=get("benchmark_df"),
        benchmark_return_series=get("benchmark_return_series"),
        industry_source_df=get("industry_source_df"),
        passthrough_cols=get("passthrough_cols"),
        industry_keep_columns=get("industry_keep_columns"),
        price_passthrough_cols=get("price_passthrough_cols"),
        bucket_cols=get("bucket_cols"),
    )


def _feature_target_from_kwargs(reader: _TrainEvalKwargReader) -> TrainEvalFeatureTarget:
    get = reader.required
    return TrainEvalFeatureTarget(
        features=get("features"),
        target=get("target"),
        train_target=get("train_target"),
        price_col=get("price_col"),
        fundamentals_mcap_col=get("fundamentals_mcap_col"),
    )


def _model_settings_from_kwargs(reader: _TrainEvalKwargReader) -> TrainEvalModelSettings:
    get = reader.required
    return TrainEvalModelSettings(
        model_type=get("model_type"),
        model_params=get("model_params"),
        model_cfg=get("model_cfg"),
        sample_weight_mode=get("sample_weight_mode"),
        sample_weight_params=get("sample_weight_params"),
        n_splits=get("n_splits"),
        embargo_steps=get("embargo_steps"),
        purge_steps=get("purge_steps"),
        cv_purge_mode=reader.optional("cv_purge_mode", "gap"),
        train_window_mode=get("train_window_mode"),
        train_window_size=get("train_window_size"),
        train_window_unit=get("train_window_unit"),
    )


def _signal_settings_from_kwargs(reader: _TrainEvalKwargReader) -> TrainEvalSignalSettings:
    get = reader.required
    return TrainEvalSignalSettings(
        signal_direction_mode=get("signal_direction_mode"),
        signal_direction=get("signal_direction"),
        min_abs_ic_to_flip=get("min_abs_ic_to_flip"),
        score_postprocess_method=get("score_postprocess_method"),
        score_postprocess_columns=get("score_postprocess_columns"),
        score_postprocess_strength=get("score_postprocess_strength"),
        score_postprocess_min_obs=get("score_postprocess_min_obs"),
        report_train_ic=get("report_train_ic"),
    )


def _live_settings_from_kwargs(reader: _TrainEvalKwargReader) -> TrainEvalLiveSettings:
    get = reader.required
    return TrainEvalLiveSettings(
        live_enabled=get("live_enabled"),
        live_as_of=get("live_as_of"),
        live_signal_asof=get("live_signal_asof"),
        live_entry_date=get("live_entry_date"),
        market=get("market"),
        provider=get("provider"),
        live_train_mode=get("live_train_mode"),
        min_symbols_per_date=get("min_symbols_per_date"),
    )


def _backtest_settings_from_kwargs(reader: _TrainEvalKwargReader) -> TrainEvalBacktestSettings:
    get = reader.required
    return TrainEvalBacktestSettings(
        backtest_top_k=get("backtest_top_k"),
        label_shift_days=get("label_shift_days"),
        backtest_weighting=get("backtest_weighting"),
        backtest_buffer_exit=get("backtest_buffer_exit"),
        backtest_buffer_entry=get("backtest_buffer_entry"),
        backtest_long_only=get("backtest_long_only"),
        backtest_short_k=get("backtest_short_k"),
        backtest_tradable_col=get("backtest_tradable_col"),
        backtest_group_col=get("backtest_group_col"),
        backtest_max_names_per_group=get("backtest_max_names_per_group"),
        backtest_liquidity_floor_col=get("backtest_liquidity_floor_col"),
        backtest_liquidity_floor_quantile=get("backtest_liquidity_floor_quantile"),
        backtest_weighting_liquidity_col=get("backtest_weighting_liquidity_col"),
        backtest_max_turnover_per_rebalance=get("backtest_max_turnover_per_rebalance"),
        backtest_selection_tiebreak_col=reader.optional(
            "backtest_selection_tiebreak_col",
            None,
        ),
        backtest_selection_score_bucket_size=reader.optional(
            "backtest_selection_score_bucket_size",
            None,
        ),
        backtest_selection_score_margin=reader.optional(
            "backtest_selection_score_margin",
            None,
        ),
        backtest_selection_score_margin_rank_limit=reader.optional(
            "backtest_selection_score_margin_rank_limit",
            None,
        ),
        backtest_post_buffer_exposure_repair=reader.optional(
            "post_buffer_exposure_repair",
            {"enabled": False},
        ),
        backtest_cash_gross_overlay=reader.optional("cash_gross_overlay", {"enabled": False}),
        backtest_freshness_overlay=reader.optional("freshness_overlay", {"enabled": False}),
        backtest_preserve_gross_exposure=reader.optional(
            "backtest_preserve_gross_exposure",
            False,
        ),
        execution_model=get("execution_model"),
        execution_sim_config=get("execution_sim_config"),
        backtest_rebalance_frequency=get("backtest_rebalance_frequency"),
        backtest_enabled=get("backtest_enabled"),
        backtest_signal_direction_raw=get("backtest_signal_direction_raw"),
        backtest_cost_bps_effective=get("backtest_cost_bps_effective"),
        backtest_trading_days_per_year=get("backtest_trading_days_per_year"),
        backtest_exit_mode=get("backtest_exit_mode"),
        backtest_exit_horizon_days=get("backtest_exit_horizon_days"),
        backtest_exit_price_policy=get("backtest_exit_price_policy"),
        backtest_exit_fallback_policy=get("backtest_exit_fallback_policy"),
    )


def _period_settings_from_kwargs(reader: _TrainEvalKwargReader) -> TrainEvalPeriodSettings:
    get = reader.required
    return TrainEvalPeriodSettings(
        rebalance_frequency=get("rebalance_frequency"),
        sample_on_rebalance_dates=get("sample_on_rebalance_dates"),
        perm_test_enabled=get("perm_test_enabled"),
        perm_test_runs=get("perm_test_runs"),
        perm_test_seed=get("perm_test_seed"),
        label_horizon_mode=get("label_horizon_mode"),
        label_horizon_effective=get("label_horizon_effective"),
        n_quantiles=get("n_quantiles"),
        top_k=get("top_k"),
        eval_buffer_exit=get("eval_buffer_exit"),
        eval_buffer_entry=get("eval_buffer_entry"),
        transaction_cost_bps=get("transaction_cost_bps"),
        bucket_ic_enabled=get("bucket_ic_enabled"),
        bucket_ic_schemes=get("bucket_ic_schemes"),
        bucket_ic_method=get("bucket_ic_method"),
        bucket_ic_min_count=get("bucket_ic_min_count"),
        rolling_windows_months=get("rolling_windows_months"),
        recency_windows=reader.optional("recency_windows", ["6m", "1m", "1w"]),
    )


def _walk_forward_settings_from_kwargs(
    reader: _TrainEvalKwargReader,
) -> TrainEvalWalkForwardSettings:
    get = reader.required
    return TrainEvalWalkForwardSettings(
        wf_enabled=get("wf_enabled"),
        wf_n_windows=get("wf_n_windows"),
        wf_test_size=get("wf_test_size"),
        wf_step_size=get("wf_step_size"),
        effective_gap_steps=get("effective_gap_steps"),
        wf_anchor_end=get("wf_anchor_end"),
        wf_feature_top_k=get("wf_feature_top_k"),
        wf_backtest_enabled=get("wf_backtest_enabled"),
        wf_perm_test_enabled=get("wf_perm_test_enabled"),
        wf_perm_test_runs=get("wf_perm_test_runs"),
        wf_perm_test_seed=get("wf_perm_test_seed"),
    )


def _services_from_kwargs(reader: _TrainEvalKwargReader) -> TrainEvalServices:
    get = reader.required
    return TrainEvalServices(
        backtest_topk_fn=get("backtest_topk_fn"),
        bucket_ic_summary_fn=get("bucket_ic_summary_fn"),
        walk_forward_backtest_fn=get("walk_forward_backtest_fn"),
    )
