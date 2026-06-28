from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from cstree.pipeline.contracts import (
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

from cstree.alpha import train_eval_stage
from cstree.alpha.train_eval_stage import _industry_exposure_columns


def _empty_train_eval_data(
    *,
    passthrough_cols: list[str],
    industry_keep_columns: list[str],
) -> TrainEvalData:
    return TrainEvalData(
        train_df=pd.DataFrame(),
        test_df=pd.DataFrame(),
        test_dates=np.array([]),
        df_features=pd.DataFrame(),
        df_full=pd.DataFrame(),
        df_model_sorted=pd.DataFrame(),
        all_dates=np.array([]),
        all_date_start_rows=np.array([]),
        all_date_end_rows=np.array([]),
        all_date_to_pos={},
        valid_dates_set=set(),
        backtest_pricing_df=pd.DataFrame(),
        benchmark_df=None,
        benchmark_return_series=pd.Series(dtype=float),
        industry_source_df=pd.DataFrame(),
        passthrough_cols=passthrough_cols,
        industry_keep_columns=industry_keep_columns,
        price_passthrough_cols=[],
        bucket_cols=[],
    )


def _request() -> TrainEvalRequest:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    frame = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["A", "B"],
            "f1": [0.0, 1.0],
            "target": [0.0, 1.0],
        }
    )
    return TrainEvalRequest(
        data=TrainEvalData(
            train_df=frame,
            test_df=frame,
            test_dates=dates.to_numpy(),
            df_features=frame,
            df_full=frame,
            df_model_sorted=frame,
            all_dates=dates.to_numpy(),
            all_date_start_rows=np.array([0, 1]),
            all_date_end_rows=np.array([1, 2]),
            all_date_to_pos={pd.Timestamp("2020-01-01"): 0},
            valid_dates_set={pd.Timestamp("2020-01-01")},
            backtest_pricing_df=frame,
            benchmark_df=None,
            benchmark_return_series=pd.Series(dtype=float),
            industry_source_df=pd.DataFrame(),
            passthrough_cols=[],
            industry_keep_columns=[],
            price_passthrough_cols=[],
            bucket_cols=[],
        ),
        feature_target=TrainEvalFeatureTarget(
            features=["f1"],
            target="target",
            train_target="target",
            price_col="close",
            fundamentals_mcap_col="market_cap",
        ),
        model=TrainEvalModelSettings(
            model_type="ridge",
            model_params={"alpha": 1.0},
            model_cfg={"type": "ridge", "params": {"alpha": 1.0}},
            sample_weight_mode="none",
            sample_weight_params={},
            n_splits=2,
            embargo_steps=0,
            purge_steps=0,
            cv_purge_mode="gap",
            train_window_mode="full",
            train_window_size=None,
            train_window_unit="dates",
        ),
        signal=TrainEvalSignalSettings(
            signal_direction_mode="fixed",
            signal_direction=1.0,
            min_abs_ic_to_flip=0.0,
            score_postprocess_method="none",
            score_postprocess_columns=[],
            score_postprocess_strength=1.0,
            score_postprocess_min_obs=5,
            report_train_ic=True,
        ),
        live=TrainEvalLiveSettings(
            live_enabled=False,
            live_as_of=None,
            market="hk",
            provider="rqdata",
            live_train_mode="full",
            min_symbols_per_date=1,
        ),
        backtest=TrainEvalBacktestSettings(
            backtest_top_k=1,
            label_shift_days=1,
            backtest_weighting="equal",
            backtest_buffer_exit=0,
            backtest_buffer_entry=0,
            backtest_long_only=True,
            backtest_short_k=None,
            backtest_tradable_col=None,
            backtest_group_col=None,
            backtest_max_names_per_group=None,
            backtest_liquidity_floor_col=None,
            backtest_liquidity_floor_quantile=None,
            backtest_weighting_liquidity_col="medadv20_amount",
            backtest_max_turnover_per_rebalance=None,
            backtest_post_buffer_exposure_repair={"enabled": False},
            backtest_cash_gross_overlay={"enabled": False},
            backtest_freshness_overlay={"enabled": False},
            backtest_preserve_gross_exposure=False,
            execution_model={},
            execution_sim_config={},
            backtest_rebalance_frequency="D",
            backtest_enabled=False,
            backtest_signal_direction_raw=None,
            backtest_cost_bps_effective=0.0,
            backtest_trading_days_per_year=252,
            backtest_exit_mode="rebalance",
            backtest_exit_horizon_days=1,
            backtest_exit_price_policy="strict",
            backtest_exit_fallback_policy="none",
        ),
        period=TrainEvalPeriodSettings(
            rebalance_frequency="D",
            sample_on_rebalance_dates=False,
            perm_test_enabled=False,
            perm_test_runs=1,
            perm_test_seed=None,
            label_horizon_mode="fixed",
            label_horizon_effective=1,
            n_quantiles=2,
            top_k=1,
            eval_buffer_exit=0,
            eval_buffer_entry=0,
            transaction_cost_bps=0.0,
            bucket_ic_enabled=False,
            bucket_ic_schemes=[],
            bucket_ic_method="spearman",
            bucket_ic_min_count=0,
            rolling_windows_months=[],
            recency_windows=["6m", "1m", "1w"],
        ),
        walk_forward=TrainEvalWalkForwardSettings(
            wf_enabled=False,
            wf_n_windows=0,
            wf_test_size=None,
            wf_step_size=None,
            effective_gap_steps=0,
            wf_anchor_end=True,
            wf_feature_top_k=1,
            wf_backtest_enabled=False,
            wf_perm_test_enabled=False,
            wf_perm_test_runs=1,
            wf_perm_test_seed=None,
        ),
        services=TrainEvalServices(
            backtest_topk_fn=lambda *args, **kwargs: None,
            bucket_ic_summary_fn=lambda *args, **kwargs: None,
        ),
    )


def test_industry_exposure_columns_prioritize_industry_labels_over_pit_metadata() -> None:
    data = _empty_train_eval_data(
        passthrough_cols=[
            "report_period",
            "disclosure_date",
            "available_date",
            "industry_name",
            "industry_system",
        ],
        industry_keep_columns=["industry_name", "industry_code"],
    )

    assert _industry_exposure_columns(data) == [
        "industry_name",
        "industry_code",
        "industry_system",
    ]


def test_run_train_eval_stage_accepts_contract_request(monkeypatch) -> None:
    captured = {}

    def _fake_impl(request: TrainEvalRequest) -> dict[str, bool]:
        captured.update(request.to_kwargs())
        return {"ok": True}

    monkeypatch.setattr(train_eval_stage, "_run_train_eval_stage_impl", _fake_impl)

    assert train_eval_stage.run_train_eval_stage(request=_request()) == {"ok": True}
    assert captured["features"] == ["f1"]
    assert captured["model_type"] == "ridge"


def test_run_train_eval_stage_accepts_legacy_kwargs(monkeypatch) -> None:
    captured = {}

    def _fake_impl(request: TrainEvalRequest) -> dict[str, bool]:
        captured.update(request.to_kwargs())
        return {"ok": True}

    monkeypatch.setattr(train_eval_stage, "_run_train_eval_stage_impl", _fake_impl)

    assert train_eval_stage.run_train_eval_stage(**_request().to_kwargs()) == {"ok": True}
    assert captured["features"] == ["f1"]
    assert captured["model_type"] == "ridge"


def test_eval_context_includes_backtest_postprocess_controls() -> None:
    request = _request()
    request = replace(
        request,
        backtest=replace(
            request.backtest,
            backtest_max_turnover_per_rebalance=0.25,
            backtest_post_buffer_exposure_repair={
                "enabled": True,
                "max_abs_momentum_active": 1.0,
            },
            backtest_cash_gross_overlay={
                "enabled": True,
                "default_gross_multiplier": 0.9,
            },
            backtest_freshness_overlay={
                "enabled": True,
                "name": "volume_only_lambda_0p05",
                "lambda": 0.05,
            },
            backtest_preserve_gross_exposure=True,
        ),
    )

    period_context = train_eval_stage._build_period_eval_context(
        request,
        live_state={"positions_by_rebalance_live": None},
        updated_signal_direction=1.0,
        backtest_signal_direction=1.0,
    )
    walk_forward_context = train_eval_stage._build_walk_forward_context(
        request,
        updated_signal_direction=1.0,
    )

    for context in (period_context, walk_forward_context):
        assert context["backtest_max_turnover_per_rebalance"] == 0.25
        assert context["post_buffer_exposure_repair"]["enabled"] is True
        assert context["cash_gross_overlay"]["default_gross_multiplier"] == 0.9
        assert context["freshness_overlay"]["name"] == "volume_only_lambda_0p05"
        assert context["backtest_preserve_gross_exposure"] is True


def test_run_train_eval_stage_honors_main_permutation_switch(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(train_eval_stage, "time_series_cv_ic", lambda *args, **kwargs: [])

    def _fake_fit(*args, **kwargs) -> train_eval_stage._TrainFitResult:
        return train_eval_stage._TrainFitResult(
            model=object(),
            train_eval_df=pd.DataFrame(),
            updated_signal_direction=1.0,
            train_signal_col="signal",
            train_ic_raw_stats={},
            train_ic_series=pd.Series(dtype=float),
            train_ic_stats={},
            train_pearson_ic_series=pd.Series(dtype=float),
            train_pearson_ic_stats={},
            cv_scores_adj=None,
        )

    def _fake_live(*args, **kwargs) -> dict[str, object]:
        return {
            "live_as_of": None,
            "live_signal_asof": None,
            "live_entry_date": None,
            "live_execution_calendar": None,
            "live_execution_open": False,
            "live_execution_status": "disabled",
            "positions_by_rebalance_live": None,
            "live_positions_ready": False,
        }

    def _fake_evaluate(*args, **kwargs) -> dict[str, object]:
        captured["run_perm_test"] = kwargs["run_perm_test"]
        return {
            "ic_series": pd.Series(dtype=float),
            "ic_stats": {},
            "pearson_ic_series": pd.Series(dtype=float),
            "pearson_ic_stats": {},
            "error_metrics": {},
            "hit_rate": {},
            "topk_positive_ratio": {},
            "bucket_ic": [],
            "quantile_ts": pd.DataFrame(),
            "quantile_mean": {},
            "turnover_series": pd.Series(dtype=float),
            "scored_data": pd.DataFrame(),
            "eval_rebalance_dates": [],
            "backtest_rebalance_dates": [],
            "positions_by_rebalance": None,
            "execution_sim_summary": None,
            "execution_sim_orders": pd.DataFrame(),
            "execution_sim_fills": pd.DataFrame(),
            "bt_stats": None,
            "bt_net_series": pd.Series(dtype=float),
            "bt_gross_series": pd.Series(dtype=float),
            "bt_turnover_series": pd.Series(dtype=float),
            "bt_benchmark_series": pd.Series(dtype=float),
            "bt_active_series": pd.Series(dtype=float),
            "bt_benchmark_stats": None,
            "bt_active_stats": None,
            "bt_periods": pd.DataFrame(),
            "bt_style_exposure": pd.DataFrame(),
            "bt_style_exposure_summary": {},
            "bt_industry_exposure": pd.DataFrame(),
            "bt_industry_exposure_summary": {},
            "bt_active_exposure_summary": {},
            "perm_stats": None,
        }

    monkeypatch.setattr(train_eval_stage, "_fit_model_and_score_train", _fake_fit)
    monkeypatch.setattr(train_eval_stage, "_prepare_live_snapshot", _fake_live)
    monkeypatch.setattr(train_eval_stage, "_evaluate_period", _fake_evaluate)
    monkeypatch.setattr(
        train_eval_stage,
        "_run_walk_forward_evaluation",
        lambda *args, **kwargs: ([], pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        train_eval_stage,
        "feature_importance_frame",
        lambda *args, **kwargs: (
            pd.DataFrame({"feature": ["f1"], "importance": [0.0]}),
            "test",
        ),
    )

    result = train_eval_stage.run_train_eval_stage(request=_request())

    assert captured["run_perm_test"] is False
    assert result["perm_stats"] is None


def test_run_train_eval_stage_rejects_mixed_request_and_kwargs() -> None:
    with pytest.raises(TypeError, match="either request or keyword"):
        train_eval_stage.run_train_eval_stage(request=_request(), train_df=pd.DataFrame())
