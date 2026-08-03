import pandas as pd

from alpha_research.feature_dataset import (
    _build_rebalance_tail_candidate_dates,
    _prepare_feature_dataset,
)
from alpha_research.research_dataset import ResearchDataset


def test_rebalance_tail_candidate_dates_keep_recent_fallback_dates():
    dates = pd.date_range("2020-01-01", "2020-02-28", freq="B")

    candidates = _build_rebalance_tail_candidate_dates(
        list(dates),
        rebalance_frequency="M",
        tail_days_per_period=3,
    )

    assert candidates == set(
        pd.to_datetime(
            [
                "2020-01-29",
                "2020-01-30",
                "2020-01-31",
                "2020-02-26",
                "2020-02-27",
                "2020-02-28",
            ]
        )
    )


def test_rebalance_tail_candidate_dates_support_biweekly_frequency():
    dates = pd.bdate_range("2020-01-01", "2020-02-14")

    candidates = _build_rebalance_tail_candidate_dates(
        list(dates),
        rebalance_frequency="2W",
        tail_days_per_period=2,
    )

    assert candidates == set(
        pd.to_datetime(
            [
                "2020-01-02",
                "2020-01-03",
                "2020-01-16",
                "2020-01-17",
                "2020-01-30",
                "2020-01-31",
                "2020-02-13",
                "2020-02-14",
            ]
        )
    )


def test_prepare_feature_dataset_prefilters_engineered_rows_but_keeps_daily_pricing():
    dates = pd.date_range("2020-01-01", "2020-02-28", freq="B")
    rows = []
    for date_idx, date in enumerate(dates):
        for symbol_idx, symbol in enumerate(["AAA", "BBB"]):
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "close": 100.0 + date_idx + symbol_idx,
                    "amount": 1_000_000.0 + symbol_idx,
                    "f1": float(symbol_idx + 1),
                }
            )
    df = pd.DataFrame(rows)

    state = _prepare_feature_dataset(
        df=df,
        features=["f1"],
        feature_params={},
        price_col="close",
        target="future_return",
        label_shift_days=0,
        label_horizon_days=1,
        label_horizon_mode="fixed",
        label_next_rebalance_map=None,
        fundamentals_allow_missing=False,
        bucket_ic_enabled=False,
        bucket_ic_schemes=[],
        feature_missing_features=[],
        feature_missing_method="none",
        feature_missing_add_indicators=False,
        feature_missing_suffix="_missing",
        fundamentals_cols=[],
        industry_cols=[],
        execution_pricing_cols={"amount"},
        backtest_tradable_col=None,
        universe_by_date=None,
        winsorize_pct=None,
        cs_method="none",
        cs_winsorize_pct=None,
        train_target="future_return",
        train_target_transform="none",
        sample_on_rebalance_dates=True,
        rebalance_frequency="M",
        min_symbols_per_date=1,
    )

    expected_model_dates = pd.to_datetime(["2020-01-31", "2020-02-27"]).to_numpy()
    assert state["all_dates_model_full"].tolist() == expected_model_dates.tolist()
    assert state["reference_trade_dates"].tolist() == dates.to_numpy().tolist()
    assert len(state["backtest_pricing_df"]) == len(df)
    assert set(state["df_features"]["trade_date"].unique()) == set(expected_model_dates)
    lifecycle = state["dataset_lifecycle"]
    assert isinstance(state["research_dataset"], ResearchDataset)
    assert lifecycle["metadata"]["backend"] == "native"
    assert lifecycle["metadata"]["raw_daily_panel_rows"] == len(df)
    assert lifecycle["metadata"]["engineered_feature_label_rows"] < len(df)
    assert "future_return" not in state["research_dataset"].fetch_infer().columns


def test_prepare_feature_dataset_keeps_extra_passthrough_out_of_model_features():
    dates = pd.date_range("2020-01-01", "2020-02-28", freq="B")
    rows = []
    for date_idx, date in enumerate(dates):
        for symbol_idx, symbol in enumerate(["AAA", "BBB"]):
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "close": 100.0 + date_idx + symbol_idx,
                    "f1": float(symbol_idx + 1),
                    "heat_capacity_prior": float(date_idx - symbol_idx),
                }
            )
    df = pd.DataFrame(rows)

    state = _prepare_feature_dataset(
        df=df,
        features=["f1"],
        feature_params={},
        price_col="close",
        target="future_return",
        label_shift_days=0,
        label_horizon_days=1,
        label_horizon_mode="fixed",
        label_next_rebalance_map=None,
        fundamentals_allow_missing=False,
        bucket_ic_enabled=False,
        bucket_ic_schemes=[],
        feature_missing_features=[],
        feature_missing_method="none",
        feature_missing_add_indicators=False,
        feature_missing_suffix="_missing",
        fundamentals_cols=[],
        industry_cols=[],
        extra_passthrough_cols=["heat_capacity_prior"],
        execution_pricing_cols=set(),
        backtest_tradable_col=None,
        universe_by_date=None,
        winsorize_pct=None,
        cs_method="none",
        cs_winsorize_pct=None,
        train_target="future_return",
        train_target_transform="none",
        sample_on_rebalance_dates=True,
        rebalance_frequency="M",
        min_symbols_per_date=1,
    )

    assert "heat_capacity_prior" not in state["features"]
    assert "heat_capacity_prior" in state["passthrough_cols"]
    assert "heat_capacity_prior" in state["df_full"].columns


def test_prepare_feature_dataset_keeps_no_label_live_scoring_date_out_of_training():
    dates = pd.date_range("2020-01-01", "2020-02-28", freq="B")
    rows = []
    for date_idx, date in enumerate(dates):
        for symbol_idx, symbol in enumerate(["AAA", "BBB"]):
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "close": 100.0 + date_idx + symbol_idx,
                    "f1": float(symbol_idx + 1),
                }
            )
    df = pd.DataFrame(rows)

    state = _prepare_feature_dataset(
        df=df,
        features=["f1"],
        feature_params={},
        price_col="close",
        target="future_return",
        label_shift_days=0,
        label_horizon_days=1,
        label_horizon_mode="fixed",
        label_next_rebalance_map=None,
        fundamentals_allow_missing=False,
        bucket_ic_enabled=False,
        bucket_ic_schemes=[],
        feature_missing_features=[],
        feature_missing_method="none",
        feature_missing_add_indicators=False,
        feature_missing_suffix="_missing",
        fundamentals_cols=[],
        industry_cols=[],
        execution_pricing_cols=set(),
        backtest_tradable_col=None,
        universe_by_date=None,
        winsorize_pct=None,
        cs_method="none",
        cs_winsorize_pct=None,
        train_target="future_return",
        train_target_transform="none",
        sample_on_rebalance_dates=True,
        rebalance_frequency="M",
        min_symbols_per_date=1,
        extra_sample_dates_without_target=[pd.Timestamp("2020-02-28")],
    )

    live_date = pd.Timestamp("2020-02-28")
    assert live_date in set(state["df_features"]["trade_date"].unique())
    assert live_date not in set(state["df_full"]["trade_date"].unique())
    assert live_date not in set(state["df_model_all"]["trade_date"].unique())
