import pandas as pd

from alpha_research.period_evaluation import _build_scored_data


def test_build_scored_data_can_retain_model_feature_columns() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"]),
            "symbol": ["AAA"],
            "close": [10.0],
            "future_return": [0.01],
            "pred": [0.5],
            "signal_eval": [0.5],
            "signal_backtest": [0.5],
            "momentum_20d": [1.2],
            "quality": [3.4],
        }
    )

    scored = _build_scored_data(
        frame,
        price_col="close",
        target="future_return",
        price_passthrough_cols=[],
        passthrough_cols=[],
        bucket_cols=[],
        feature_cols=["momentum_20d", "quality", "missing_feature"],
        backtest_tradable_col=None,
    )

    assert scored.columns.tolist() == [
        "trade_date",
        "symbol",
        "close",
        "future_return",
        "pred",
        "signal_eval",
        "signal_backtest",
        "momentum_20d",
        "quality",
    ]


def test_build_scored_data_keeps_freshness_overlay_audit_columns() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "close": [10.0],
            "target": [0.1],
            "pred": [0.2],
            "signal_eval": [0.2],
            "signal_backtest": [0.7],
            "signal_backtest_base": [0.6],
            "signal_backtest_freshness_volume_rank": [0.9],
        }
    )

    scored = _build_scored_data(
        frame,
        price_col="close",
        target="target",
        price_passthrough_cols=[],
        passthrough_cols=[],
        bucket_cols=[],
        feature_cols=[],
        backtest_tradable_col=None,
    )

    assert "signal_backtest_base" in scored.columns
    assert "signal_backtest_freshness_volume_rank" in scored.columns
