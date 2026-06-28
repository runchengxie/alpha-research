from __future__ import annotations

import pandas as pd
import pytest

from cstree.alpha.benchmarking import build_benchmark_series


def test_build_benchmark_series_compounds_daily_returns_over_periods() -> None:
    returns = pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        name="benchmark_return",
    )
    period_info = [
        {
            "entry_date": pd.Timestamp("2024-01-01"),
            "exit_date": pd.Timestamp("2024-01-04"),
        }
    ]

    series, used_periods = build_benchmark_series(
        None,
        "open",
        "close",
        period_info,
        benchmark_return_series=returns,
    )

    assert used_periods == period_info
    assert round(float(series.iloc[0]), 8) == round((1.01 * 1.02 * 0.99) - 1.0, 8)


def test_build_benchmark_series_uses_price_columns_when_returns_missing() -> None:
    benchmark = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-05"]),
            "open": [100.0, 105.0],
            "close": [101.0, 110.0],
        }
    )
    period_info = [
        {
            "entry_date": pd.Timestamp("2024-01-02"),
            "exit_date": pd.Timestamp("2024-01-05"),
        }
    ]

    series, used_periods = build_benchmark_series(benchmark, "open", "close", period_info)

    assert used_periods == period_info
    assert float(series.iloc[0]) == pytest.approx(0.1)
