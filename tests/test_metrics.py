import numpy as np
import pandas as pd

from alpha_research.metrics import (
    assign_daily_quantile_bucket,
    bucket_ic_summary,
    hit_rate,
    normalize_bucket_schemes,
    normalize_window_months,
    regression_error_metrics,
    topk_positive_ratio,
)


def test_normalize_window_months_and_bucket_schemes() -> None:
    assert normalize_window_months([12, 6, 6, 0], [3]) == [6, 12]
    assert normalize_bucket_schemes(
        ["industry", {"col": "size", "bins": 4}, None]
    ) == [
        {"name": "industry", "column": "industry", "type": "category", "n_bins": 0},
        {"name": "size", "column": "size", "type": "category", "n_bins": 4},
    ]


def test_regression_error_metrics_basic():
    y_true = pd.Series([1.0, 2.0])
    y_pred = pd.Series([1.0, 1.0])
    stats = regression_error_metrics(y_true, y_pred)
    assert stats["n"] == 2
    assert np.isclose(stats["mae"], 0.5)
    assert np.isclose(stats["rmse"], np.sqrt(0.5))
    assert np.isclose(stats["r2"], -1.0)


def test_regression_error_metrics_empty():
    stats = regression_error_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
    assert stats["n"] == 0
    assert np.isnan(stats["mae"])
    assert np.isnan(stats["rmse"])
    assert np.isnan(stats["r2"])


def test_hit_rate_and_topk_positive_ratio():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 4),
            "symbol": ["A", "B", "C", "D"],
            "pred": [0.2, -0.1, 0.0, 0.3],
            "target": [0.1, -0.05, 0.0, -0.2],
        }
    )
    stats = hit_rate(df["target"], df["pred"])
    assert stats["n"] == 4
    assert np.isclose(stats["hit_rate"], 0.75)

    topk_stats = topk_positive_ratio(df, "pred", "target", k=2)
    assert topk_stats["n_dates"] == 1
    assert np.isclose(topk_stats["topk_positive_ratio"], 0.5)


def test_bucket_ic_summary_quantile():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 4),
            "symbol": ["A", "B", "C", "D"] * 2,
            "pred": [1, 2, 3, 4, 4, 3, 2, 1],
            "target": [1, 2, 3, 4, 4, 3, 2, 1],
            "mcap": [10, 20, 30, 40, 10, 20, 30, 40],
        }
    )
    df["mcap_bucket"] = assign_daily_quantile_bucket(df, "mcap", n_bins=2)
    summary = bucket_ic_summary(df, "target", "pred", "mcap_bucket")
    assert not summary.empty
    assert "mean" in summary.columns


def test_topk_positive_ratio_insufficient_symbols_returns_empty():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 2),
            "symbol": ["A", "B"],
            "pred": [0.2, 0.1],
            "target": [0.1, -0.1],
        }
    )
    stats = topk_positive_ratio(df, "pred", "target", k=3)
    assert stats["n_dates"] == 0
    assert np.isnan(stats["topk_positive_ratio"])
