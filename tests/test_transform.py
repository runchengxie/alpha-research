import numpy as np
import pandas as pd
import pytest

from cstree.alpha.transform import (
    apply_cross_sectional_series_transform,
    apply_cross_sectional_transform,
    apply_score_postprocess,
    neutralize_cross_sectional_series,
    rank_blend_cross_sectional_series,
)


def test_cross_sectional_zscore_by_date():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3),
            "symbol": ["A", "B", "C"] * 2,
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [2.0, 3.0, 4.0, 1.0, 0.0, -1.0],
        }
    )
    out = apply_cross_sectional_transform(df, ["f1", "f2"], method="zscore", winsorize_pct=None)
    for date in out["trade_date"].unique():
        subset = out[out["trade_date"] == date]
        assert np.isclose(subset["f1"].mean(), 0.0, atol=1e-8)
        assert np.isclose(subset["f2"].mean(), 0.0, atol=1e-8)


def test_cross_sectional_series_transform_preserves_missing_values():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "target": [1.0, 2.0, np.nan, 4.0],
        }
    )

    out = apply_cross_sectional_series_transform(df, "target", method="zscore")

    assert np.isclose(float(out.iloc[0]), -1.0)
    assert np.isclose(float(out.iloc[1]), 1.0)
    assert np.isnan(out.iloc[2])
    assert np.isclose(float(out.iloc[3]), 0.0)


def test_cross_sectional_series_transform_can_group_by_date_and_industry():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 4),
            "industry": ["tech", "tech", "bank", "bank"],
            "target": [1.0, 4.0, 2.0, 3.0],
        }
    )

    out = apply_cross_sectional_series_transform(
        df,
        "target",
        method="rank",
        group_cols=["trade_date", "industry"],
    )

    assert out.tolist() == [0.0, 0.5, 0.0, 0.5]


def test_cross_sectional_series_transform_can_create_rank_relevance_labels():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 3),
            "target": [0.01, 0.04, 0.04, -0.02, 0.03, 0.01, 0.02],
        }
    )

    out = apply_cross_sectional_series_transform(df, "target", method="rank_relevance")

    assert out.tolist() == [1.0, 2.0, 2.0, 0.0, 2.0, 0.0, 1.0]
    assert all(float(value).is_integer() for value in out.dropna())


def test_neutralize_cross_sectional_series_removes_linear_size_component():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 4),
            "pred": [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0],
            "log_mcap": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    neutralized = neutralize_cross_sectional_series(
        df,
        "pred",
        ["log_mcap"],
        strength=1.0,
        min_obs=4,
    )
    out = df.assign(pred_adj=neutralized)

    for _, group in out.groupby("trade_date", sort=False):
        assert float(group["pred_adj"].std(ddof=0)) < 1e-8


def test_apply_score_postprocess_strength_zero_returns_original_series():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 3),
            "pred": [1.0, 2.0, 3.0],
            "log_mcap": [10.0, 11.0, 12.0],
        }
    )

    out = apply_score_postprocess(
        df,
        "pred",
        method="neutralize",
        columns=["log_mcap"],
        strength=0.0,
        min_obs=3,
    )

    assert out.tolist() == df["pred"].tolist()


def test_rank_blend_cross_sectional_series_blends_base_and_overlay_ranks_by_date():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-08"] * 3),
            "pred": [1.0, 2.0, 3.0, 3.0, 1.0, 2.0],
            "volume_heat": [30.0, 10.0, 20.0, 10.0, 30.0, 20.0],
            "industry_heat": [3.0, 2.0, 1.0, np.nan, np.nan, np.nan],
        }
    )

    out = rank_blend_cross_sectional_series(
        df,
        "pred",
        ["volume_heat", "industry_heat"],
        strength=0.05,
    )

    base_rank = df.groupby("trade_date")["pred"].rank(method="average", pct=True)
    overlay_ranks = df.groupby("trade_date")[["volume_heat", "industry_heat"]].rank(
        method="average",
        pct=True,
    )
    expected = 0.95 * base_rank + 0.05 * overlay_ranks.mean(axis=1, skipna=True)

    assert out.tolist() == pytest.approx(expected.tolist())


def test_apply_score_postprocess_rank_blend_falls_back_to_base_rank_when_overlay_is_missing():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 3),
            "pred": [1.0, 2.0, 3.0],
            "volume_heat": [np.nan, np.nan, np.nan],
        }
    )

    out = apply_score_postprocess(
        df,
        "pred",
        method="rank_blend",
        columns=["volume_heat"],
        strength=0.05,
    )

    expected = df.groupby("trade_date")["pred"].rank(method="average", pct=True)
    assert out.tolist() == pytest.approx(expected.tolist())
