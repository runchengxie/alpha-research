"""Feature engineering short-series robustness tests.

pandas_ta returns None for series shorter than the indicator window.
Assigning None into a feature column yields an object dtype, which then
pollutes the concatenated feature panel and breaks xgboost training.

These tests guard the _ta_series wrapper so short-listed stocks produce
float NaN features instead of object dtype.
"""

import pandas as pd
import pandas_ta as ta
import pytest

from alpha_research.feature_engineering import (
    _add_rsi_features,
    _add_sma_features,
    _add_volume_features,
    _ta_series,
)


def test_ta_series_returns_float_nan_when_pandas_ta_returns_none():
    short = pd.Series([10.0, 10.1, 10.2])  # shorter than window 20
    out = _ta_series(ta.sma, short, length=20)
    assert isinstance(out, pd.Series)
    assert out.dtype == float
    assert out.isna().all()
    assert list(out.index) == list(short.index)


def test_ta_series_passthrough_normal_series():
    series = pd.Series([10.0, 10.1, 10.2, 10.3, 10.4])
    out = _ta_series(ta.sma, series, length=3)
    assert isinstance(out, pd.Series)
    assert out.dtype == float
    # 前两个窗口为 NaN, 之后有值
    assert out.isna().iloc[:2].all()
    assert not out.isna().iloc[2:].any()


def test_add_sma_features_short_series_stays_float():
    group = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "tr_close": [10.0, 10.1],
        }
    )
    _add_sma_features(
        group,
        features=["sma_20"],
        feature_params={"sma_windows": [20]},
        needed={"sma_20"},
        price_series=group["tr_close"],
    )
    assert group["sma_20"].dtype == float
    assert group["sma_20"].isna().all()


def test_add_rsi_features_short_series_stays_float():
    group = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "tr_close": [10.0, 10.1],
        }
    )
    _add_rsi_features(
        group,
        features=["rsi_7"],
        feature_params={"rsi": [7]},
        price_series=group["tr_close"],
    )
    assert group["rsi_7"].dtype == float
    assert group["rsi_7"].isna().all()


def test_add_volume_features_short_series_stays_float():
    group = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "vol": [100.0, 200.0],
        }
    )
    _add_volume_features(
        group,
        features=["volume_sma20_ratio"],
        feature_params={"volume_sma_windows": [20]},
        needed={"volume_sma20_ratio"},
    )
    assert group["volume_sma20"].dtype == float
    assert group["volume_sma20"].isna().all()


def test_concat_of_short_and_normal_series_keeps_float():
    """Merging short and normal series keeps feature column float."""
    short_group = pd.DataFrame(
        {
            "symbol": ["301707.SZ"] * 2,
            "tr_close": [10.0, 10.1],
        }
    )
    _add_sma_features(
        short_group,
        features=["sma_20"],
        feature_params={"sma_windows": [20]},
        needed={"sma_20"},
        price_series=short_group["tr_close"],
    )

    normal_group = pd.DataFrame(
        {
            "symbol": ["600000.SH"] * 5,
            "tr_close": [10.0, 10.1, 10.2, 10.3, 10.4],
        }
    )
    _add_sma_features(
        normal_group,
        features=["sma_20"],
        feature_params={"sma_windows": [20]},
        needed={"sma_20"},
        price_series=normal_group["tr_close"],
    )

    merged = pd.concat([short_group, normal_group], ignore_index=True)
    assert merged["sma_20"].dtype == float
    # xgboost 只接受数值列, 构造 DMatrix 验证
    pytest.importorskip("xgboost")
    from xgboost import DMatrix

    frame = merged[["sma_20"]].dropna()
    if not frame.empty:
        DMatrix(frame[["sma_20"]])
