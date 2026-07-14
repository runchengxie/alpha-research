from __future__ import annotations

import pandas as pd

from alpha_research.factor_diagnostics import (
    compute_factor_diagnostics,
    factor_diagnostics_options_from_config,
)


def _scored_data() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-01-01", periods=4, freq="7D")
    symbols = [f"S{idx:03d}" for idx in range(9)]
    for date_idx, date in enumerate(dates):
        for symbol_idx, symbol in enumerate(symbols):
            cap = float(symbol_idx + 1)
            mid_shape = 5.0 - abs(cap - 5.0)
            f_size = cap + date_idx * 0.1
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "future_return": 0.01 * mid_shape + 0.001 * date_idx,
                    "factor_size": f_size,
                    "factor_clone": f_size * 2.0,
                    "factor_mid": mid_shape + date_idx * 0.05,
                    "log_mkt_cap": cap,
                    "ret_20": float(symbol_idx % 3),
                    "industry": "Tech" if symbol_idx < 5 else "Bank",
                }
            )
    return pd.DataFrame(rows)


def test_compute_factor_diagnostics_profiles_top_features() -> None:
    result = compute_factor_diagnostics(
        _scored_data(),
        feature_columns=["factor_size", "factor_clone", "factor_mid"],
        feature_importance=pd.DataFrame(
            {
                "feature": ["factor_size", "factor_clone", "factor_mid"],
                "importance": [0.9, 0.8, 0.7],
            }
        ),
        style_columns=["log_mkt_cap", "ret_20"],
        market_cap_col="log_mkt_cap",
        industry_columns=["industry"],
        top_n=3,
        min_obs=5,
        min_bucket_obs=2,
        autocorr_lags=(1,),
        correlation_threshold=0.95,
    )

    assert result.summary["status"] == "ok"
    assert result.summary["factors"] == 3
    assert result.summary["residual_ic_available"] is True
    assert result.summary["size_bucket_available"] is True
    assert result.by_factor["factor"].tolist() == [
        "factor_size",
        "factor_clone",
        "factor_mid",
    ]
    assert "rank_autocorr_1" in result.by_factor.columns
    assert set(result.size_bucket["size_bucket"]) == {"small", "mid", "large"}
    assert set(result.industry["industry"]) == {"Bank", "Tech"}
    assert not result.style_exposure.empty
    assert not result.residual_ic.empty

    high_corr = result.correlation.loc[
        (result.correlation["factor_a"] == "factor_size")
        & (result.correlation["factor_b"] == "factor_clone")
    ].iloc[0]
    assert bool(high_corr["is_high_corr"]) is True


def test_factor_diagnostics_config_can_disable_or_override_defaults() -> None:
    disabled = factor_diagnostics_options_from_config(
        {"eval": {"factor_diagnostics": {"enabled": False}}}
    )
    assert disabled["enabled"] is False

    options = factor_diagnostics_options_from_config(
        {
            "eval": {
                "factor_diagnostics": {
                    "top_n": 5,
                    "style_columns": ["log_mkt_cap"],
                    "size_buckets": {"count": 3, "labels": ["small", "mid", "large"]},
                    "drift": {"autocorr_lags": [1, 3]},
                }
            }
        }
    )
    assert options["enabled"] is True
    assert options["top_n"] == 5
    assert options["style_columns"] == ["log_mkt_cap"]
    assert options["autocorr_lags"] == (1, 3)
