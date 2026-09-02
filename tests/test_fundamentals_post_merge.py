import numpy as np
import pandas as pd

from alpha_research.fundamentals_post_merge import derive_post_merge_fundamental_features


def test_derives_report_and_valuation_age_days():
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-06-10", "2026-06-10"]),
            "report_trade_date": ["2026-06-01", None],
            "valuation_trade_date": ["2026-06-08", "2026-06-09"],
        }
    )

    result, features = derive_post_merge_fundamental_features(
        frame,
        ["days_since_report", "valuation_age_days"],
    )

    assert result is frame
    assert result["days_since_report"].iloc[0] == 9
    assert np.isnan(result["days_since_report"].iloc[1])
    assert result["valuation_age_days"].tolist() == [2, 1]
    assert features == ["days_since_report", "valuation_age_days"]


def test_derives_positive_log_market_cap_and_auto_adds_feature():
    frame = pd.DataFrame({"total_mv": [100.0, 0.0, -1.0, np.nan]})

    result, features = derive_post_merge_fundamental_features(
        frame,
        ["ret_20"],
        log_market_cap=True,
        market_cap_col="total_mv",
        log_market_cap_col="log_mcap",
        auto_add_log_market_cap=True,
    )

    assert result["log_mcap"].iloc[0] == np.log(100.0)
    assert result["log_mcap"].iloc[1:].isna().all()
    assert features == ["ret_20", "log_mcap"]


def test_does_not_add_log_market_cap_when_source_column_is_missing():
    frame = pd.DataFrame({"trade_date": pd.to_datetime(["2026-06-10"])})

    result, features = derive_post_merge_fundamental_features(
        frame,
        ["ret_20"],
        log_market_cap=True,
        market_cap_col="total_mv",
        log_market_cap_col="log_mcap",
        auto_add_log_market_cap=True,
    )

    assert "log_mcap" not in result.columns
    assert features == ["ret_20"]


def test_missing_age_source_columns_leave_requested_features_unchanged():
    frame = pd.DataFrame({"trade_date": pd.to_datetime(["2026-06-10"])})

    result, features = derive_post_merge_fundamental_features(
        frame,
        ["days_since_report", "valuation_age_days"],
    )

    assert "days_since_report" not in result.columns
    assert "valuation_age_days" not in result.columns
    assert features == ["days_since_report", "valuation_age_days"]
