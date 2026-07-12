from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cstree.alpha.daily_watch20_features import (
    DAILY_WATCH20_FEATURES,
    MINUTE_FEATURES,
    DailyWatch20FeatureConfig,
    build_daily_watch20_feature_frame,
)


def _daily_panel(
    n_dates: int = 36,
    symbols: tuple[str, ...] = ("000001.SZ", "600000.SH"),
) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=n_dates)
    rows: list[dict[str, object]] = []
    for symbol_number, symbol in enumerate(symbols):
        for date_number, trade_date in enumerate(dates):
            if symbol_number == 0:
                price = 10.0 + date_number
            elif symbol.endswith(".BJ"):
                price = 20.0 + 2.0 * date_number
            else:
                price = 80.0 - 0.2 * date_number + symbol_number
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "adj_open": price,
                    "tr_close": price * (1.0 + 0.002 * ((date_number % 3) - 1)),
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "amount": float((symbol_number + 1) * 100_000_000 + date_number),
                    "turnover_rate": 1.0 + symbol_number * 0.1,
                    "total_mv": float((symbol_number + 1) * 10_000_000_000),
                    "pb": 1.5 + symbol_number * 0.1,
                    "pe_ttm": 10.0 + symbol_number,
                    "listed_days": 100 + date_number,
                    "is_st": False,
                    "is_suspended": False,
                }
            )
    return pd.DataFrame(rows)


def _row(frame: pd.DataFrame, date: pd.Timestamp, symbol: str) -> pd.Series:
    selected = frame.loc[frame["trade_date"].eq(date) & frame["symbol"].eq(symbol)]
    assert len(selected) == 1
    return selected.iloc[0]


def test_feature_config_has_an_honest_fixed_five_day_contract() -> None:
    with pytest.raises(ValueError, match="fixed at 5"):
        DailyWatch20FeatureConfig(forward_days=3)
    with pytest.raises(ValueError, match="non-negative integer"):
        DailyWatch20FeatureConfig(minute_lag_trade_days=0.5)  # type: ignore[arg-type]


def test_rolling_features_are_symbol_local_and_input_order_invariant() -> None:
    daily = _daily_panel()
    shuffled = daily.sample(frac=1.0, random_state=7)
    shuffled.index = np.arange(len(shuffled)) * 11 + 3

    expected = build_daily_watch20_feature_frame(daily)
    actual = build_daily_watch20_feature_frame(shuffled)

    columns = ["trade_date", "symbol", "ret_1d", "mom_5", "vol_5", "amount_log_20"]
    assert_frame_equal(actual[columns], expected[columns])
    first_rows = actual.groupby("symbol", sort=False).head(1)
    assert first_rows["ret_1d"].isna().all()
    assert set(DAILY_WATCH20_FEATURES).issubset(actual.columns)


@pytest.mark.parametrize(("lag", "source_number", "target_number"), [(0, 4, 4), (1, 4, 5)])
def test_minute_features_carry_an_explicit_nonfuture_source_date(
    lag: int,
    source_number: int,
    target_number: int,
) -> None:
    daily = _daily_panel()
    dates = pd.Index(daily["trade_date"].unique()).sort_values()
    minute = pd.DataFrame(
        {
            "source_trade_date": dates,
            "ts_code": "000001.SZ",
            "minute_realized_vol": np.arange(len(dates), dtype=float),
        }
    )

    features = build_daily_watch20_feature_frame(
        daily,
        minute,
        config=DailyWatch20FeatureConfig(minute_lag_trade_days=lag),
    )

    selected = _row(features, pd.Timestamp(dates[target_number]), "000001.SZ")
    assert selected["minute_source_date"] == dates[source_number]
    assert selected["minute_realized_vol"] == float(source_number)
    assert selected["minute_source_date"] <= selected["trade_date"]
    assert bool(selected["minute_feature_available"])
    assert set(MINUTE_FEATURES).issubset(features.columns)


def test_next_open_label_uses_t_plus_one_to_t_plus_six_and_eligible_universe() -> None:
    symbols = ("000001.SZ", "600000.SH", "830001.BJ")
    daily = _daily_panel(symbols=symbols)
    dates = pd.Index(daily["trade_date"].unique()).sort_values()

    features = build_daily_watch20_feature_frame(daily)

    signal_number = 25
    signal_date = pd.Timestamp(dates[signal_number])
    a_share = _row(features, signal_date, "000001.SZ")
    sh_share = _row(features, signal_date, "600000.SH")
    bse_share = _row(features, signal_date, "830001.BJ")
    expected_return = (10.0 + signal_number + 6) / (10.0 + signal_number + 1) - 1.0
    assert a_share["forward_return_5d"] == pytest.approx(expected_return)
    assert a_share["liquidity_pct"] == 0.5
    assert a_share["forward_label_start_date"] == dates[signal_number + 1]
    assert a_share["forward_label_end_date"] == dates[signal_number + 6]
    assert a_share["forward_rank_5d"] == 1.0
    assert sh_share["forward_rank_5d"] == 0.5
    assert not bool(bse_share["hard_eligible"])
    assert pd.isna(bse_share["liquidity_pct"])
    assert pd.isna(bse_share["forward_rank_5d"])
    trailing = features.loc[features["symbol"].eq("000001.SZ")].tail(6)
    assert trailing["forward_return_5d"].isna().all()


def test_label_does_not_stretch_across_a_missing_symbol_date() -> None:
    daily = _daily_panel()
    dates = pd.Index(daily["trade_date"].unique()).sort_values()
    signal_number = 25
    missing_entry = daily["trade_date"].eq(dates[signal_number + 1]) & daily["symbol"].eq(
        "000001.SZ"
    )

    features = build_daily_watch20_feature_frame(daily.loc[~missing_entry])

    signal = _row(features, pd.Timestamp(dates[signal_number]), "000001.SZ")
    assert signal["forward_label_start_date"] == dates[signal_number + 1]
    assert signal["forward_label_end_date"] == dates[signal_number + 6]
    assert pd.isna(signal["forward_return_5d"])
    assert pd.isna(signal["forward_rank_5d"])


def test_hard_eligibility_fails_closed_for_status_market_and_data_gaps() -> None:
    symbols = (
        "000001.SZ",
        "600001.SH",
        "600002.SH",
        "600003.SH",
        "600004.SH",
        "600005.SH",
        "830001.BJ",
        "600006.SH",
    )
    daily = _daily_panel(symbols=symbols)
    audit_date = pd.Index(daily["trade_date"].unique()).sort_values()[25]
    daily["is_st"] = daily["is_st"].astype(object)
    daily.loc[daily["symbol"].eq("600006.SH"), "amount"] = 1.0
    daily.loc[daily["trade_date"].eq(audit_date) & daily["symbol"].eq("600001.SH"), "is_st"] = True
    daily.loc[
        daily["trade_date"].eq(audit_date) & daily["symbol"].eq("600002.SH"),
        "is_suspended",
    ] = True
    daily.loc[daily["trade_date"].eq(audit_date) & daily["symbol"].eq("600003.SH"), "is_st"] = ""
    daily.loc[
        daily["trade_date"].eq(audit_date) & daily["symbol"].eq("600004.SH"), "listed_days"
    ] = 10
    daily.loc[daily["trade_date"].eq(audit_date) & daily["symbol"].eq("600005.SH"), "amount"] = 0

    features = build_daily_watch20_feature_frame(daily)
    cross_section = features.loc[features["trade_date"].eq(audit_date)].set_index("symbol")

    assert bool(cross_section.loc["000001.SZ", "hard_eligible"])
    excluded = {
        "600001.SH",
        "600002.SH",
        "600003.SH",
        "600004.SH",
        "600005.SH",
        "600006.SH",
        "830001.BJ",
    }
    assert not cross_section.loc[list(excluded), "hard_eligible"].any()
    assert cross_section.loc[list(excluded), "forward_rank_5d"].isna().all()
