from __future__ import annotations

import pandas as pd

from cstree.alpha.rebalance_calendar import estimate_rebalance_gap, get_rebalance_dates


def test_rebalance_calendar_selects_period_ends() -> None:
    dates = pd.date_range("2024-01-01", "2024-01-12", freq="B")

    assert get_rebalance_dates(dates, "W") == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-12"),
    ]


def test_rebalance_calendar_supports_anchored_multiweek_frequency() -> None:
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="B")

    assert get_rebalance_dates(dates, "2W-FRI@2024-01-05") == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-19"),
        pd.Timestamp("2024-01-31"),
    ]


def test_estimate_rebalance_gap_uses_calendar_positions() -> None:
    trade_dates = pd.date_range("2024-01-01", "2024-01-12", freq="B")
    rebalance_dates = [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]

    assert estimate_rebalance_gap(trade_dates, rebalance_dates) == 5.0
