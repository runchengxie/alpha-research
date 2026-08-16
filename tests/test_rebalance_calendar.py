from __future__ import annotations

from typing import cast

import pandas as pd

from alpha_research.rebalance_calendar import (
    estimate_rebalance_gap,
    get_rebalance_dates,
    sample_rebalance_frame,
)


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
    rebalance_dates = [
        cast(pd.Timestamp, pd.Timestamp("2024-01-05")),
        cast(pd.Timestamp, pd.Timestamp("2024-01-12")),
    ]

    assert estimate_rebalance_gap(trade_dates, rebalance_dates) == 5.0


def test_sample_rebalance_frame_sorts_and_filters_dates() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2024-01-12",
                    "2024-01-05",
                    "2024-01-19",
                    "2024-01-12",
                ]
            ),
            "symbol": ["CCC", "AAA", "DDD", "BBB"],
            "pred": [3.0, 1.0, 4.0, 2.0],
        }
    )

    sampled, rebalance_dates = sample_rebalance_frame(
        frame,
        frequency="W",
        valid_dates={
            cast(pd.Timestamp, pd.Timestamp("2024-01-05")),
            cast(pd.Timestamp, pd.Timestamp("2024-01-12")),
        },
        allowed_dates=pd.DatetimeIndex(["2024-01-12", "2024-01-19"]),
    )

    assert rebalance_dates == [pd.Timestamp("2024-01-12")]
    assert sampled["symbol"].tolist() == ["CCC", "BBB"]


def test_sample_rebalance_frame_handles_empty_input() -> None:
    frame = pd.DataFrame(columns=pd.Index(["trade_date", "symbol"]))

    sampled, rebalance_dates = sample_rebalance_frame(frame, frequency="W")

    assert sampled.empty
    assert sampled.columns.tolist() == ["trade_date", "symbol"]
    assert rebalance_dates == []
