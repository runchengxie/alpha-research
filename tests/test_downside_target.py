import pandas as pd
import pytest


def test_downside_rms_excludes_entry_move_and_keeps_positive_days_in_denominator():
    from alpha_research.downside_target import next_close_downside_target

    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.Series([5.0, 10.0, 9.0, 9.9], index=dates)
    value, end = next_close_downside_target(prices, dates[0], 2, dates)
    assert value == pytest.approx(0.1 / 2**0.5)
    assert end == dates[3]


def test_missing_session_does_not_stretch_downside_target():
    from alpha_research.downside_target import next_close_downside_target

    dates = pd.date_range("2020-01-01", periods=5)
    prices = pd.Series([5.0, 10.0, 8.0, 9.0], index=dates.delete(2))
    value, end = next_close_downside_target(prices, dates[0], 2, dates)
    assert pd.isna(value)
    assert end == dates[3]


def test_unmatured_target_is_unavailable():
    from alpha_research.downside_target import next_close_downside_target

    dates = pd.date_range("2020-01-01", periods=4)
    value, end = next_close_downside_target(pd.Series([1.0] * 4, index=dates), dates[-1], 2, dates)
    assert pd.isna(value) and pd.isna(end)


def test_trailing_downside_uses_decision_close_but_never_future_prices():
    from alpha_research.downside_target import trailing_downside_rms

    dates = pd.date_range("2020-01-01", periods=5)
    prices = pd.Series([10.0, 9.0, 9.9, 1.0, 2.0], index=dates)
    assert trailing_downside_rms(prices, dates[2], 2, dates) == pytest.approx(0.1 / 2**0.5)
    prices.iloc[3:] = 1000.0
    assert trailing_downside_rms(prices, dates[2], 2, dates) == pytest.approx(0.1 / 2**0.5)
    assert pd.isna(trailing_downside_rms(prices, dates[0], 2, dates))
