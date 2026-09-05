from __future__ import annotations

import pandas as pd
import pytest

from alpha_research.stable_compounder import (
    build_quarterly_operating_panel,
    build_rolling_stability_labels,
    select_latest_pit_report_events,
)


def test_pit_selector_rejects_missing_as_of_date():
    frame = pd.DataFrame(columns=pd.Index(["symbol", "report_period", "available_date"]))
    with pytest.raises(ValueError, match="as_of"):
        select_latest_pit_report_events(frame, as_of_date="NaT")


def test_quarterly_panel_converts_ytd_flows_without_future_dates() -> None:
    periods = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"])
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "report_period": periods,
            "available_date": periods + pd.Timedelta(days=30),
            "trade_date": periods + pd.Timedelta(days=30),
            "revenue": [10.0, 25.0, 42.0, 70.0],
            "n_income_attr_p": [1.0, 2.5, 4.0, 7.0],
            "n_cashflow_act": [2.0, 4.5, 7.0, 11.0],
        }
    )
    result = build_quarterly_operating_panel(frame)
    assert result["standalone_revenue"].tolist() == [10.0, 15.0, 17.0, 28.0]
    assert result["standalone_n_income_attr_p"].tolist() == [1.0, 1.5, 1.5, 3.0]
    assert result["standalone_n_cashflow_act"].tolist() == [2.0, 2.5, 2.5, 4.0]


def test_stability_label_requires_twelve_contiguous_positive_quarters() -> None:
    periods = pd.date_range("2021-03-31", periods=12, freq="QE")
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 12,
            "report_period": periods,
            "quarter_index": pd.Series(periods).dt.year * 4 + pd.Series(periods).dt.quarter,
            "standalone_n_income_attr_p": 1.0,
            "standalone_n_cashflow_act": 1.2,
            "standalone_cfo_margin": 0.12,
            "standalone_cfo_to_profit": 1.2,
        }
    )
    result = build_rolling_stability_labels(frame)
    assert result["stable_compounder_strict"].iloc[-1]
    assert result["positive_cfo_ratio"].iloc[-1] == 1.0


def test_stability_label_does_not_treat_missing_cashflow_as_negative() -> None:
    periods = pd.date_range("2021-03-31", periods=12, freq="QE")
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 12,
            "report_period": periods,
            "quarter_index": pd.Series(periods).dt.year * 4 + pd.Series(periods).dt.quarter,
            "standalone_n_income_attr_p": 1.0,
            "standalone_n_cashflow_act": [1.2] * 11 + [None],
            "standalone_cfo_margin": 0.12,
            "standalone_cfo_to_profit": 1.2,
        }
    )
    result = build_rolling_stability_labels(frame)
    assert result["positive_cfo_ratio"].iloc[-1] == 1.0
    assert result["operating_quarters_observed"].iloc[-1] == 11
    assert not result["stable_compounder_strict"].iloc[-1]


def test_select_latest_pit_event_is_as_of_safe() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "report_period": ["2023-12-31"] * 3,
            "available_date": ["2024-03-01", "2024-04-01", "2025-01-01"],
            "value": [1, 2, 3],
        }
    )
    result = select_latest_pit_report_events(frame, as_of_date="2024-06-01")
    assert result["value"].tolist() == [2]
