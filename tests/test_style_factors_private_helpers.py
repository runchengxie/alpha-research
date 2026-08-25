from __future__ import annotations

import pandas as pd

from alpha_research.style_factors.factor_calc import (
    EARNINGS_STABILITY_COL,
    _overlay_formation_fundamentals,
    _prepare_fundamentals,
    _winsorize,
)
from alpha_research.style_factors.helpers._aux import _merge_aux


def test_prepare_fundamentals_computes_stability_on_report_rows() -> None:
    rows = []
    quarter_ends = pd.date_range("2022-03-31", periods=8, freq="QE")
    for quarter, end_date in enumerate(quarter_ends):
        rows.append(
            {
                "symbol": "000001",
                "end_date": end_date,
                "ann_date": end_date + pd.Timedelta(days=30),
                "roe": 10.0,
                "netprofit_yoy": float(quarter),
            }
        )

    prepared = _prepare_fundamentals(pd.DataFrame(rows))

    assert prepared[EARNINGS_STABILITY_COL].iloc[:3].isna().all()
    assert prepared[EARNINGS_STABILITY_COL].iloc[3:].notna().all()


def test_winsorize_is_cross_sectional_by_trade_date() -> None:
    dates = pd.Series([pd.Timestamp("2024-01-02")] * 3 + [pd.Timestamp("2024-01-03")] * 3)
    values = pd.Series([1.0, 2.0, 1000.0, 10.0, 20.0, 30.0])

    winsorized = _winsorize(values, dates)

    assert winsorized.iloc[2] > winsorized.iloc[5]
    assert winsorized.iloc[2] < 1000.0


def test_daily_auxiliary_values_are_not_forward_filled() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        }
    )
    auxiliary = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "trade_date": pd.to_datetime(["2024-01-02"]),
            "event": [1.0],
        }
    )

    merged = _merge_aux(panel, auxiliary, ["event"])

    assert merged.loc[0, "event"] == 1.0
    assert pd.isna(merged.loc[1, "event"])


def test_pit_panel_overlays_non_null_fields_including_growth_inputs() -> None:
    date = pd.Timestamp("2024-01-31")
    legacy = pd.DataFrame(
        {
            "trade_date": [date],
            "symbol": ["000001.SZ"],
            "roe": [8.0],
            "debt_to_assets": [60.0],
            "netprofit_yoy": [12.0],
        }
    )
    panel = pd.DataFrame(
        {
            "trade_date": [date],
            "symbol": ["000001.SZ"],
            "roe": [10.0],
            "debt_to_assets": [float("nan")],
            "netprofit_yoy": [18.0],
        }
    )

    result, used = _overlay_formation_fundamentals(legacy, panel)

    assert used
    assert result.loc[0, "roe"] == 10.0
    assert result.loc[0, "debt_to_assets"] == 60.0
    assert result.loc[0, "netprofit_yoy"] == 18.0
