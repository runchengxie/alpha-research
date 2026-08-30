import numpy as np
import pandas as pd
import pytest

from alpha_research.daily_watch20_pit_features import GROWTH_FEATURES, QUALITY_FEATURES


def test_family_registry_reuses_canonical_qg_and_has_no_overlap() -> None:
    from alpha_research.daily_watch20_fundamental_families import (
        FUNDAMENTAL_FAMILY_SCHEMA,
        FUND_CONTEXT_FEATURES,
        STYLE_CONTROL_FEATURES,
        VALUE_FEATURES,
        fundamental_family_registry,
    )

    registry = fundamental_family_registry()
    assert FUNDAMENTAL_FAMILY_SCHEMA == "daily_watch20.fundamental_families.research.v1"
    assert registry["value"] == VALUE_FEATURES
    assert registry["quality"] == QUALITY_FEATURES
    assert registry["growth"] == GROWTH_FEATURES
    assert registry["style_controls"] == STYLE_CONTROL_FEATURES
    assert registry["fund_context"] == FUND_CONTEXT_FEATURES

    primary_names = [
        name
        for family in ("value", "quality", "growth")
        for name in registry[family]
    ]
    assert len(primary_names) == len(set(primary_names))


def test_value_panel_uses_positive_finite_denominators_and_same_date_ranks() -> None:
    from alpha_research.daily_watch20_fundamental_families import build_value_feature_panel

    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-28"] * 4),
            "symbol": ["A", "B", "C", "D"],
            "pb": [1.0, 2.0, 0.0, np.inf],
            "pe_ttm": [10.0, 20.0, -5.0, 40.0],
            "ps_ttm": [2.0, 4.0, 8.0, np.nan],
        }
    )

    panel = build_value_feature_panel(frame)
    out = panel.frame.set_index("symbol")

    assert out.loc["A", "value_book_yield_pct"] == pytest.approx(1.0)
    assert out.loc["B", "value_book_yield_pct"] == pytest.approx(0.5)
    assert pd.isna(out.loc["C", "value_book_yield_pct"])
    assert pd.isna(out.loc["D", "value_book_yield_pct"])
    assert out.loc["A", "value_sales_yield_pct"] == pytest.approx(1.0)
    assert panel.receipt["cross_section_transform"] == "same-date percentile rank"
    assert panel.receipt["forward_fill"] is False
