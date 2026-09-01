import numpy as np
import pandas as pd
import pytest

from alpha_research.daily_watch20_pit_features import GROWTH_FEATURES, QUALITY_FEATURES


def test_family_registry_reuses_canonical_qg_and_has_no_overlap() -> None:
    from alpha_research.daily_watch20_fundamental_families import (
        FUND_CONTEXT_FEATURES,
        FUNDAMENTAL_FAMILY_SCHEMA,
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


def test_value_panel_requires_all_owner_input_columns() -> None:
    from alpha_research.daily_watch20_fundamental_families import build_value_feature_panel

    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-28"]),
            "symbol": ["A"],
            "pb": [1.0],
        }
    )

    with pytest.raises(ValueError, match=r"pe_ttm.*ps_ttm"):
        build_value_feature_panel(frame)


def test_value_panel_rejects_duplicate_stock_date_rows() -> None:
    from alpha_research.daily_watch20_fundamental_families import build_value_feature_panel

    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-28", "2026-08-28"]),
            "symbol": ["A", "A"],
            "pb": [1.0, 1.0],
            "pe_ttm": [10.0, 10.0],
            "ps_ttm": [2.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate stock-date"):
        build_value_feature_panel(frame)


def test_ablation_baseline_removes_existing_value_features_without_mutating_p0() -> None:
    from alpha_research.daily_watch20_fundamental_families import (
        VALUE_FEATURES,
        family_ablation_feature_sets,
    )

    production = ("mom_20", "value_yield", "earnings_yield", "size_pct")

    sets = family_ablation_feature_sets(production)

    assert sets["P0"] == production
    assert sets["T0"] == ("mom_20", "size_pct")
    assert sets["V"] == ("mom_20", "size_pct", *VALUE_FEATURES)
    assert set(sets) == {"P0", "T0", "V", "Q", "G", "VQ", "VG", "QG", "VQG"}
    assert production == ("mom_20", "value_yield", "earnings_yield", "size_pct")


def test_ablation_builder_rejects_p0_without_current_value_anchor() -> None:
    from alpha_research.daily_watch20_fundamental_families import (
        family_ablation_feature_sets,
    )

    with pytest.raises(ValueError, match="current value anchor"):
        family_ablation_feature_sets(("mom_20", "size_pct"))


def test_fundamental_horizon_profiles_are_frozen_and_horizon_aware() -> None:
    from alpha_research.daily_watch20_fundamental_families import (
        fundamental_horizon_profiles,
    )

    profiles = fundamental_horizon_profiles()
    assert set(profiles) == {5, 20, 60}
    assert profiles[5].role == "diagnostic"
    assert profiles[20].role == "primary"
    assert profiles[60].role == "slow_challenger"
    for horizon, profile in profiles.items():
        assert profile.horizon_days == horizon
        assert profile.forward_days == horizon
        assert profile.label_horizon_weights == ((horizon, 1.0),)
        assert profile.embargo_trade_days == horizon
        assert profile.rebalance_trade_days == horizon


@pytest.mark.parametrize("horizon", [5, 20, 60])
def test_horizon_profile_builds_matching_daily_watch20_feature_config(horizon: int) -> None:
    from alpha_research.daily_watch20_features import DailyWatch20FeatureConfig
    from alpha_research.daily_watch20_fundamental_families import (
        fundamental_horizon_profiles,
    )

    profile = fundamental_horizon_profiles()[horizon]
    cfg = DailyWatch20FeatureConfig(
        forward_days=profile.forward_days,
        label_horizon_weights=profile.label_horizon_weights,
    )
    assert cfg.label_col == f"forward_rank_{horizon}d"
    assert cfg.forward_return_col == f"forward_return_{horizon}d"


def test_research_family_features_do_not_enter_production_feature_tuple() -> None:
    from alpha_research.daily_watch20_features import DAILY_WATCH20_FEATURES
    from alpha_research.daily_watch20_fundamental_families import VALUE_FEATURES

    assert set(VALUE_FEATURES).isdisjoint(DAILY_WATCH20_FEATURES)
    assert set(QUALITY_FEATURES).isdisjoint(DAILY_WATCH20_FEATURES)
    assert set(GROWTH_FEATURES).isdisjoint(DAILY_WATCH20_FEATURES)
