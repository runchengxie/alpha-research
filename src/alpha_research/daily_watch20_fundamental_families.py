"""Canonical research-only fundamental family contracts for DailyWatch20."""

from __future__ import annotations

from .daily_watch20_pit_features import GROWTH_FEATURES, QUALITY_FEATURES

FUNDAMENTAL_FAMILY_SCHEMA = "daily_watch20.fundamental_families.research.v1"
VALUE_FEATURES = (
    "value_book_yield_pct",
    "value_earnings_yield_pct",
    "value_sales_yield_pct",
)
STYLE_CONTROL_FEATURES = (
    "size_pct",
    "liquidity_pct",
    "low_volatility_pct",
)
FUND_CONTEXT_FEATURES = (
    "fund_crowding_level",
    "fund_ownership_change",
    "fund_holder_count_change",
    "fund_low_crowding_accumulation",
    "fund_top10_concentration",
    "fund_accumulation_without_crowding",
)


def fundamental_family_registry() -> dict[str, tuple[str, ...]]:
    """Return the frozen research family membership without duplicating PIT Q/G definitions."""

    return {
        "value": VALUE_FEATURES,
        "quality": QUALITY_FEATURES,
        "growth": GROWTH_FEATURES,
        "style_controls": STYLE_CONTROL_FEATURES,
        "fund_context": FUND_CONTEXT_FEATURES,
    }


__all__ = [
    "FUNDAMENTAL_FAMILY_SCHEMA",
    "FUND_CONTEXT_FEATURES",
    "STYLE_CONTROL_FEATURES",
    "VALUE_FEATURES",
    "fundamental_family_registry",
]
