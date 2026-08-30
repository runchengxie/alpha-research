"""Canonical research-only fundamental family contracts for DailyWatch20."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
_VALUE_SOURCE_COLUMNS = {
    "value_book_yield_pct": "pb",
    "value_earnings_yield_pct": "pe_ttm",
    "value_sales_yield_pct": "ps_ttm",
}
_CURRENT_VALUE_ANCHOR_FEATURES = frozenset({"value_yield", "earnings_yield"})


@dataclass(frozen=True)
class ValueFeaturePanel:
    """Research-only same-date valuation ranks and their coverage metadata."""

    frame: pd.DataFrame
    coverage_daily: pd.DataFrame
    receipt: dict[str, object]


def fundamental_family_registry() -> dict[str, tuple[str, ...]]:
    """Return the frozen research family membership without duplicating PIT Q/G definitions."""

    return {
        "value": VALUE_FEATURES,
        "quality": QUALITY_FEATURES,
        "growth": GROWTH_FEATURES,
        "style_controls": STYLE_CONTROL_FEATURES,
        "fund_context": FUND_CONTEXT_FEATURES,
    }


def build_value_feature_panel(frame: pd.DataFrame) -> ValueFeaturePanel:
    """Build PB/PE/PS valuation yields as same-date percentile ranks."""

    required = {"trade_date", "symbol", *_VALUE_SOURCE_COLUMNS.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"value feature frame missing columns: {missing}")
    out = frame.loc[:, ["trade_date", "symbol", *_VALUE_SOURCE_COLUMNS.values()]].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="raise").dt.normalize()
    if out.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("value feature frame contains duplicate stock-date rows")
    observed_columns: list[str] = []
    for target, source in _VALUE_SOURCE_COLUMNS.items():
        denominator = pd.to_numeric(out[source], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        raw_yield = 1.0 / denominator.where(denominator > 0)
        out[target] = raw_yield.groupby(out["trade_date"], sort=False).rank(pct=True)
        observed = f"{target}__observed"
        out[observed] = raw_yield.notna()
        observed_columns.append(observed)
    coverage = out.groupby("trade_date", sort=True)[observed_columns].mean().reset_index()
    receipt: dict[str, object] = {
        "schema_version": FUNDAMENTAL_FAMILY_SCHEMA,
        "status": "research_only",
        "source_columns": list(_VALUE_SOURCE_COLUMNS.values()),
        "value_features": list(VALUE_FEATURES),
        "cross_section_transform": "same-date percentile rank",
        "forward_fill": False,
        "production_feature_schema_changed": False,
    }
    return ValueFeaturePanel(out, coverage, receipt)


def family_ablation_feature_sets(
    production_features: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Build the frozen P0/T0/V/Q/G family experiment matrix."""

    p0 = tuple(production_features)
    if not _CURRENT_VALUE_ANCHOR_FEATURES.issubset(p0):
        raise ValueError("P0 lacks the frozen current value anchor features")
    t0 = tuple(name for name in p0 if name not in _CURRENT_VALUE_ANCHOR_FEATURES)
    result = {
        "P0": p0,
        "T0": t0,
        "V": (*t0, *VALUE_FEATURES),
        "Q": (*t0, *QUALITY_FEATURES),
        "G": (*t0, *GROWTH_FEATURES),
        "VQ": (*t0, *VALUE_FEATURES, *QUALITY_FEATURES),
        "VG": (*t0, *VALUE_FEATURES, *GROWTH_FEATURES),
        "QG": (*t0, *QUALITY_FEATURES, *GROWTH_FEATURES),
        "VQG": (*t0, *VALUE_FEATURES, *QUALITY_FEATURES, *GROWTH_FEATURES),
    }
    for code, features in result.items():
        if len(features) != len(set(features)):
            raise ValueError(f"duplicate features in family arm {code}")
    return result


__all__ = [
    "FUNDAMENTAL_FAMILY_SCHEMA",
    "FUND_CONTEXT_FEATURES",
    "STYLE_CONTROL_FEATURES",
    "VALUE_FEATURES",
    "ValueFeaturePanel",
    "build_value_feature_panel",
    "family_ablation_feature_sets",
    "fundamental_family_registry",
]
