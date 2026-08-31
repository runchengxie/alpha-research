"""A-share style-factor and style-signal computation owners.

Core factor transforms and style signals are pure DataFrame-in / value-out
computation. Data loading, publication, reporting, and delivery live in their
respective owners (see ADR-0006).
"""

from __future__ import annotations

from .factor_calc import (
    FACTOR_COLS,
    VALUE_CLUSTER_COL,
    VALUE_CLUSTER_MEMBERS,
    compute_factors,
    standardize_factor_panel,
)
from .helpers import add_new_factors, merge_sw_industry_pit
from .size_style import (
    CROWDING_LOOKBACK_DAYS,
    LARGE_LOW_THRESHOLD,
    MOMENTUM_WINDOWS,
    SMALL_HIGH_THRESHOLD,
    SizeStyleSignal,
    compute_crowding_series,
    compute_size_style_signal,
)

__all__ = [
    "CROWDING_LOOKBACK_DAYS",
    "FACTOR_COLS",
    "LARGE_LOW_THRESHOLD",
    "MOMENTUM_WINDOWS",
    "SMALL_HIGH_THRESHOLD",
    "SizeStyleSignal",
    "VALUE_CLUSTER_COL",
    "VALUE_CLUSTER_MEMBERS",
    "add_new_factors",
    "compute_crowding_series",
    "compute_factors",
    "compute_size_style_signal",
    "merge_sw_industry_pit",
    "standardize_factor_panel",
]
