"""A-share style-factor proxy computation (19 candidate factors).

Size, Value, Momentum, Quality (composite), Earnings Yield, LowVol, Growth,
Leverage, Beta, Liquidity, plus locally-landed tushare auxiliary factors
(liquidity flow, chip concentration, institution holding, public-fund top-10
breadth, public-fund top-10 breadth change, public-fund top-10 ownership,
public-fund top-10 ownership change, dividend yield, PS value). Pure
DataFrame-in / DataFrame-out computation; data loading and reporting live in
their respective owners (see ADR-0006 R4 slice 7).
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

__all__ = [
    "FACTOR_COLS",
    "VALUE_CLUSTER_COL",
    "VALUE_CLUSTER_MEMBERS",
    "add_new_factors",
    "compute_factors",
    "merge_sw_industry_pit",
    "standardize_factor_panel",
]
