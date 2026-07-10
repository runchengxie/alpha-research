"""StyleReplica-A80B20-v0 — Rule-based AI hardware chain style replication model.

This module implements the first-phase style replica model described in the
StyleReplica-A80B20-v0 design document. It is a **rule-based** (not ML-trained)
daily-frequency signal generator that produces two-leg scores:

- **Leg A** (80 slots): AI hardware active-growth style — theme-quota constrained,
  scored on RESVOL, liquidity, small size, momentum, beta, and industry momentum.
- **Leg B** (20 slots): low-volatility convergence supplement — industry-capped,
  scored on volatility convergence, low RESVOL, liquidity, and momentum.

Output: ``signals_style_replica.parquet`` + ``signals_style_replica.meta.json``
conforming to the ``cstree.signals`` artifact contract.
"""

from .factors import (
    compute_beta_factor,
    compute_hermite_stability_factor,
    compute_industry_momentum,
    compute_liquidity_factor,
    compute_momentum_factor,
    compute_size_factor,
    compute_vol_convergence_factor,
    compute_volume_activity_factor,
)
from .resvol import compute_resvol_factor
from .score_a import compute_score_a
from .score_b import compute_score_b
from .signal_generator import StyleReplicaSignalGenerator, generate_daily_signals
from .theme_map import AI_HARDWARE_THEME_QUOTAS, map_stock_to_theme
from .universe import filter_style_replica_universe

__all__ = [
    "AI_HARDWARE_THEME_QUOTAS",
    "StyleReplicaSignalGenerator",
    "compute_beta_factor",
    "compute_hermite_stability_factor",
    "compute_industry_momentum",
    "compute_liquidity_factor",
    "compute_momentum_factor",
    "compute_resvol_factor",
    "compute_score_a",
    "compute_score_b",
    "compute_size_factor",
    "compute_vol_convergence_factor",
    "compute_volume_activity_factor",
    "filter_style_replica_universe",
    "generate_daily_signals",
    "map_stock_to_theme",
]
