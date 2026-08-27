"""Rule-based StyleReplica alpha signal research package."""

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
from .signal_generator import (
    StyleReplicaConfig,
    StyleReplicaSignalGenerator,
    generate_daily_signals,
)
from .theme_map import map_stock_to_theme
from .universe import filter_style_replica_universe

__all__ = [
    "StyleReplicaConfig",
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
