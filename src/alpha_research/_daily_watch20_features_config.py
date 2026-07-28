"""DailyWatch20 feature constants, config dataclass, and label-horizon helpers.

Split out of the historical single-file
:mod:`alpha_research.daily_watch20_features` implementation to keep individual
files smaller while preserving the exact public/private symbol surface. The
feature-building transforms live in
:mod:`alpha_research._daily_watch20_features_calc`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

DEFAULT_LABEL_HORIZON_WEIGHTS = ((1, 0.50), (3, 0.30), (5, 0.20))
LEGACY_FIVE_DAY_LABEL_HORIZON_WEIGHTS = ((5, 1.0),)
LIMIT_AWARE_NEXT_OPEN_LABEL_POLICY_ID = "next_open_unsuspended_open_limit_aware.v3"
BLENDED_FORWARD_RANK_COL = "forward_rank_blended"
BLENDED_FORWARD_RETURN_COL = "forward_return_blended"

DAILY_WATCH20_FEATURES = (
    "ret_1d",
    "mom_5",
    "mom_20",
    "mom_60",
    "mom_120",
    "vol_5",
    "vol_20",
    "vol_60",
    "downside_vol_20",
    "amount_log_20",
    "turnover_20",
    "size_pct",
    "liquidity_pct",
    "low_volatility_pct",
    "vol_convergence_pct",
    "mom20_pct",
    "mom120_pct",
    "value_yield",
    "earnings_yield",
    "range_pct",
    "close_location",
    "market_mom_20",
    "market_vol_20",
    "breadth_20",
    "minute_realized_vol",
    "minute_downside_vol",
    "minute_range_pct",
    "minute_close_location",
    "minute_last_30m_return",
    "minute_open_30m_volume_share",
    "minute_last_30m_volume_share",
    "minute_volume_concentration",
    "minute_active_ratio",
    "minute_price_volume_corr",
    "minute_volume_activity",
    "hermite_stability",
)

# Research-only candidates.  Keeping them outside DAILY_WATCH20_FEATURES is
# deliberate: feature experiments must not mutate the production model schema.
DAILY_WATCH20_MARKET_SHADOW_FEATURES = (
    "beta_60",
    "vol_regime_20_60_pct",
)
DAILY_WATCH20_MARKET_SHADOW_DIAGNOSTICS = ("value_yield_pct",)

_BETA_WINDOW = 60
_BETA_MIN_OBS = 40

_MINUTE_PREFIX = "minute_"
MINUTE_ORIGIN_FEATURES = tuple(
    name for name in DAILY_WATCH20_FEATURES if name.startswith(_MINUTE_PREFIX)
)
DERIVED_MINUTE_FEATURES = ("hermite_stability",)
MINUTE_FEATURES = MINUTE_ORIGIN_FEATURES + DERIVED_MINUTE_FEATURES


@dataclass(frozen=True)
class DailyWatch20FeatureConfig:
    """Feature and label timing for a close-to-next-open daily watchlist."""

    forward_days: int = 5
    label_horizon_weights: Mapping[int, float] | Sequence[tuple[int, float]] = (
        DEFAULT_LABEL_HORIZON_WEIGHTS
    )
    minute_lag_trade_days: int = 0
    min_listed_days: int = 60
    liquidity_floor_quantile: float = 0.20
    include_market_shadow_features: bool = False

    def __post_init__(self) -> None:
        horizon_weights = normalize_label_horizon_weights(self.label_horizon_weights)
        if int(self.forward_days) != max(horizon for horizon, _weight in horizon_weights):
            raise ValueError("forward_days must equal the longest configured label horizon")
        try:
            minute_lag = int(self.minute_lag_trade_days)
            min_listed = int(self.min_listed_days)
        except (TypeError, ValueError) as exc:
            raise ValueError("trade-day settings must be integers") from exc
        if minute_lag != self.minute_lag_trade_days or minute_lag < 0:
            raise ValueError("minute_lag_trade_days must be a non-negative integer")
        if min_listed != self.min_listed_days or min_listed < 0:
            raise ValueError("min_listed_days must be a non-negative integer")
        if not 0 <= self.liquidity_floor_quantile < 1:
            raise ValueError("liquidity_floor_quantile must be in [0, 1)")
        if not isinstance(self.include_market_shadow_features, bool):
            raise ValueError("include_market_shadow_features must be a boolean")
        object.__setattr__(self, "minute_lag_trade_days", minute_lag)
        object.__setattr__(self, "min_listed_days", min_listed)
        object.__setattr__(self, "forward_days", int(self.forward_days))
        object.__setattr__(self, "label_horizon_weights", horizon_weights)

    @property
    def label_col(self) -> str:
        """Training target column implied by the configured horizon mix."""

        return label_columns_for_horizon_weights(self.label_horizon_weights)[0]

    @property
    def forward_return_col(self) -> str:
        """Forward-return column implied by the configured horizon mix."""

        return label_columns_for_horizon_weights(self.label_horizon_weights)[1]


def normalize_label_horizon_weights(
    weights: Mapping[int, float] | Sequence[tuple[int, float]],
) -> tuple[tuple[int, float], ...]:
    """Validate, sort, and normalize positive trading-day horizon weights."""

    items = tuple(weights.items()) if isinstance(weights, Mapping) else tuple(weights)
    if not items:
        raise ValueError("label_horizon_weights must not be empty")
    normalized: list[tuple[int, float]] = []
    seen: set[int] = set()
    for raw_horizon, raw_weight in items:
        try:
            horizon = int(cast(Any, raw_horizon))
            weight = float(cast(Any, raw_weight))
        except (TypeError, ValueError) as exc:
            raise ValueError("label horizons and weights must be numeric") from exc
        if horizon != raw_horizon or horizon <= 0:
            raise ValueError("label horizons must be positive integers")
        if horizon in seen:
            raise ValueError("label horizons must be unique")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("label horizon weights must be finite and positive")
        seen.add(horizon)
        normalized.append((horizon, weight))
    total = sum(weight for _horizon, weight in normalized)
    return tuple((horizon, weight / total) for horizon, weight in sorted(normalized))


def label_columns_for_horizon_weights(
    weights: Mapping[int, float] | Sequence[tuple[int, float]],
) -> tuple[str, str]:
    """Return canonical rank and return target columns for a horizon mix."""

    normalized = normalize_label_horizon_weights(weights)
    if len(normalized) == 1:
        horizon = normalized[0][0]
        return f"forward_rank_{horizon}d", f"forward_return_{horizon}d"
    return BLENDED_FORWARD_RANK_COL, BLENDED_FORWARD_RETURN_COL
