"""Point-in-time daily and lagged intraday features for DailyWatch20.

This module is a thin public surface for the DailyWatch20 feature
implementation. The historical single-file implementation has been split into
private submodules (``_daily_watch20_features_config`` /
``_daily_watch20_features_calc``), to keep individual files smaller while
preserving the exact public and private symbol surface. Everything below is
re-exported so existing ``alpha_research.daily_watch20_features`` imports keep
working unchanged.
"""

from __future__ import annotations

import pandas as pd

from ._daily_watch20_features_calc import (
    _add_eligibility,
    _add_hermite_stability,
    _add_liquidity_and_style_features,
    _add_market_regime_features,
    _add_market_shadow_features,
    _add_next_open_labels,
    _add_price_features,
    _adjacent_trade_day_returns,
    _equal_weight_market_return,
    _future_date,
    _join_minute_features,
    _known_false_flag,
    _lag_minute_features,
    _lookup_on_dates,
    _numeric_series,
    _open_is_away_from_limit,
    _prepare_daily_input,
    _require_columns,
    _rolling,
    _rolling_ols_slope,
    _rolling_sum,
    _series,
    _target_market,
    _weighted_complete_row_sum,
)
from ._daily_watch20_features_config import (
    BLENDED_FORWARD_RANK_COL,
    BLENDED_FORWARD_RETURN_COL,
    DAILY_WATCH20_FEATURES,
    DAILY_WATCH20_MARKET_SHADOW_DIAGNOSTICS,
    DAILY_WATCH20_MARKET_SHADOW_FEATURES,
    DEFAULT_LABEL_HORIZON_WEIGHTS,
    LEGACY_FIVE_DAY_LABEL_HORIZON_WEIGHTS,
    LIMIT_AWARE_NEXT_OPEN_LABEL_POLICY_ID,
    MINUTE_FEATURES,
    DailyWatch20FeatureConfig,
    label_columns_for_horizon_weights,
    normalize_label_horizon_weights,
)

__all__ = [
    "BLENDED_FORWARD_RANK_COL",
    "BLENDED_FORWARD_RETURN_COL",
    "DAILY_WATCH20_FEATURES",
    "DAILY_WATCH20_MARKET_SHADOW_DIAGNOSTICS",
    "DAILY_WATCH20_MARKET_SHADOW_FEATURES",
    "DEFAULT_LABEL_HORIZON_WEIGHTS",
    "LEGACY_FIVE_DAY_LABEL_HORIZON_WEIGHTS",
    "LIMIT_AWARE_NEXT_OPEN_LABEL_POLICY_ID",
    "MINUTE_FEATURES",
    "DailyWatch20FeatureConfig",
    "_add_eligibility",
    "_add_hermite_stability",
    "_add_liquidity_and_style_features",
    "_add_market_regime_features",
    "_add_market_shadow_features",
    "_add_next_open_labels",
    "_add_price_features",
    "_adjacent_trade_day_returns",
    "_equal_weight_market_return",
    "_future_date",
    "_join_minute_features",
    "_known_false_flag",
    "_lag_minute_features",
    "_lookup_on_dates",
    "_numeric_series",
    "_open_is_away_from_limit",
    "_prepare_daily_input",
    "_require_columns",
    "_rolling",
    "_rolling_ols_slope",
    "_rolling_sum",
    "_series",
    "_target_market",
    "_weighted_complete_row_sum",
    "build_daily_watch20_feature_frame",
    "future_trade_date",
    "known_false_flag",
    "label_columns_for_horizon_weights",
    "lookup_on_dates",
    "normalize_label_horizon_weights",
    "open_is_away_from_limit",
]


def future_trade_date(frame: pd.DataFrame, offset: int) -> pd.Series:
    """Return each row's future trade date at the requested trading-date offset."""

    return _future_date(frame, offset)


def lookup_on_dates(frame: pd.DataFrame, values: pd.Series, dates: pd.Series) -> pd.Series:
    """Lookup symbol-level values on an aligned series of target dates."""

    return _lookup_on_dates(frame, values, dates)


def open_is_away_from_limit(
    open_price: pd.Series,
    limit_price: pd.Series,
    *,
    side: str,
) -> pd.Series:
    """Return whether an open price is tradable away from the side-specific limit."""

    return _open_is_away_from_limit(open_price, limit_price, side=side)


def known_false_flag(values: pd.Series) -> pd.Series:
    """Normalize known false-like values while treating unknown values as false."""

    return _known_false_flag(values)


def build_daily_watch20_feature_frame(
    daily: pd.DataFrame,
    minute_daily: pd.DataFrame | None = None,
    *,
    config: DailyWatch20FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build point-in-time features and next-open multi-horizon rank labels."""

    cfg = config or DailyWatch20FeatureConfig()
    out = _prepare_daily_input(daily)
    out = _add_price_features(out)
    out = _add_liquidity_and_style_features(out)
    if cfg.include_market_shadow_features:
        out = _add_market_shadow_features(out)
    out = _add_market_regime_features(out)
    out = _join_minute_features(out, minute_daily, lag_trade_days=cfg.minute_lag_trade_days)
    out = _add_hermite_stability(out)
    out = _add_eligibility(out, cfg)
    out = _add_next_open_labels(out, horizon_weights=cfg.label_horizon_weights)
    return out.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)
