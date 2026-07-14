# ruff: noqa: RUF002
"""A-leg scoring for StyleReplica-A80B20-v0.

A-leg targets high-elasticity, active-growth AI hardware chain stocks.
Scores are computed as weighted cross-sectional percentile ranks:

    S_A = 0.25·R_RESVOL + 0.15·R_Liquidity + 0.05·R_VolumeActivity + 0.15·R_SmallSize
        + 0.15·R_Mom20 + 0.10·R_Beta + 0.10·R_Mom120 + 0.05·R_IndustryMom

All R_* values are daily cross-sectional percentile ranks (0–1).
Higher score = better candidate for A-leg.
"""

from __future__ import annotations

import pandas as pd

# ── A-leg weights ──────────────────────────────────────────────────────────────

_A_WEIGHTS: dict[str, float] = {
    "resvol": 0.25,
    "liquidity": 0.15,
    "volume_activity": 0.05,  # minute-level intraday volume intensity
    "size": 0.15,  # already negated in factor computation
    "mom20": 0.15,
    "beta": 0.10,
    "mom120": 0.10,
    "industry_mom": 0.05,
}
_A_OPTIONAL_FACTORS = frozenset({"volume_activity"})

# Note: volume_activity is optional — if missing, liquidity weight is effectively higher
_A_TIEBREAK_WEIGHT = 0.01
_A_TIEBREAK_FACTOR = "resvol"


def _validate_factors(factor_map: dict[str, pd.DataFrame]) -> None:
    missing = [
        factor
        for factor in _A_WEIGHTS
        if factor not in factor_map and factor not in _A_OPTIONAL_FACTORS
    ]
    if missing:
        raise ValueError(f"Missing A-leg factors: {missing}")


def _rank_factor(factor_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank (0–1) within each date."""
    return factor_df.rank(axis=1, method="average", pct=True, na_option="bottom")


def compute_score_a(
    factor_map: dict[str, pd.DataFrame],
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute A-leg composite score for all stocks on all dates.

    Args:
        factor_map: Dict of factor name → wide DataFrame (dates × symbols).
        weights: Optional custom weights; defaults to A-leg design weights.

    Returns:
        Wide DataFrame (dates × symbols) of A-leg composite scores (0–1 range).
        Higher = better A-leg candidate.
    """
    _validate_factors(factor_map)
    w = weights or _A_WEIGHTS

    # Build weighted sum of ranked factors
    dates = factor_map["resvol"].index
    symbols = factor_map["resvol"].columns
    composite = pd.DataFrame(0.0, index=dates, columns=symbols, dtype=float)
    total_weight = 0.0

    for factor_name, weight in w.items():
        factor_df = factor_map.get(factor_name)
        if factor_df is None or factor_df.empty:
            continue
        aligned = factor_df.reindex(index=dates, columns=symbols)
        ranked = _rank_factor(aligned)
        composite += weight * ranked.fillna(0.0)
        total_weight += weight

    # Add tiny tiebreak component
    if _A_TIEBREAK_FACTOR in factor_map:
        tb = _rank_factor(factor_map[_A_TIEBREAK_FACTOR].reindex(index=dates, columns=symbols))
        composite += _A_TIEBREAK_WEIGHT * tb.fillna(0.0)
        total_weight += _A_TIEBREAK_WEIGHT

    if total_weight > 0:
        composite = composite / total_weight

    return composite.clip(0.0, 1.0)


def compute_score_a_with_explanations(
    factor_map: dict[str, pd.DataFrame],
    *,
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute A-leg score and return per-factor ranked contributions.

    Returns:
        (composite_score, per_factor_contributions) where contributions is a dict
        of factor_name → percentile contribution (weight × rank).
    """
    composite = compute_score_a(factor_map, weights=weights)
    w = weights or _A_WEIGHTS

    dates = composite.index
    symbols = composite.columns
    contributions: dict[str, pd.DataFrame] = {}

    for factor_name, weight in w.items():
        factor_df = factor_map.get(factor_name)
        if factor_df is None or factor_df.empty:
            continue
        aligned = factor_df.reindex(index=dates, columns=symbols)
        ranked = _rank_factor(aligned)
        contributions[factor_name] = weight * ranked.fillna(0.0)

    return composite, contributions
