# ruff: noqa: RUF002
"""B-leg scoring for StyleReplica-A80B20-v0.

B-leg targets low-volatility convergence stocks as a stabilizing supplement.
Scores are computed as weighted cross-sectional percentile ranks:

    S_B = 0.25·R_VolConvergence + 0.10·R_HermiteStability + 0.20·R_LowRESVOL
        + 0.20·R_Liquidity + 0.15·R_Mom20 + 0.10·R_Mom120

Where:
- VolConvergence = -(Vol_20 / Vol_120) — short vol declining vs long vol
- LowRESVOL = negated RESVOL — lower residual vol → higher score
- Liquidity, Mom20, Mom120 are standard factors

All R_* values are daily cross-sectional percentile ranks (0–1).
Higher score = better candidate for B-leg.
"""

from __future__ import annotations

import pandas as pd

# ── B-leg weights ──────────────────────────────────────────────────────────────

_B_WEIGHTS: dict[str, float] = {
    "vol_convergence": 0.25,
    "hermite_stability": 0.10,  # Cross-day volume-activity stability; higher is steadier.
    "resvol": 0.20,  # note: RESVOL itself is negated for B-leg internally
    "liquidity": 0.20,
    "mom20": 0.15,
    "mom120": 0.10,
}
_B_OPTIONAL_FACTORS = frozenset({"hermite_stability"})

_B_TIEBREAK_WEIGHT = 0.01
_B_TIEBREAK_FACTOR = "vol_convergence"


def _validate_factors(factor_map: dict[str, pd.DataFrame]) -> None:
    missing = [
        factor
        for factor in _B_WEIGHTS
        if factor not in factor_map and factor not in _B_OPTIONAL_FACTORS
    ]
    if missing:
        raise ValueError(f"Missing B-leg factors: {missing}")


def _rank_factor(factor_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank (0–1) within each date."""
    return factor_df.rank(axis=1, method="average", pct=True, na_option="bottom")


def compute_score_b(
    factor_map: dict[str, pd.DataFrame],
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute B-leg composite score for all stocks on all dates.

    B-leg prefers LOWER residual volatility, so we negate the RESVOL factor
    before ranking (or equivalently, rank the raw value and treat low as high).

    Args:
        factor_map: Dict of factor name → wide DataFrame (dates × symbols).
                    Must contain: vol_convergence, resvol, liquidity, mom20, mom120.
        hermite_stability is optional — if missing, vol_convergence weight is effectively higher.
        weights: Optional custom weights; defaults to B-leg design weights.

    Returns:
        Wide DataFrame (dates × symbols) of B-leg composite scores (0–1 range).
        Higher = better B-leg candidate.
    """
    _validate_factors(factor_map)
    w = weights or _B_WEIGHTS

    dates = factor_map["vol_convergence"].index
    symbols = factor_map["vol_convergence"].columns
    composite = pd.DataFrame(0.0, index=dates, columns=symbols, dtype=float)
    total_weight = 0.0

    for factor_name, weight in w.items():
        factor_df = factor_map.get(factor_name)
        if factor_df is None or factor_df.empty:
            continue
        aligned = factor_df.reindex(index=dates, columns=symbols)

        ranked = _rank_factor(-aligned) if factor_name == "resvol" else _rank_factor(aligned)

        composite += weight * ranked.fillna(0.0)
        total_weight += weight

    # Tiebreak
    if _B_TIEBREAK_FACTOR in factor_map:
        tb = _rank_factor(factor_map[_B_TIEBREAK_FACTOR].reindex(index=dates, columns=symbols))
        composite += _B_TIEBREAK_WEIGHT * tb.fillna(0.0)
        total_weight += _B_TIEBREAK_WEIGHT

    if total_weight > 0:
        composite = composite / total_weight

    return composite.clip(0.0, 1.0)


def compute_score_b_with_explanations(
    factor_map: dict[str, pd.DataFrame],
    *,
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute B-leg score and return per-factor ranked contributions."""
    composite = compute_score_b(factor_map, weights=weights)
    w = weights or _B_WEIGHTS

    dates = composite.index
    symbols = composite.columns
    contributions: dict[str, pd.DataFrame] = {}

    for factor_name, weight in w.items():
        factor_df = factor_map.get(factor_name)
        if factor_df is None or factor_df.empty:
            continue
        aligned = factor_df.reindex(index=dates, columns=symbols)

        ranked = _rank_factor(-aligned) if factor_name == "resvol" else _rank_factor(aligned)

        contributions[factor_name] = weight * ranked.fillna(0.0)

    return composite, contributions
