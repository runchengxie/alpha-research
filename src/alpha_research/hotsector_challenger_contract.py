"""Frozen ranking contract for the historical hot-sector challenger.

Owner note: this module moved from strategy-app to alpha-research as part of
SA-15 (strategy-knowledge repatriation to the compute kernel layer). It is a
pure constant bundle plus an immutable dataclass with no cross-repo imports,
so it belongs in the lowest compute-kernel layer.

The guarded momentum/risk arm may only rerank the numeric Top15 names whose
numeric score is at least 95% of the Numeric Top10 cutoff.  The ``0.95`` value
is exposed on the immutable contract so receipts can record it, but validation
freezes it to prevent a retrospective cutoff sweep.

The producer field ``risk_score`` means intraday stability: higher is better.
Ranking code therefore publishes it as ``intraday_stability_score`` before it
is combined with trend, volume, and liquidity.
"""

from __future__ import annotations

from dataclasses import dataclass

NUMERIC_VARIANT = "NUMERIC"
MR_GUARDED15_VARIANT = "MR_GUARDED15"
HOTSECTOR_CHALLENGER_VARIANTS = (NUMERIC_VARIANT, MR_GUARDED15_VARIANT)

HOTSECTOR_TOP_K = 10
MR_GUARD_RANK = 15
MR_CUTOFF_RATIO = 0.95
HOTSECTOR_HORIZONS = (1, 3, 5)
HOTSECTOR_SINGLE_SIDE_COSTS_BPS = (10.0, 20.0, 50.0)

MR_SOURCE_COLUMNS = (
    "trend_score",
    "volume_score",
    "risk_score",
    "liquidity_score",
)
MR_PUBLISHED_COLUMNS = (
    "trend_score",
    "volume_score",
    "intraday_stability_score",
    "liquidity_score",
)
MR_EQUAL_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


@dataclass(frozen=True)
class HotsectorChallengerRankingContract:
    """Immutable, preregistered Numeric and guarded momentum/risk policy."""

    variants: tuple[str, ...] = HOTSECTOR_CHALLENGER_VARIANTS
    top_k: int = HOTSECTOR_TOP_K
    guard_rank: int = MR_GUARD_RANK
    cutoff_ratio: float = MR_CUTOFF_RATIO
    horizons: tuple[int, ...] = HOTSECTOR_HORIZONS
    single_side_costs_bps: tuple[float, ...] = HOTSECTOR_SINGLE_SIDE_COSTS_BPS
    mr_weights: tuple[float, ...] = MR_EQUAL_WEIGHTS

    def __post_init__(self) -> None:
        frozen = {
            "variants": HOTSECTOR_CHALLENGER_VARIANTS,
            "top_k": HOTSECTOR_TOP_K,
            "guard_rank": MR_GUARD_RANK,
            "cutoff_ratio": MR_CUTOFF_RATIO,
            "horizons": HOTSECTOR_HORIZONS,
            "single_side_costs_bps": HOTSECTOR_SINGLE_SIDE_COSTS_BPS,
            "mr_weights": MR_EQUAL_WEIGHTS,
        }
        actual = {
            "variants": tuple(self.variants),
            "top_k": self.top_k,
            "guard_rank": self.guard_rank,
            "cutoff_ratio": float(self.cutoff_ratio),
            "horizons": tuple(self.horizons),
            "single_side_costs_bps": tuple(float(value) for value in self.single_side_costs_bps),
            "mr_weights": tuple(float(value) for value in self.mr_weights),
        }
        changed = [name for name, expected in frozen.items() if actual[name] != expected]
        if changed:
            raise ValueError(f"hot-sector challenger contract is frozen: {', '.join(changed)}")

    def as_dict(self) -> dict[str, object]:
        """Return the exact ranking policy for immutable campaign receipts."""

        return {
            "variants": list(self.variants),
            "numeric_order": "candidate_relevance desc, candidate_score desc, symbol asc",
            "top_k": self.top_k,
            "mr_guard_rank": self.guard_rank,
            "mr_cutoff_ratio": self.cutoff_ratio,
            "mr_source_columns": list(MR_SOURCE_COLUMNS),
            "mr_published_columns": list(MR_PUBLISHED_COLUMNS),
            "mr_weights": list(self.mr_weights),
            "risk_score_semantics": "intraday_stability_score; higher_is_better",
            "horizons": list(self.horizons),
            "single_side_costs_bps": list(self.single_side_costs_bps),
        }


__all__ = [
    "HOTSECTOR_CHALLENGER_VARIANTS",
    "HOTSECTOR_HORIZONS",
    "HOTSECTOR_SINGLE_SIDE_COSTS_BPS",
    "HOTSECTOR_TOP_K",
    "MR_CUTOFF_RATIO",
    "MR_EQUAL_WEIGHTS",
    "MR_GUARDED15_VARIANT",
    "MR_GUARD_RANK",
    "MR_PUBLISHED_COLUMNS",
    "MR_SOURCE_COLUMNS",
    "NUMERIC_VARIANT",
    "HotsectorChallengerRankingContract",
]
