"""Deterministic Numeric and guarded momentum/risk hot-sector rankings.

Owner note: this module moved from strategy-app to alpha-research as part of
SA-15 (strategy-knowledge repatriation to the compute kernel layer). It depends
only on the co-located ``hotsector_challenger_contract`` and pandas/numpy, with
no cross-repo imports, so it is a pure compute-kernel module. strategy-app now
re-exports it from this location to preserve frozen provenance paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from .hotsector_challenger_contract import (
    MR_GUARDED15_VARIANT,
    MR_PUBLISHED_COLUMNS,
    NUMERIC_VARIANT,
    HotsectorChallengerRankingContract,
)

_BASE_COLUMNS = ("trade_date", "symbol", "candidate_relevance", "candidate_score")
_OUTPUT_COLUMNS = (
    "variant",
    "trade_date",
    "symbol",
    "rank",
    "score",
    "numeric_rank",
    "numeric_score",
    "candidate_score",
    "mr_composite_score",
    "trend_score",
    "volume_score",
    "intraday_stability_score",
    "liquidity_score",
    "mr_guard_eligible",
    "mr_cutoff_score",
)


@dataclass(frozen=True)
class HotsectorChallengerRankingResult:
    """Selected Top10 rows plus date-level fail-closed MR diagnostics."""

    scores: pd.DataFrame
    diagnostics: pd.DataFrame


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(frame[column], errors="coerce"))


def _validate_and_normalize(candidates: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    required = {*_BASE_COLUMNS, *MR_PUBLISHED_COLUMNS}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"hot-sector candidates are missing columns: {missing}")
    work = candidates.loc[:, sorted(required)].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    work["symbol"] = work["symbol"].astype("string").str.strip().str.upper()
    if work[["trade_date", "symbol"]].isna().any().any() or work["symbol"].eq("").any():
        raise ValueError("hot-sector candidates contain invalid trade_date or symbol")
    if work.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("hot-sector candidates must be unique by trade_date and symbol")
    for column in ("candidate_relevance", "candidate_score", *MR_PUBLISHED_COLUMNS):
        work[column] = _numeric(work, column)
    base = work[["candidate_relevance", "candidate_score"]].to_numpy(dtype=float)
    if not np.isfinite(base).all():
        raise ValueError("hot-sector numeric ranking scores must be finite")
    if not work["candidate_relevance"].between(0.0, 1.0).all():
        raise ValueError("candidate_relevance must be in [0, 1]")
    counts = work.groupby("trade_date", sort=True).size()
    if counts.empty or not counts.ge(top_k).all():
        raise ValueError("every hot-sector date requires at least Top10 candidates")
    return work


def _numeric_ranked(group: pd.DataFrame) -> pd.DataFrame:
    ranked = group.sort_values(
        ["candidate_relevance", "candidate_score", "symbol"],
        ascending=[False, False, True],
        kind="mergesort",
    ).copy()
    ranked["numeric_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["numeric_score"] = ranked["candidate_relevance"]
    return ranked


def _published_rows(
    selected: pd.DataFrame,
    *,
    variant: str,
    cutoff_score: float,
) -> pd.DataFrame:
    output = selected.copy()
    output.insert(0, "variant", variant)
    output["rank"] = np.arange(1, len(output) + 1, dtype=int)
    output["score"] = (len(output) - output["rank"] + 1) / len(output)
    output["mr_cutoff_score"] = cutoff_score
    for column in ("mr_composite_score", *MR_PUBLISHED_COLUMNS):
        if column not in output.columns:
            output[column] = np.nan
    if "mr_guard_eligible" not in output.columns:
        output["mr_guard_eligible"] = False
    return output.loc[:, _OUTPUT_COLUMNS]


def _mr_guarded_rows(
    numeric: pd.DataFrame,
    contract: HotsectorChallengerRankingContract,
    cutoff_score: float,
) -> tuple[pd.DataFrame | None, str | None, int]:
    guard = numeric.loc[numeric["numeric_rank"].le(contract.guard_rank)].copy()
    if len(guard) < contract.guard_rank:
        return None, "fewer_than_guard_rank_candidates", 0
    threshold = cutoff_score * contract.cutoff_ratio
    guard["mr_guard_eligible"] = guard["numeric_score"].ge(threshold)
    eligible = cast(pd.DataFrame, guard.loc[guard["mr_guard_eligible"]]).copy()
    if len(eligible) < contract.top_k:
        return None, "fewer_than_top_k_cutoff_eligible", len(eligible)
    feature_values = eligible.loc[:, MR_PUBLISHED_COLUMNS].to_numpy(dtype=float)
    valid = np.isfinite(feature_values).all() and bool(
        eligible.loc[:, MR_PUBLISHED_COLUMNS].apply(lambda col: col.between(0.0, 1.0)).all().all()
    )
    if not valid:
        return None, "missing_or_invalid_mr_features_in_guard", len(eligible)
    eligible["mr_composite_score"] = np.average(
        eligible.loc[:, MR_PUBLISHED_COLUMNS].to_numpy(dtype=float),
        axis=1,
        weights=np.asarray(contract.mr_weights, dtype=float),
    )
    ranked = eligible.sort_values(
        ["mr_composite_score", "numeric_score", "candidate_score", "symbol"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return ranked.head(contract.top_k), None, len(ranked)


def _rank_one_date(
    trade_date: pd.Timestamp,
    group: pd.DataFrame,
    contract: HotsectorChallengerRankingContract,
) -> tuple[list[pd.DataFrame], dict[str, object]]:
    numeric = _numeric_ranked(group)
    cutoff_score = float(numeric.iloc[contract.top_k - 1]["numeric_score"])
    numeric_top = numeric.head(contract.top_k).copy()
    numeric_top["mr_guard_eligible"] = False
    outputs = [_published_rows(numeric_top, variant=NUMERIC_VARIANT, cutoff_score=cutoff_score)]
    mr_rows, reason, eligible_rows = _mr_guarded_rows(numeric, contract, cutoff_score)
    if mr_rows is not None:
        outputs.append(
            _published_rows(mr_rows, variant=MR_GUARDED15_VARIANT, cutoff_score=cutoff_score)
        )
    diagnostics = {
        "trade_date": trade_date,
        "universe_size": len(numeric),
        "numeric_cutoff_score": cutoff_score,
        "mr_cutoff_threshold": cutoff_score * contract.cutoff_ratio,
        "mr_guard_rows": min(len(numeric), contract.guard_rank),
        "mr_eligible_rows": eligible_rows,
        "mr_available": mr_rows is not None,
        "mr_exclusion_reason": reason,
    }
    return outputs, diagnostics


def build_hotsector_challenger_rankings(
    candidates: pd.DataFrame,
    contract: HotsectorChallengerRankingContract | None = None,
) -> HotsectorChallengerRankingResult:
    """Build paired Top10 rankings while retaining Numeric on MR-invalid dates."""

    policy = contract or HotsectorChallengerRankingContract()
    work = _validate_and_normalize(candidates, top_k=policy.top_k)
    score_parts: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    for raw_date, group in work.groupby("trade_date", sort=True):
        trade_date = cast(pd.Timestamp, raw_date)
        outputs, diagnostics = _rank_one_date(trade_date, group, policy)
        score_parts.extend(outputs)
        diagnostic_rows.append(diagnostics)
    scores = pd.concat(score_parts, ignore_index=True)
    variant_order = {variant: index for index, variant in enumerate(policy.variants)}
    scores["_variant_order"] = scores["variant"].map(variant_order)
    scores = scores.sort_values(
        ["trade_date", "_variant_order", "rank"],
        kind="mergesort",
    ).drop(columns="_variant_order")
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values("trade_date", kind="mergesort")
    return HotsectorChallengerRankingResult(
        scores=scores.reset_index(drop=True),
        diagnostics=diagnostics.reset_index(drop=True),
    )


__all__ = [
    "HotsectorChallengerRankingResult",
    "build_hotsector_challenger_rankings",
]
