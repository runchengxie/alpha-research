"""Pure Hotsector Numeric v2 ranking kernel.

Campaign identity, frozen resource digests, and promotion eligibility stay in
strategy-app. This module owns only deterministic ranking mathematics and takes
all economic policy as an explicit immutable input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

_FEATURE_COLUMNS = (
    "candidate_relevance",
    "candidate_score",
    "daily_confirm_score",
    "intraday_stability_score",
    "liquidity_score",
    "trend_score",
    "ret_5d",
    "ret_10d",
    "amount_ratio_20d",
    "close_to_20d_high",
)
_OUTPUT_COLUMNS = (
    "variant",
    "trade_date",
    "symbol",
    "rank",
    "score",
    "candidate_pool_rank",
    "numeric_v2_rank",
    "candidate_relevance",
    "candidate_score",
    "daily_confirm_score",
    "intraday_stability_score",
    "liquidity_score",
    "trend_score",
    "ret_5d",
    "ret_10d",
    "amount_ratio_20d",
    "close_to_20d_high",
    "component_score",
    "ret_5d_penalty",
    "ret_10d_penalty",
    "amount_ratio_penalty",
    "near_high_x_short_heat_penalty",
    "overheat_penalty",
    "numeric_v2_score",
    "hysteresis_retained",
)


@dataclass(frozen=True, slots=True)
class HotsectorNumericV2RankingPolicy:
    """Explicit ranking policy supplied by the campaign owner."""

    numeric_variant: str
    numeric_v2_variant: str
    buffer_variant: str
    pool_variant: str
    candidate_pool_size: int
    top_k: int
    buffer_rank: int
    component_weights: tuple[tuple[str, float], ...]
    ret5_threshold: float
    ret5_full_excess: float
    ret5_weight: float
    ret10_threshold: float
    ret10_full_excess: float
    ret10_weight: float
    amount_threshold: float
    amount_full_excess: float
    amount_weight: float
    near_high_threshold: float
    near_high_full_excess: float
    short_heat_threshold: float
    short_heat_full_excess: float
    near_high_weight: float

    def __post_init__(self) -> None:
        if self.candidate_pool_size < self.buffer_rank or self.buffer_rank < self.top_k:
            raise ValueError("candidate_pool_size >= buffer_rank >= top_k is required")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        weights = dict(self.component_weights)
        expected = {
            "candidate_relevance",
            "daily_confirm_score",
            "intraday_stability_score",
            "liquidity_score",
            "trend_score",
        }
        if set(weights) != expected:
            raise ValueError("component_weights must define the five Numeric v2 components")
        if abs(sum(weights.values()) - 1.0) > 1e-12:
            raise ValueError("component_weights must sum to one")
        for name in (
            "ret5_full_excess",
            "ret10_full_excess",
            "amount_full_excess",
            "near_high_full_excess",
            "short_heat_full_excess",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")

    @property
    def variants(self) -> tuple[str, str, str, str]:
        return (
            self.numeric_variant,
            self.numeric_v2_variant,
            self.buffer_variant,
            self.pool_variant,
        )

    @property
    def weight_map(self) -> dict[str, float]:
        return dict(self.component_weights)


@dataclass(frozen=True)
class HotsectorNumericV2RankingResult:
    rankings: pd.DataFrame
    diagnostics: pd.DataFrame


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(frame[column], errors="coerce"))


def _validate(
    candidates: pd.DataFrame,
    policy: HotsectorNumericV2RankingPolicy,
) -> pd.DataFrame:
    required = {"trade_date", "symbol", *_FEATURE_COLUMNS}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Numeric v2 candidates are missing columns: {missing}")
    work = candidates.loc[:, sorted(required)].copy()
    work["trade_date"] = pd.to_datetime(
        work["trade_date"],
        errors="coerce",
    ).dt.normalize()
    work["symbol"] = work["symbol"].astype("string").str.strip().str.upper()
    if work[["trade_date", "symbol"]].isna().any().any() or work["symbol"].eq("").any():
        raise ValueError("Numeric v2 candidates contain invalid keys")
    if work.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("Numeric v2 candidates must be unique by date and symbol")
    for column in _FEATURE_COLUMNS:
        work[column] = _numeric(work, column)
    mandatory = work[["candidate_relevance", "candidate_score", "liquidity_score"]]
    if not np.isfinite(mandatory.to_numpy(dtype=float)).all():
        raise ValueError("Numeric v2 mandatory candidate fields must be finite")
    unit_fields = (
        "candidate_relevance",
        "daily_confirm_score",
        "intraday_stability_score",
        "liquidity_score",
        "trend_score",
        "close_to_20d_high",
    )
    for column in unit_fields:
        observed = work[column].dropna()
        if not observed.between(0.0, 1.0).all():
            raise ValueError(f"Numeric v2 {column} must be in [0, 1] when present")
    counts = work.groupby("trade_date", sort=True).size()
    if counts.empty or not counts.ge(policy.candidate_pool_size).all():
        raise ValueError("every Numeric v2 date requires a complete candidate pool")
    return work


def _clipped_excess(
    series: pd.Series,
    *,
    threshold: float,
    full_excess: float,
) -> pd.Series:
    values = (series - threshold) / full_excess
    return values.clip(lower=0.0, upper=1.0).fillna(0.0)


def score_numeric_v2_components(
    candidates: pd.DataFrame,
    policy: HotsectorNumericV2RankingPolicy,
) -> pd.DataFrame:
    """Apply one explicit Numeric v2 score policy to eligible candidates."""

    missing = sorted(set(_FEATURE_COLUMNS) - set(candidates.columns))
    if missing:
        raise ValueError(f"Numeric v2 score candidates are missing columns: {missing}")
    scored = candidates.copy()
    for column in _FEATURE_COLUMNS:
        scored[column] = _numeric(scored, column)
    mandatory = scored[["candidate_relevance", "candidate_score", "liquidity_score"]]
    if not np.isfinite(mandatory.to_numpy(dtype=float)).all():
        raise ValueError("Numeric v2 mandatory score fields must be finite")
    for column in (
        "candidate_relevance",
        "daily_confirm_score",
        "intraday_stability_score",
        "liquidity_score",
        "trend_score",
        "close_to_20d_high",
    ):
        observed = scored[column].dropna()
        if not observed.between(0.0, 1.0).all():
            raise ValueError(f"Numeric v2 {column} must be in [0, 1] when present")
    component = pd.Series(0.0, index=scored.index, dtype=float)
    for column, weight in policy.weight_map.items():
        component = component + scored[column].fillna(0.0) * weight
    scored["component_score"] = component

    ret5_excess = _clipped_excess(
        scored["ret_5d"],
        threshold=policy.ret5_threshold,
        full_excess=policy.ret5_full_excess,
    )
    ret10_excess = _clipped_excess(
        scored["ret_10d"],
        threshold=policy.ret10_threshold,
        full_excess=policy.ret10_full_excess,
    )
    amount_excess = _clipped_excess(
        scored["amount_ratio_20d"],
        threshold=policy.amount_threshold,
        full_excess=policy.amount_full_excess,
    )
    near_high_excess = _clipped_excess(
        scored["close_to_20d_high"],
        threshold=policy.near_high_threshold,
        full_excess=policy.near_high_full_excess,
    )
    short_heat_excess = _clipped_excess(
        scored["ret_5d"],
        threshold=policy.short_heat_threshold,
        full_excess=policy.short_heat_full_excess,
    )
    scored["ret_5d_penalty"] = ret5_excess * policy.ret5_weight
    scored["ret_10d_penalty"] = ret10_excess * policy.ret10_weight
    scored["amount_ratio_penalty"] = amount_excess * policy.amount_weight
    scored["near_high_x_short_heat_penalty"] = (
        near_high_excess * short_heat_excess * policy.near_high_weight
    )
    penalty_columns = (
        "ret_5d_penalty",
        "ret_10d_penalty",
        "amount_ratio_penalty",
        "near_high_x_short_heat_penalty",
    )
    scored["overheat_penalty"] = scored.loc[:, penalty_columns].sum(axis=1)
    scored["numeric_v2_score"] = scored["component_score"] - scored["overheat_penalty"]
    return scored


def _candidate_pool(
    group: pd.DataFrame,
    policy: HotsectorNumericV2RankingPolicy,
) -> pd.DataFrame:
    ordered = group.sort_values(
        ["candidate_relevance", "candidate_score", "symbol"],
        ascending=[False, False, True],
        kind="mergesort",
    ).head(policy.candidate_pool_size)
    pool = ordered.copy()
    pool["candidate_pool_rank"] = np.arange(1, len(pool) + 1, dtype=int)
    scored = score_numeric_v2_components(pool, policy)
    ranked = scored.sort_values(
        ["numeric_v2_score", "candidate_relevance", "candidate_score", "symbol"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).copy()
    ranked["numeric_v2_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def _publish(selected: pd.DataFrame, *, variant: str, order: str) -> pd.DataFrame:
    output = selected.sort_values(order, kind="mergesort").copy()
    output.insert(0, "variant", variant)
    output["rank"] = np.arange(1, len(output) + 1, dtype=int)
    output["score"] = (len(output) - output["rank"] + 1) / len(output)
    if "hysteresis_retained" not in output:
        output["hysteresis_retained"] = False
    return output.loc[:, _OUTPUT_COLUMNS]


def _buffer_selection(
    ranked: pd.DataFrame,
    previous: set[str] | None,
    policy: HotsectorNumericV2RankingPolicy,
) -> pd.DataFrame:
    retain = (
        set()
        if previous is None
        else set(
            ranked.loc[
                ranked["numeric_v2_rank"].le(policy.buffer_rank)
                & ranked["symbol"].isin(previous),
                "symbol",
            ].astype(str)
        )
    )
    chosen = list(
        ranked.loc[ranked["symbol"].isin(retain)]
        .sort_values("numeric_v2_rank", kind="mergesort")["symbol"]
        .astype(str)
    )
    symbols = (
        ranked.sort_values("numeric_v2_rank", kind="mergesort")["symbol"]
        .astype(str)
        .tolist()
    )
    for symbol in symbols:
        if len(chosen) >= policy.top_k:
            break
        if symbol not in chosen:
            chosen.append(symbol)
    selected = ranked.loc[ranked["symbol"].isin(chosen)].copy()
    selected["hysteresis_retained"] = selected["symbol"].isin(retain)
    return selected


def build_hotsector_numeric_v2_rankings(
    candidates: pd.DataFrame,
    policy: HotsectorNumericV2RankingPolicy,
) -> HotsectorNumericV2RankingResult:
    """Build original Numeric, fixed v2, buffer, and candidate-pool arms."""

    work = _validate(candidates, policy)
    outputs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    previous_buffer: set[str] | None = None
    for trade_date, group in work.groupby("trade_date", sort=True):
        ranked = _candidate_pool(group, policy)
        numeric = ranked.sort_values("candidate_pool_rank", kind="mergesort").head(
            policy.top_k
        )
        v2 = ranked.sort_values("numeric_v2_rank", kind="mergesort").head(policy.top_k)
        buffer = _buffer_selection(ranked, previous_buffer, policy)
        previous_overlap = (
            np.nan
            if previous_buffer is None
            else len(previous_buffer & set(buffer["symbol"].astype(str)))
        )
        previous_buffer = set(buffer["symbol"].astype(str))
        outputs.extend(
            (
                _publish(
                    numeric,
                    variant=policy.numeric_variant,
                    order="candidate_pool_rank",
                ),
                _publish(
                    v2,
                    variant=policy.numeric_v2_variant,
                    order="numeric_v2_rank",
                ),
                _publish(
                    buffer,
                    variant=policy.buffer_variant,
                    order="numeric_v2_rank",
                ),
                _publish(
                    ranked,
                    variant=policy.pool_variant,
                    order="candidate_pool_rank",
                ),
            )
        )
        diagnostics.append(
            {
                "trade_date": trade_date,
                "source_universe_size": len(group),
                "candidate_pool_size": len(ranked),
                "missing_daily_confirm_rows": int(
                    ranked["daily_confirm_score"].isna().sum()
                ),
                "numeric_v2_overheat_penalty_mean": float(
                    ranked["overheat_penalty"].mean()
                ),
                "numeric_v2_overheat_penalty_top10_mean": float(
                    v2["overheat_penalty"].mean()
                ),
                "numeric_vs_v2_top10_overlap": len(
                    set(numeric["symbol"].astype(str))
                    & set(v2["symbol"].astype(str))
                ),
                "buffer_retained_names": int(buffer["hysteresis_retained"].sum()),
                "buffer_overlap_with_previous": previous_overlap,
            }
        )
    rankings = pd.concat(outputs, ignore_index=True)
    variant_order = {
        variant: index for index, variant in enumerate(policy.variants)
    }
    rankings["_variant_order"] = rankings["variant"].map(variant_order)
    rankings = rankings.sort_values(
        ["trade_date", "_variant_order", "rank"],
        kind="mergesort",
    ).drop(columns="_variant_order")
    return HotsectorNumericV2RankingResult(
        rankings=rankings.reset_index(drop=True),
        diagnostics=pd.DataFrame(diagnostics).reset_index(drop=True),
    )


def visible_field_control_ranking(
    candidates: pd.DataFrame,
    policy: HotsectorNumericV2RankingPolicy,
) -> pd.DataFrame:
    """Rank a band without hidden relevance or raw candidate score."""

    visible_columns = {
        "symbol",
        "daily_confirm_score",
        "intraday_stability_score",
        "liquidity_score",
        "trend_score",
        "ret_5d",
        "ret_10d",
        "amount_ratio_20d",
        "close_to_20d_high",
    }
    missing = sorted(visible_columns - set(candidates.columns))
    if missing:
        raise ValueError(f"visible-field control candidates are missing columns: {missing}")
    work = candidates.loc[:, sorted(visible_columns)].copy()
    work["symbol"] = work["symbol"].astype("string").str.strip().str.upper()
    if work["symbol"].eq("").any() or work["symbol"].duplicated().any():
        raise ValueError("visible-field control symbols must be non-empty and unique")
    for column in visible_columns - {"symbol"}:
        work[column] = _numeric(work, column)
    work["candidate_relevance"] = 0.0
    work["candidate_score"] = 0.0
    scored = score_numeric_v2_components(work, policy)
    return scored.sort_values(
        ["numeric_v2_score", "symbol"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def bounded_visible_control_picks(
    boundary_candidates: pd.DataFrame,
    policy: HotsectorNumericV2RankingPolicy,
    *,
    selection_count: int = 3,
) -> tuple[str, ...]:
    """Choose a boundary subset with the visible-field negative control."""

    if selection_count <= 0 or selection_count > len(boundary_candidates):
        raise ValueError("selection_count must fit within the boundary candidate count")
    ranked = visible_field_control_ranking(boundary_candidates, policy)
    return tuple(ranked.head(selection_count)["symbol"].astype(str))


def risk_veto_visible_control_symbol(
    top10_candidates: pd.DataFrame,
    policy: HotsectorNumericV2RankingPolicy,
) -> str | None:
    """Veto the single largest positive overheat penalty, if any."""

    ranked = visible_field_control_ranking(top10_candidates, policy)
    hottest = ranked.sort_values(
        ["overheat_penalty", "symbol"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return str(hottest["symbol"]) if float(hottest["overheat_penalty"]) > 0.0 else None


__all__ = [
    "HotsectorNumericV2RankingPolicy",
    "HotsectorNumericV2RankingResult",
    "bounded_visible_control_picks",
    "build_hotsector_numeric_v2_rankings",
    "risk_veto_visible_control_symbol",
    "score_numeric_v2_components",
    "visible_field_control_ranking",
]
