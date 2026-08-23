"""Pure Hotsector DeepSeek V4 ranking-trial analysis.

Campaign resources, model/provider execution, artifact identity, execution accounting and
promotion policy stay with their owners. This module only analyzes frozen ranking records
under an explicit policy supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class DeepSeekV4RankingPolicy:
    """Explicit policy for ranking stability and score materialization."""

    models: tuple[str, ...]
    sample_dates: tuple[str, ...]
    screen_dates: tuple[str, ...]
    screen_arms: tuple[str, ...]
    canonical_arm: str
    shuffle_arm: str
    opaque_arm: str
    top_k: int
    numeric_variant: str
    model_variants: tuple[tuple[str, str], ...]
    relative_percentile_column: str
    max_invalid_model_responses_per_model: int
    max_invalid_ranking_contracts_per_model: int
    max_invalid_pair_dates_per_model: int
    shuffle_minimum_pair_overlap: int
    shuffle_minimum_mean_top_k_overlap: float
    opaque_minimum_pair_overlap: int
    opaque_minimum_mean_top_k_overlap: float
    publication_contract_required: bool

    def __post_init__(self) -> None:
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be non-empty and unique")
        if not self.sample_dates or not self.screen_dates:
            raise ValueError("sample_dates and screen_dates must be non-empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if set(dict(self.model_variants)) != set(self.models):
            raise ValueError("model_variants must map every model exactly once")
        if {self.canonical_arm, self.shuffle_arm, self.opaque_arm} - set(self.screen_arms):
            raise ValueError("screen_arms must contain canonical, shuffle and opaque arms")
        for value in (
            self.max_invalid_model_responses_per_model,
            self.max_invalid_ranking_contracts_per_model,
            self.max_invalid_pair_dates_per_model,
        ):
            if value < 0:
                raise ValueError("invalid-count budgets must be non-negative")

    @property
    def variant_map(self) -> dict[str, str]:
        return dict(self.model_variants)


def _text_list(value: object, *, field: str, expected_size: int | None = None) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field} must be a non-empty string array")
    result = cast(list[str], value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicate symbols")
    if expected_size is not None and len(result) != expected_size:
        raise ValueError(f"{field} must contain exactly {expected_size} symbols")
    return result


def _trial_index(
    trials: Sequence[Mapping[str, object]],
    *,
    dates: Sequence[str],
    models: Sequence[str],
    arms: Sequence[str],
) -> dict[tuple[str, str, str], Mapping[str, object]]:
    expected = {(date, model, arm) for date in dates for model in models for arm in arms}
    indexed: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for trial in trials:
        key = (str(trial.get("date")), str(trial.get("model")), str(trial.get("arm")))
        if key in indexed:
            raise ValueError(f"duplicate ranking trial: {key}")
        indexed[key] = trial
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)
    if missing or extra:
        raise ValueError(f"ranking trial grid changed: missing={missing}, extra={extra}")
    return indexed


def _ranking_valid(trial: Mapping[str, object], model: str, *, top_k: int) -> bool:
    order = trial.get("ranking_order")
    return (
        trial.get("actual_model") == model
        and trial.get("ranking_contract_valid") is True
        and isinstance(order, list)
        and len(order) == top_k
        and len(set(cast(list[object], order))) == top_k
    )


def _pair_metrics(
    indexed: Mapping[tuple[str, str, str], Mapping[str, object]],
    *,
    policy: DeepSeekV4RankingPolicy,
    model: str,
    comparison_arm: str,
    minimum_overlap: int,
) -> dict[str, Any]:
    overlaps: list[int] = []
    for date in policy.screen_dates:
        canonical = indexed[(date, model, policy.canonical_arm)]
        comparison = indexed[(date, model, comparison_arm)]
        if not (
            _ranking_valid(canonical, model, top_k=policy.top_k)
            and _ranking_valid(comparison, model, top_k=policy.top_k)
        ):
            continue
        left = _text_list(
            canonical.get("ranking_order"),
            field="canonical ranking_order",
            expected_size=policy.top_k,
        )
        right = _text_list(
            comparison.get("ranking_order"),
            field="comparison ranking_order",
            expected_size=policy.top_k,
        )
        overlaps.append(len(set(left) & set(right)))
    return {
        "valid_pairs": len(overlaps),
        "overlaps": overlaps,
        "mean_overlap": float(np.mean(overlaps)) if overlaps else np.nan,
        "dates_meeting_minimum_overlap": sum(value >= minimum_overlap for value in overlaps),
    }


def _model_phase1_metrics(
    indexed: Mapping[tuple[str, str, str], Mapping[str, object]],
    *,
    policy: DeepSeekV4RankingPolicy,
    model: str,
) -> dict[str, Any]:
    rows = [
        indexed[(date, model, arm)]
        for date in policy.screen_dates
        for arm in policy.screen_arms
    ]
    shuffle = _pair_metrics(
        indexed,
        policy=policy,
        model=model,
        comparison_arm=policy.shuffle_arm,
        minimum_overlap=policy.shuffle_minimum_pair_overlap,
    )
    opaque = _pair_metrics(
        indexed,
        policy=policy,
        model=model,
        comparison_arm=policy.opaque_arm,
        minimum_overlap=policy.opaque_minimum_pair_overlap,
    )
    return {
        "model": model,
        "trials": len(rows),
        "actual_model_matches": sum(row.get("actual_model") == model for row in rows),
        "ranking_contract_valid": sum(row.get("ranking_contract_valid") is True for row in rows),
        "publication_contract_valid": sum(
            row.get("publication_contract_valid") is True for row in rows
        ),
        "shuffle_valid_pairs": shuffle["valid_pairs"],
        "shuffle_overlaps": shuffle["overlaps"],
        "shuffle_mean_overlap": shuffle["mean_overlap"],
        "shuffle_dates_meeting_minimum": shuffle["dates_meeting_minimum_overlap"],
        "opaque_valid_pairs": opaque["valid_pairs"],
        "opaque_overlaps": opaque["overlaps"],
        "opaque_mean_overlap": opaque["mean_overlap"],
        "opaque_dates_meeting_minimum": opaque["dates_meeting_minimum_overlap"],
    }


def analyze_deepseek_v4_phase1(
    trials: Sequence[Mapping[str, object]],
    policy: DeepSeekV4RankingPolicy,
) -> dict[str, Any]:
    """Apply ranking and presentation-stability gates to a complete trial grid."""

    indexed = _trial_index(
        trials,
        dates=policy.screen_dates,
        models=policy.models,
        arms=policy.screen_arms,
    )
    model_metrics = {
        model: _model_phase1_metrics(indexed, policy=policy, model=model)
        for model in policy.models
    }
    trial_count = len(policy.screen_dates) * len(policy.screen_arms)
    minimum_model_matches = trial_count - policy.max_invalid_model_responses_per_model
    minimum_ranking = trial_count - policy.max_invalid_ranking_contracts_per_model
    minimum_pairs = len(policy.screen_dates) - policy.max_invalid_pair_dates_per_model
    gates: dict[str, bool] = {}
    for model, metrics in model_metrics.items():
        prefix = model
        gates[f"{prefix}:model_identity"] = metrics["actual_model_matches"] >= minimum_model_matches
        gates[f"{prefix}:ranking_contract"] = metrics["ranking_contract_valid"] >= minimum_ranking
        gates[f"{prefix}:shuffle_pairs"] = metrics["shuffle_valid_pairs"] >= minimum_pairs
        gates[f"{prefix}:shuffle_mean_overlap"] = bool(
            metrics["shuffle_mean_overlap"] >= policy.shuffle_minimum_mean_top_k_overlap
        )
        gates[f"{prefix}:shuffle_pair_overlap"] = (
            metrics["shuffle_dates_meeting_minimum"] >= minimum_pairs
        )
        gates[f"{prefix}:opaque_pairs"] = metrics["opaque_valid_pairs"] >= minimum_pairs
        gates[f"{prefix}:opaque_mean_overlap"] = bool(
            metrics["opaque_mean_overlap"] >= policy.opaque_minimum_mean_top_k_overlap
        )
        gates[f"{prefix}:opaque_pair_overlap"] = (
            metrics["opaque_dates_meeting_minimum"] >= minimum_pairs
        )
        if policy.publication_contract_required:
            gates[f"{prefix}:publication_contract"] = (
                metrics["publication_contract_valid"] >= minimum_ranking
            )
    return {
        "model_metrics": model_metrics,
        "gates": gates,
        "proceed_to_execution": all(gates.values()),
    }


def build_intention_to_deploy_scores(
    canonical_trials: Sequence[Mapping[str, object]],
    numeric_rankings: Mapping[str, Sequence[str]],
    policy: DeepSeekV4RankingPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build baseline/model score rows with a fail-closed baseline fallback."""

    indexed = _trial_index(
        canonical_trials,
        dates=policy.sample_dates,
        models=policy.models,
        arms=(policy.canonical_arm,),
    )
    score_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    for date in policy.sample_dates:
        numeric = _text_list(
            list(numeric_rankings.get(date, ())),
            field=f"numeric rankings {date}",
            expected_size=policy.top_k,
        )
        variant_orders: dict[str, list[str]] = {policy.numeric_variant: numeric}
        for model in policy.models:
            trial = indexed[(date, model, policy.canonical_arm)]
            valid = _ranking_valid(trial, model, top_k=policy.top_k)
            selected = (
                _text_list(
                    trial.get("ranking_order"),
                    field="ranking_order",
                    expected_size=policy.top_k,
                )
                if valid
                else numeric
            )
            variant = policy.variant_map[model]
            variant_orders[variant] = selected
            contract_rows.append(
                {
                    "date": date,
                    "model": model,
                    "variant": variant,
                    "actual_model_matches": trial.get("actual_model") == model,
                    "ranking_contract_valid": trial.get("ranking_contract_valid") is True,
                    "publication_contract_valid": trial.get("publication_contract_valid") is True,
                    "fallback_used": not valid,
                    "fallback_reason": None if valid else "invalid_canonical_ranking_contract",
                }
            )
        trade_date = pd.Timestamp(date)
        for variant, symbols in variant_orders.items():
            for rank, symbol in enumerate(symbols, start=1):
                score_rows.append(
                    {
                        "trade_date": trade_date,
                        "available_at": trade_date.tz_localize("Asia/Shanghai")
                        + pd.Timedelta(hours=15, minutes=1),
                        "variant": variant,
                        "symbol": symbol,
                        "rank": rank,
                        policy.relative_percentile_column: float(policy.top_k - rank + 1)
                        / policy.top_k,
                    }
                )
    return pd.DataFrame(score_rows), pd.DataFrame(contract_rows)


__all__ = [
    "DeepSeekV4RankingPolicy",
    "analyze_deepseek_v4_phase1",
    "build_intention_to_deploy_scores",
]
