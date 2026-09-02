"""Composable research-only score builders for cross-sectional strategy arms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyArmSpec:
    name: str
    score_col: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.score_col.strip():
            raise ValueError("strategy arm name and score_col must be non-empty")


def build_strategy_arm_scores(
    frame: pd.DataFrame,
    arms: tuple[StrategyArmSpec, ...],
    *,
    date_col: str = "signal_date",
) -> pd.DataFrame:
    """Copy named score columns into stable, arm-named columns."""

    if not arms:
        raise ValueError("strategy arms must be non-empty")
    missing = sorted({date_col, *(arm.score_col for arm in arms)} - set(frame.columns))
    if missing:
        raise ValueError(f"strategy arm frame missing columns: {missing}")
    out = frame.copy()
    for arm in arms:
        out[arm.name] = pd.to_numeric(out[arm.score_col], errors="coerce")
    return out


def build_candidate_filtered_signal(
    frame: pd.DataFrame,
    *,
    candidate_col: str,
    short_signal_col: str,
    output_col: str = "daily_watch20_filtered",
) -> pd.DataFrame:
    """Keep a short-cycle score only for the fundamental candidate universe."""

    missing = sorted({candidate_col, short_signal_col} - set(frame.columns))
    if missing:
        raise ValueError(f"candidate-filtered signal missing columns: {missing}")
    out = frame.copy()
    out[output_col] = out[short_signal_col].where(out[candidate_col].fillna(False).astype(bool))
    return out


def build_fused_signal(
    frame: pd.DataFrame,
    *,
    score_columns: Mapping[str, float],
    output_col: str = "fused_score",
) -> pd.DataFrame:
    """Combine already normalized scores with explicit weights and coverage."""

    if not score_columns:
        raise ValueError("fused signal score_columns must be non-empty")
    missing = sorted(set(score_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"fused signal frame missing columns: {missing}")
    weights = pd.Series(score_columns, dtype=float)
    if (weights <= 0).any() or not weights.index.is_unique:
        raise ValueError("fused signal weights must be positive and unique")
    out = frame.copy()
    values = out[list(score_columns)].apply(pd.to_numeric, errors="coerce")
    observed = values.notna()
    weighted_observed = observed.mul(weights, axis="columns").sum(axis=1)
    out[output_col] = values.mul(weights, axis="columns").sum(axis=1) / weighted_observed.replace(
        0.0, pd.NA
    )
    out[f"{output_col}_coverage"] = weighted_observed / weights.sum()
    return out
