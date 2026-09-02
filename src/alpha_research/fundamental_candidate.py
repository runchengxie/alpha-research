"""Candidate-pool and low-turnover buffer utilities for fundamental research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .fundamental_state import FundamentalScoreSpec, build_fundamental_forecast_score


@dataclass(frozen=True)
class FundamentalCandidateSpec:
    score_specs: tuple[FundamentalScoreSpec, ...]
    min_coverage: float = 1.0
    top_quantile: float = 0.10
    buffer_quantile: float = 0.25

    def __post_init__(self) -> None:
        if not self.score_specs:
            raise ValueError("fundamental candidate requires score specs")
        if not 0.0 < self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0, 1]")
        if not 0.0 < self.top_quantile <= 1.0:
            raise ValueError("top_quantile must be in (0, 1]")
        if not self.top_quantile <= self.buffer_quantile <= 1.0:
            raise ValueError("buffer_quantile must be >= top_quantile and <= 1")


def build_fundamental_candidate_score(
    frame: pd.DataFrame,
    spec: FundamentalCandidateSpec,
    *,
    date_col: str = "signal_date",
    score_col: str = "fundamental_score",
) -> pd.DataFrame:
    """Build a percentile score and an entry-eligible fundamental candidate flag."""

    scored = build_fundamental_forecast_score(
        frame,
        spec.score_specs,
        date_col=date_col,
        score_col=score_col,
    )
    value_columns = [item.column for item in spec.score_specs]
    scored["fundamental_coverage"] = scored[value_columns].notna().mean(axis=1)
    scored["fundamental_candidate"] = False
    for _, group in scored.groupby(date_col, sort=False, dropna=False):
        eligible = group["fundamental_coverage"] >= spec.min_coverage
        cutoff = group.loc[eligible, score_col].quantile(1.0 - spec.top_quantile)
        scored.loc[group.index, "fundamental_candidate"] = eligible & (group[score_col] >= cutoff)
    return scored


def apply_candidate_buffer(
    scored: pd.DataFrame,
    *,
    previous_holdings: pd.DataFrame | None = None,
    date_col: str = "signal_date",
    symbol_col: str = "symbol",
    score_col: str = "fundamental_score",
    top_quantile: float = 0.10,
    buffer_quantile: float = 0.25,
) -> pd.DataFrame:
    """Keep incumbents inside a wider percentile buffer to reduce turnover."""

    if not 0.0 < top_quantile <= buffer_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 < top_quantile <= buffer_quantile <= 1")
    required = {date_col, symbol_col, score_col}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"candidate buffer missing columns: {missing}")
    out = scored.copy()
    held = set() if previous_holdings is None else set(previous_holdings[symbol_col].astype(str))
    out["requalified"] = False
    out["candidate_entry"] = False
    out["candidate_selected"] = False
    for _, group in out.groupby(date_col, sort=False, dropna=False):
        ranked = group[score_col].rank(method="first", ascending=False, pct=True)
        entry = ranked <= top_quantile
        buffer = ranked <= buffer_quantile
        incumbent = group[symbol_col].astype(str).isin(held)
        out.loc[group.index, "candidate_entry"] = entry
        out.loc[group.index, "requalified"] = incumbent & buffer & ~entry
        out.loc[group.index, "candidate_selected"] = entry | (incumbent & buffer)
    return out
