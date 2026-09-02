"""Cross-sectional stable-compounder profiles for fundamental research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StableCompounderSpec:
    """Describe the observable characteristics of a stable compounder."""

    quality_cols: tuple[str, ...] = ()
    growth_cols: tuple[str, ...] = ()
    stability_cols: tuple[str, ...] = ()
    cashflow_cols: tuple[str, ...] = ()
    risk_cols: tuple[str, ...] = ()
    valuation_cols: tuple[str, ...] = ()
    loose_threshold: float = 0.60
    strict_threshold: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 <= self.loose_threshold <= 1.0:
            raise ValueError("loose_threshold must be between 0 and 1")
        if not 0.0 <= self.strict_threshold <= 1.0:
            raise ValueError("strict_threshold must be between 0 and 1")
        if self.strict_threshold < self.loose_threshold:
            raise ValueError("strict_threshold must be at least loose_threshold")
        if not self.component_columns:
            raise ValueError("stable compounder requires at least one component")

    @property
    def component_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                self.quality_cols
                + self.growth_cols
                + self.stability_cols
                + self.cashflow_cols
                + self.risk_cols
                + self.valuation_cols
            )
        )


@dataclass(frozen=True)
class StableCompounderResult:
    frame: pd.DataFrame
    audit: dict[str, object]


_GROUPS = (
    "quality",
    "growth",
    "stability",
    "cashflow",
    "risk",
    "valuation",
)


def _percentile(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    group_cols: list[str],
    higher: bool,
) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    rank = values.groupby([frame[col] for col in group_cols], sort=False, dropna=False).rank(
        pct=True, method="average"
    )
    score = rank.mean(axis=1, skipna=True)
    if not higher:
        score = 1.0 - score
    return score


def build_stable_compounder_label(
    frame: pd.DataFrame,
    spec: StableCompounderSpec,
    *,
    date_col: str = "signal_date",
    symbol_col: str = "symbol",
    industry_col: str | None = None,
    score_col: str = "stable_compounder_score",
) -> StableCompounderResult:
    """Score current, observable compounder characteristics cross-sectionally.

    Each characteristic is converted to a percentile within each signal date and,
    when supplied, industry. Higher percentiles are preferred for quality, growth,
    and cash flow; lower raw values are preferred for stability, risk, and valuation.
    The result is a research profile, not a future-outcome label.
    """

    required = {date_col, symbol_col, *spec.component_columns}
    if industry_col:
        required.add(industry_col)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"stable compounder frame missing columns: {missing}")
    if frame.empty:
        raise ValueError("stable compounder frame must not be empty")

    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    if out[date_col].isna().any():
        raise ValueError(f"stable compounder requires valid dates in {date_col}")
    if out[symbol_col].isna().any():
        raise ValueError("stable compounder requires non-null symbols")

    group_cols = [date_col] + ([industry_col] if industry_col else [])
    component_groups = {
        "quality": (spec.quality_cols, True),
        "growth": (spec.growth_cols, True),
        "stability": (spec.stability_cols, False),
        "cashflow": (spec.cashflow_cols, True),
        "risk": (spec.risk_cols, False),
        "valuation": (spec.valuation_cols, False),
    }
    group_scores: list[pd.Series] = []
    for name in _GROUPS:
        columns, higher = component_groups[name]
        result = _percentile(out, columns, group_cols=group_cols, higher=higher)
        out[f"{score_col}_{name}"] = result
        group_scores.append(result)

    out["stable_compounder_coverage"] = out[list(spec.component_columns)].notna().mean(axis=1)
    out[score_col] = pd.concat(group_scores, axis=1).mean(axis=1, skipna=True)
    out["stable_compounder_loose"] = out[score_col] >= spec.loose_threshold
    out["stable_compounder_strict"] = out[score_col] >= spec.strict_threshold

    audit: dict[str, object] = {
        "schema_version": "stable_compounder.v1",
        "rows": len(out),
        "component_columns": list(spec.component_columns),
        "ranking_scope": ",".join(group_cols),
        "complete_component_rows": int(out[list(spec.component_columns)].notna().all(axis=1).sum()),
        "loose_rows": int(out["stable_compounder_loose"].sum()),
        "strict_rows": int(out["stable_compounder_strict"].sum()),
        "interpretation": "current observable profile; not a future outcome label",
    }
    return StableCompounderResult(out, audit)
