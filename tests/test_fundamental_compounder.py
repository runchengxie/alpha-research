from __future__ import annotations

import pandas as pd
import pytest

from alpha_research.fundamental_compounder import (
    StableCompounderSpec,
    build_stable_compounder_label,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": ["2026-01-01"] * 4,
            "symbol": ["A", "B", "C", "D"],
            "roa": [0.20, 0.12, 0.08, 0.03],
            "growth": [0.18, 0.10, 0.05, -0.02],
            "margin_vol": [0.02, 0.04, 0.08, 0.15],
            "cash_conversion": [1.2, 1.0, 0.8, 0.4],
            "debt_to_assets": [0.20, 0.35, 0.55, 0.80],
            "pe": [18.0, 25.0, 40.0, 80.0],
        }
    )


def test_stable_compounder_score_rewards_quality_growth_stability_and_value() -> None:
    result = build_stable_compounder_label(
        _frame(),
        StableCompounderSpec(
            quality_cols=("roa",),
            growth_cols=("growth",),
            stability_cols=("margin_vol",),
            cashflow_cols=("cash_conversion",),
            risk_cols=("debt_to_assets",),
            valuation_cols=("pe",),
            loose_threshold=0.8,
            strict_threshold=0.9,
        ),
    )

    scored = result.frame.sort_values("stable_compounder_score", ascending=False)
    assert scored.iloc[0]["symbol"] == "A"
    assert scored.iloc[-1]["symbol"] == "D"
    assert scored["stable_compounder_score"].between(0.0, 1.0).all()
    assert scored["stable_compounder_loose"].tolist() == [True, False, False, False]
    assert result.audit["rows"] == 4


def test_stable_compounder_can_rank_within_industry() -> None:
    frame = _frame().assign(industry=["tech", "tech", "bank", "bank"])
    result = build_stable_compounder_label(
        frame,
        StableCompounderSpec(
            quality_cols=("roa",),
            growth_cols=("growth",),
            stability_cols=("margin_vol",),
            cashflow_cols=("cash_conversion",),
            risk_cols=("debt_to_assets",),
            valuation_cols=("pe",),
        ),
        industry_col="industry",
    )

    assert result.frame["stable_compounder_score"].notna().all()
    assert result.audit["ranking_scope"] == "signal_date,industry"


def test_stable_compounder_rejects_missing_columns_and_empty_groups() -> None:
    spec = StableCompounderSpec(quality_cols=("missing",), growth_cols=("growth",))

    with pytest.raises(ValueError, match="missing columns"):
        build_stable_compounder_label(_frame(), spec)

    with pytest.raises(ValueError, match="at least one component"):
        build_stable_compounder_label(_frame(), StableCompounderSpec())
