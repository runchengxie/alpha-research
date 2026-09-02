from __future__ import annotations

import pandas as pd

from alpha_research.cross_sectional_strategy_arms import (
    StrategyArmSpec,
    build_candidate_filtered_signal,
    build_fused_signal,
    build_strategy_arm_scores,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": ["2026-01-01"] * 3,
            "symbol": ["A", "B", "C"],
            "fundamental_score": [0.9, 0.4, 0.1],
            "daily_watch20_score": [0.2, 0.8, 0.4],
            "fundamental_candidate": [True, False, False],
        }
    )


def test_build_four_arm_scores_and_fusion() -> None:
    scored = build_strategy_arm_scores(
        _frame(),
        (
            StrategyArmSpec("pure_fundamental", "fundamental_score"),
            StrategyArmSpec("daily_watch20", "daily_watch20_score"),
        ),
    )
    assert {"pure_fundamental", "daily_watch20"} <= set(scored.columns)

    filtered = build_candidate_filtered_signal(
        scored,
        candidate_col="fundamental_candidate",
        short_signal_col="daily_watch20",
    )
    assert filtered.loc[filtered["symbol"].eq("A"), "daily_watch20_filtered"].item() == 0.2
    assert pd.isna(filtered.loc[filtered["symbol"].eq("B"), "daily_watch20_filtered"].item())

    fused = build_fused_signal(
        scored,
        score_columns={"fundamental_score": 0.5, "daily_watch20_score": 0.5},
    )
    assert fused.loc[fused["symbol"].eq("A"), "fused_score"].item() == 0.55
