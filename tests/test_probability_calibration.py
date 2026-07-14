from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_research.probability_calibration import (
    expanding_probability_calibration,
    probability_to_bet_size,
)


def test_probability_bet_size_is_monotonic_and_discrete() -> None:
    probabilities = pd.Series([0.5, 0.6, 0.8])
    sizes = probability_to_bet_size(probabilities, step_size=0.1)
    assert sizes.tolist() == sorted(sizes.tolist())
    assert np.allclose((sizes / 0.1).round(), sizes / 0.1)
    assert sizes.iloc[0] == 0.0


def test_expanding_calibration_never_uses_same_date_outcomes() -> None:
    dates = pd.date_range("2024-01-01", periods=12)
    frame = pd.DataFrame(
        {
            "trade_date": np.repeat(dates, 20),
            "score": np.tile(np.linspace(-2.0, 2.0, 20), len(dates)),
        }
    )
    frame["outcome"] = (frame["score"] + np.sin(np.arange(len(frame))) * 0.2 > 0).astype(int)
    result, summary = expanding_probability_calibration(
        frame,
        score_col="score",
        outcome_col="outcome",
        min_train_observations=40,
    )
    assert result.loc[result["trade_date"] <= dates[1], "calibrated_probability"].isna().all()
    assert result.loc[result["trade_date"] >= dates[2], "calibrated_probability"].notna().any()
    assert summary.calibration_windows > 0
