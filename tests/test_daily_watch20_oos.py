from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_research.daily_watch20_oos import score_rolling_oos


def test_rolling_oos_deduplicates_pure_one_day_label_passthrough() -> None:
    @dataclass(frozen=True)
    class Summary:
        observations: int = 1

    class Ranker:
        training_summary = Summary()

        def fit(self, _frame: pd.DataFrame, *, as_of_date: pd.Timestamp) -> None:
            assert as_of_date is not None

        def predict_relative(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame[["trade_date", "symbol"]].assign(relative_percentile=0.5)

    dates = pd.bdate_range("2026-01-05", periods=2)
    frame = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["000001.SZ", "000001.SZ"],
            "forward_return_1d": [0.01, 0.02],
            "forward_label_start_date": dates + pd.offsets.BDay(),
            "forward_label_end_date": dates + pd.offsets.BDay(2),
            "hard_eligible": True,
            "feature": 1.0,
        }
    )

    scored, _refits = score_rolling_oos(
        frame,
        ranker_factory=lambda _features: Ranker(),
        group_features=("feature",),
        label_col="forward_return_1d",
        return_col="forward_return_1d",
        evaluation_dates=dates,
        rolling_folds=2,
    )

    assert list(scored.columns).count("forward_return_1d") == 1
    assert list(scored.columns).count("forward_label_end_date") == 1
    assert isinstance(scored["forward_return_1d"], pd.Series)
