from __future__ import annotations

import pandas as pd
import pytest

from alpha_research.signal_stability import compute_signal_stability_diagnostics


def test_compute_signal_stability_diagnostics_explains_churn_and_buffers() -> None:
    scored = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                ]
            ),
            "symbol": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "signal_backtest": [4.0, 3.0, 2.0, 1.0, 1.0, 4.0, 3.0, 2.0],
            "momentum_20d": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5],
        }
    )
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101", "20200101", "20200108", "20200108"],
            "entry_date": ["20200102", "20200102", "20200109", "20200109"],
            "symbol": ["A", "B", "B", "C"],
            "weight": [0.5, 0.5, 0.5, 0.5],
        }
    )

    result = compute_signal_stability_diagnostics(
        positions,
        scored,
        feature_columns=["momentum_20d"],
        buffer_width=1,
    )

    assert result.summary["status"] == "ok"
    assert result.summary["windows"] == 1
    row = result.by_window.iloc[0]
    assert row["entrant_count"] == 1
    assert row["exit_count"] == 1
    assert row["entrant_prev_rank_mean"] == pytest.approx(3.0)
    assert row["exit_curr_rank_mean"] == pytest.approx(4.0)
    assert row["entrant_from_buffer_count"] == 1
    assert result.by_symbol["symbol"].tolist() == ["A", "C"]
    assert result.by_feature.iloc[0]["feature"] == "momentum_20d"
