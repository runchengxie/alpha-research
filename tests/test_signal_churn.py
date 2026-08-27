from __future__ import annotations

import pandas as pd

from alpha_research.signal_churn import estimate_topk_membership_churn


def _scored() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2026-01-02"] * 4 + ["2026-01-05"] * 4 + ["2026-01-06"] * 4
            ),
            "symbol": ["A", "B", "C", "D"] * 3,
            "score": [4.0, 3.0, 2.0, 1.0, 4.0, 2.0, 3.0, 1.0, 1.0, 4.0, 3.0, 2.0],
        }
    )


def test_topk_membership_churn_measures_signal_set_change_only() -> None:
    churn = estimate_topk_membership_churn(
        _scored(),
        "score",
        2,
        list(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])),
    )

    assert churn.name == "topk_membership_churn"
    assert churn.index.tolist() == list(pd.to_datetime(["2026-01-05", "2026-01-06"]))
    assert churn.tolist() == [0.5, 0.5]


def test_topk_membership_churn_has_no_portfolio_buffer_or_weight_arguments() -> None:
    import inspect

    parameters = inspect.signature(estimate_topk_membership_churn).parameters
    assert "buffer_exit" not in parameters
    assert "buffer_entry" not in parameters
    assert "weighting" not in parameters
    assert "transaction_cost_bps" not in parameters
