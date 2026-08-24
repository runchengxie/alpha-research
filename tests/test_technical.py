from __future__ import annotations

import pandas as pd
import pytest

from alpha_research.technical import (
    average_true_range,
    rolling_high,
    true_range,
    volume_weighted_cost,
)


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "A", "B", "B"],
            "high": [10.0, 12.0, 13.0, 14.0, 20.0, 21.0],
            "low": [9.0, 10.0, 11.0, 12.0, 19.0, 19.5],
            "close": [9.5, 11.0, 12.0, 13.0, 19.5, 20.5],
            "volume": [10.0, 20.0, 10.0, 10.0, 5.0, 5.0],
            "amount": [100.0, 220.0, 130.0, 140.0, 100.0, 105.0],
        }
    )


def test_true_range_uses_previous_close_without_crossing_symbols() -> None:
    frame = _prices()
    result = true_range(frame, group_col="symbol")

    assert result.iloc[:4].tolist() == pytest.approx([1.0, 2.5, 2.0, 2.0])
    assert result.iloc[4:].tolist() == pytest.approx([1.0, 1.5])


def test_atr_and_prior_rolling_high_match_formula() -> None:
    frame = _prices()
    atr = average_true_range(frame, window=2, group_col="symbol")
    prior_high = rolling_high(frame, window=2, group_col="symbol")

    assert pd.isna(atr.iloc[0])
    assert atr.iloc[1:4].tolist() == pytest.approx([1.75, 2.25, 2.0])
    assert prior_high.iloc[:2].isna().all()
    assert prior_high.iloc[2:4].tolist() == pytest.approx([12.0, 13.0])
    assert prior_high.iloc[4:].isna().all()


def test_volume_weighted_cost_supports_amount_and_price_volume_forms() -> None:
    frame = _prices()
    stock_cost = volume_weighted_cost(
        frame,
        window=2,
        amount_col="amount",
        amount_divisor=100.0,
        group_col="symbol",
    )
    index_cost = volume_weighted_cost(frame, window=2, group_col="symbol")

    assert pd.isna(stock_cost.iloc[0])
    assert stock_cost.iloc[1:4].tolist() == pytest.approx([0.1066666667, 0.1166666667, 0.135])
    assert pd.isna(index_cost.iloc[0])
    assert index_cost.iloc[1:4].tolist() == pytest.approx([10.5, 11.3333333333, 12.5])
    assert pd.isna(index_cost.iloc[4])
    assert index_cost.iloc[5] == pytest.approx(20.0)


def test_technical_features_reject_invalid_windows_and_missing_fields() -> None:
    frame = _prices()
    with pytest.raises(ValueError, match="positive integer"):
        rolling_high(frame, window=0)
    with pytest.raises(ValueError, match="missing required columns"):
        true_range(frame.drop(columns=["close"]))
    with pytest.raises(ValueError, match="amount_divisor"):
        volume_weighted_cost(frame, window=2, amount_divisor=0)
