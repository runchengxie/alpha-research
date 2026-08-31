from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_research.style_factors import compute_size_style_signal


def _frames(periods: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2025-01-02", periods=periods)
    large = pd.DataFrame(
        {
            "close": np.linspace(100.0, 112.0, periods),
            "amount": np.linspace(1_000.0, 1_100.0, periods),
        },
        index=index,
    )
    small = pd.DataFrame(
        {
            "close": np.linspace(100.0, 165.0, periods),
            "amount": np.linspace(800.0, 1_400.0, periods),
        },
        index=index,
    )
    return large, small


def test_size_style_signal_is_bounded_and_prefers_small_when_relative_strength_falls() -> None:
    large, small = _frames()
    cutoff = large.index[-5]

    result = compute_size_style_signal(large, small, as_of=cutoff)

    assert result.as_of == cutoff.normalize()
    assert result.small_series.index.max() == cutoff
    assert result.large_series.index.max() == cutoff
    assert result.relative_strength.index.max() == cutoff
    assert result.signal == "small_cap"
    assert result.crowding_zone in {"high_crowding", "low_crowding"}
    assert (result.short_window, result.long_window) in {(5, 20), (20, 60)}


def test_expected_through_rejects_stale_data() -> None:
    large, small = _frames()
    expected = large.index[-1] + pd.offsets.BDay(1)

    with pytest.raises(ValueError, match="size-style data is stale"):
        compute_size_style_signal(large, small, expected_through=expected)


def test_owner_kernel_rejects_missing_required_columns() -> None:
    large, small = _frames()

    with pytest.raises(ValueError, match="missing columns"):
        compute_size_style_signal(large.drop(columns=["amount"]), small)
