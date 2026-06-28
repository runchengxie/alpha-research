from __future__ import annotations

import pandas as pd
import pytest

from cstree.alpha.freshness_overlay import apply_freshness_overlay


def test_volume_only_freshness_overlay_blends_base_and_volume_ranks() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-05"] * 3),
            "signal": [0.2, 0.8, 0.5],
            "volume_sma5_ratio": [3.0, 1.0, 2.0],
        }
    )

    overlaid, meta = apply_freshness_overlay(
        frame,
        score_col="signal",
        cfg={
            "enabled": True,
            "columns": ["volume_sma5_ratio"],
            "lambda": 0.25,
        },
    )

    assert meta["enabled"] is True
    assert "signal_base" in overlaid.columns
    assert "signal_freshness_volume_rank" in overlaid.columns
    assert overlaid["signal"].between(0.0, 1.0).all()


def test_freshness_overlay_fails_when_enabled_columns_are_missing() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-05"]),
            "signal": [0.2],
        }
    )

    with pytest.raises(ValueError, match="missing volume columns"):
        apply_freshness_overlay(
            frame,
            score_col="signal",
            cfg={"enabled": True, "columns": ["volume_sma5_ratio"]},
        )
