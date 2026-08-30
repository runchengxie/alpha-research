from __future__ import annotations

import pandas as pd
import pytest

from alpha_research.freshness_overlay import apply_freshness_overlay


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


def test_freshness_overlay_preserves_exact_rank_blend_and_metadata() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-05"] * 3),
            "signal": [3.0, 2.0, 1.0],
            "volume_sma5_ratio": [1.0, 3.0, 2.0],
            "volume_sma20_ratio": [1.0, 3.0, 2.0],
            "volume_sma60_ratio": [1.0, 3.0, 2.0],
        }
    )

    overlaid, meta = apply_freshness_overlay(
        frame,
        score_col="signal",
        cfg={"enabled": True, "name": "volume-only", "lambda": 0.05},
    )

    assert overlaid["signal_base"].tolist() == [3.0, 2.0, 1.0]
    assert overlaid["signal"].tolist() == pytest.approx(
        [
            0.95 * 1.0 + 0.05 * (1 / 3),
            0.95 * (2 / 3) + 0.05 * 1.0,
            0.95 * (1 / 3) + 0.05 * (2 / 3),
        ]
    )
    assert meta == {
        "enabled": True,
        "name": "volume-only",
        "lambda": 0.05,
        "base_score_col": "signal",
        "output_col": "signal",
        "volume_rank_cols": [
            "volume_sma5_ratio",
            "volume_sma20_ratio",
            "volume_sma60_ratio",
        ],
        "rows": 3,
        "dates": 1,
    }


def test_freshness_overlay_disabled_is_noop() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-05"]),
            "signal": [1.0],
        }
    )

    overlaid, meta = apply_freshness_overlay(frame, score_col="signal", cfg=None)

    pd.testing.assert_frame_equal(overlaid, frame)
    assert meta == {"enabled": False}


def test_freshness_overlay_empty_frame_reports_empty() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.Series(dtype="datetime64[ns]"),
            "signal": pd.Series(dtype=float),
        }
    )

    overlaid, meta = apply_freshness_overlay(
        frame,
        score_col="signal",
        cfg={"enabled": True},
    )

    assert overlaid.empty
    assert meta == {"enabled": True, "status": "empty"}


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_freshness_overlay_rejects_lambda_outside_unit_interval(value: float) -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-05"]),
            "signal": [1.0],
            "volume_sma5_ratio": [1.0],
            "volume_sma20_ratio": [1.0],
            "volume_sma60_ratio": [1.0],
        }
    )

    with pytest.raises(ValueError, match="lambda must be between 0 and 1"):
        apply_freshness_overlay(
            frame,
            score_col="signal",
            cfg={"enabled": True, "lambda": value},
        )


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


def test_freshness_overlay_is_exported_from_alpha_package() -> None:
    from alpha_research import apply_freshness_overlay as public_apply

    assert public_apply is apply_freshness_overlay
