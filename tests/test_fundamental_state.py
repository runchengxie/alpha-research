from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alpha_research.fundamental_state import (
    FundamentalScoreSpec,
    FundamentalTargetSpec,
    build_annual_fundamental_target_panel,
    build_fundamental_forecast_score,
    build_persistence_baseline,
    evaluate_fundamental_forecast,
    purge_and_embargo_fundamental_rows,
)


def _annual_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "report_period": [
                "2022-12-31",
                "2023-12-31",
                "2024-12-31",
                "2022-12-31",
                "2023-12-31",
                "2024-12-31",
            ],
            "available_date": [
                "2023-03-20",
                "2024-03-20",
                "2025-03-20",
                "2023-04-01",
                "2024-04-01",
                "2025-04-01",
            ],
            "roa": [0.10, 0.12, 0.11, 0.05, 0.04, 0.06],
            "revenue": [100.0, 120.0, 150.0, 80.0, 88.0, 96.8],
            "gross_margin": [0.30, 0.33, 0.32, 0.20, 0.19, 0.21],
        }
    )


def test_build_annual_target_panel_tracks_label_availability_and_transforms() -> None:
    specs = (
        FundamentalTargetSpec("delta_roa_1y", "roa", "delta"),
        FundamentalTargetSpec("revenue_growth_1y", "revenue", "pct_change"),
        FundamentalTargetSpec("future_gross_margin_1y", "gross_margin", "level"),
    )

    result = build_annual_fundamental_target_panel(_annual_frame(), specs)
    frame = result.frame
    mask = (frame["symbol"] == "A") & (
        frame["report_period"] == pd.Timestamp("2022-12-31")
    )
    row = frame.loc[mask].iloc[0]

    assert row["feature_as_of_date"] == pd.Timestamp("2023-03-20")
    assert row["target_report_period"] == pd.Timestamp("2023-12-31")
    assert row["target_available_date"] == pd.Timestamp("2024-03-20")
    assert row["fundamental_label_end_date"] == pd.Timestamp("2024-03-20")
    assert row["delta_roa_1y"] == pytest.approx(0.02)
    assert row["revenue_growth_1y"] == pytest.approx(0.20)
    assert row["future_gross_margin_1y"] == pytest.approx(0.33)
    assert result.audit["complete_label_rows"] == 4
    assert result.audit["rows"] == 6


def test_build_annual_target_panel_rejects_duplicate_report_periods() -> None:
    frame = pd.concat([_annual_frame(), _annual_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        build_annual_fundamental_target_panel(
            frame,
            (FundamentalTargetSpec("delta_roa_1y", "roa", "delta"),),
        )


def test_pct_change_requires_positive_finite_base() -> None:
    frame = _annual_frame()
    frame.loc[
        (frame["symbol"] == "B") & (frame["report_period"] == "2022-12-31"),
        "revenue",
    ] = 0.0

    result = build_annual_fundamental_target_panel(
        frame,
        (FundamentalTargetSpec("revenue_growth_1y", "revenue", "pct_change"),),
    )
    row = result.frame[
        (result.frame["symbol"] == "B")
        & (result.frame["report_period"] == pd.Timestamp("2022-12-31"))
    ].iloc[0]

    assert math.isnan(float(row["revenue_growth_1y"]))


def test_persistence_baseline_matches_target_semantics() -> None:
    frame = pd.DataFrame({"roa": [0.10, 0.20]})

    assert build_persistence_baseline(
        frame, FundamentalTargetSpec("future_roa", "roa", "level")
    ).tolist() == [0.10, 0.20]
    assert build_persistence_baseline(
        frame, FundamentalTargetSpec("delta_roa", "roa", "delta")
    ).tolist() == [0.0, 0.0]
    assert build_persistence_baseline(
        frame, FundamentalTargetSpec("growth_roa", "roa", "pct_change")
    ).tolist() == [0.0, 0.0]


def test_forecast_metrics_are_cross_sectionally_interpretable() -> None:
    frame = pd.DataFrame(
        {
            "actual": [-0.2, -0.1, 0.1, 0.3],
            "pred": [-0.1, -0.05, 0.2, 0.25],
        }
    )

    metrics = evaluate_fundamental_forecast(frame, "actual", "pred", directional=True)

    assert metrics["count"] == 4
    assert metrics["mae"] == pytest.approx(0.075)
    expected_rmse = np.sqrt((0.1**2 + 0.05**2 + 0.1**2 + 0.05**2) / 4)
    assert metrics["rmse"] == pytest.approx(expected_rmse)
    assert metrics["rank_ic"] == pytest.approx(1.0)
    assert metrics["direction_accuracy"] == pytest.approx(1.0)


def test_forecast_score_combines_quality_growth_and_valuation_by_date() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": ["2026-01-01"] * 3,
            "symbol": ["A", "B", "C"],
            "pred_quality": [0.9, 0.5, 0.1],
            "pred_growth": [0.7, 0.6, 0.2],
            "earnings_yield": [0.03, 0.08, 0.05],
        }
    )
    specs = (
        FundamentalScoreSpec("pred_quality", weight=2.0),
        FundamentalScoreSpec("pred_growth", weight=1.0),
        FundamentalScoreSpec("earnings_yield", weight=1.0),
    )

    scored = build_fundamental_forecast_score(frame, specs)
    ranked = scored.sort_values("fundamental_rank")

    assert ranked.iloc[0]["symbol"] == "A"
    assert ranked["fundamental_rank"].tolist() == [1.0, 2.0, 3.0]
    assert scored["fundamental_score"].between(0.0, 1.0).all()


def test_purge_and_embargo_removes_label_overlap_and_post_test_buffer() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["old", "overlap", "test", "embargo", "safe_after"],
            "feature_as_of_date": [
                "2018-03-01",
                "2019-03-01",
                "2020-06-01",
                "2021-01-10",
                "2021-02-15",
            ],
            "fundamental_label_end_date": [
                "2019-03-20",
                "2020-03-20",
                "2021-03-20",
                "2022-03-20",
                "2022-04-20",
            ],
        }
    )

    result = purge_and_embargo_fundamental_rows(
        frame,
        test_start="2020-01-01",
        test_end="2020-12-31",
        embargo_days=31,
    )

    assert result.frame["symbol"].tolist() == ["old", "safe_after"]
    assert result.audit["purged_overlap_rows"] == 2
    assert result.audit["embargoed_rows"] == 1
    assert result.audit["kept_rows"] == 2


def test_forecast_rank_ic_can_average_within_cross_sections() -> None:
    frame = pd.DataFrame(
        {
            "report_period": ["2023-12-31"] * 3 + ["2024-12-31"] * 3,
            "actual": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "pred": [1.0, 2.0, 3.0, 30.0, 20.0, 10.0],
        }
    )

    metrics = evaluate_fundamental_forecast(
        frame,
        "actual",
        "pred",
        date_col="report_period",
    )

    assert metrics["rank_ic"] == pytest.approx(0.0)
