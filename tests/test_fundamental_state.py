from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from alpha_research.fundamental_state import (
    FundamentalScoreSpec,
    FundamentalTargetSpec,
    _learning_target,
    add_cashflow_yield,
    build_annual_fundamental_target_panel,
    build_fundamental_forecast_score,
    build_operating_quality_persistence_targets,
    build_periodic_fundamental_target_panel,
    build_persistence_baseline,
    evaluate_fundamental_forecast,
    purge_and_embargo_fundamental_rows,
    run_walk_forward_fundamental_forecast,
)


def test_purge_accepts_individually_valid_mixed_date_formats() -> None:
    frame = pd.DataFrame(
        {
            "feature_as_of_date": pd.to_datetime(["2019-01-01"]),
            "fundamental_label_end_date": pd.to_datetime(["2019-12-31"]),
        }
    )
    result = purge_and_embargo_fundamental_rows(
        frame, test_start="2020-01-01", test_end="2020-12-31 16:00:00"
    )
    assert len(result.frame) == 1


def test_purge_rejects_missing_boundary_with_interval_error():
    frame = pd.DataFrame(columns=pd.Index(["feature_as_of_date", "fundamental_label_end_date"]))
    with pytest.raises(ValueError, match="valid interval"):
        purge_and_embargo_fundamental_rows(frame, test_start="NaT", test_end="2020-01-01")


def test_build_operating_quality_persistence_targets_is_future_and_auditable() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "report_period": pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31"] * 2),
            "available_date": pd.to_datetime(["2022-03-01", "2023-03-01", "2024-03-01"] * 2),
            "roa": [0.10, 0.11, 0.12, 0.10, 0.08, 0.04],
            "gross_margin": [0.30, 0.31, 0.32, 0.30, 0.29, 0.20],
            "revenue_growth": [0.12, 0.11, 0.10, 0.12, 0.03, -0.05],
        }
    )
    result = build_operating_quality_persistence_targets(frame, horizon_years=1)
    row_a = result.loc[
        result["symbol"].eq("A") & result["report_period"].eq(pd.Timestamp("2021-12-31"))
    ].iloc[0]
    row_b = result.loc[
        result["symbol"].eq("B") & result["report_period"].eq(pd.Timestamp("2022-12-31"))
    ].iloc[0]
    assert row_a["future_roa_1y"] == 0.11
    assert bool(row_a["quality_persistent_1y"])
    assert not bool(row_b["quality_persistent_1y"])
    assert row_a["quality_label_end_date"] == pd.Timestamp("2023-03-01")
    assert result.attrs["audit"]["pit_policy"] == "future target availability is retained"


def test_quality_persistence_target_column_tracks_non_default_horizon() -> None:
    frame = _annual_frame()
    frame["revenue_growth"] = frame["revenue"].groupby(frame["symbol"]).pct_change()
    result = build_operating_quality_persistence_targets(frame, horizon_years=2)
    assert "future_roa_2y" in result.columns
    assert "future_gross_margin_2y" in result.columns
    assert "future_revenue_growth_2y" in result.columns
    assert "quality_persistent_2y" in result.columns


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
            "revenue_growth": [0.20, 0.25, 0.10, 0.10, 0.08, 0.10],
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
    mask = (frame["symbol"] == "A") & (frame["report_period"] == pd.Timestamp("2022-12-31"))
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


def test_build_periodic_target_panel_supports_quarter_horizons_and_pit_dates() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "report_period": pd.to_datetime(
                ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
            ),
            "available_date": pd.to_datetime(
                ["2024-04-30", "2024-08-01", "2024-10-30", "2025-04-01"]
            ),
            "roa": [0.10, 0.11, 0.12, 0.13],
        }
    )
    result = build_periodic_fundamental_target_panel(
        frame,
        (FundamentalTargetSpec("future_roa_1q", "roa", "level"),),
        horizon_periods=1,
        period_months=3,
    )
    row = result.frame.iloc[0]
    assert row["target_report_period"] == pd.Timestamp("2024-06-30")
    assert row["target_available_date"] == pd.Timestamp("2024-08-01")
    assert row["fundamental_label_end_date"] == pd.Timestamp("2024-08-01")
    assert row["future_roa_1q"] == pytest.approx(0.11)
    assert result.audit["horizon_periods"] == 1
    assert result.audit["period_months"] == 3


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


def test_cashflow_yield_requires_positive_same_date_market_cap() -> None:
    frame = pd.DataFrame(
        {
            "n_cashflow_act": [20.0, 10.0, 5.0],
            "total_mv": [100.0, 0.0, None],
        }
    )
    result = add_cashflow_yield(frame)
    assert result["cashflow_yield"].iloc[0] == pytest.approx(0.20)
    assert result["cashflow_yield"].iloc[1:].isna().all()


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


def test_learning_target_supports_pointwise_ranker_and_listwise_relevance() -> None:
    frame = pd.DataFrame(
        {
            "formation": ["2024-01-01"] * 4 + ["2024-02-01"] * 4,
            "target": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    pointwise = _learning_target(
        frame,
        "target",
        date_col="formation",
        model_type="xgb_regressor",
        model_params={"objective": "reg:squarederror"},
        target_transform="cross_sectional_rank",
    )
    listwise = _learning_target(
        frame,
        "target",
        date_col="formation",
        model_type="xgb_ranker",
        model_params={"objective": "rank:ndcg"},
        target_transform="auto",
    )
    assert pointwise.iloc[:4].tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert listwise.iloc[:4].tolist() == [7, 15, 23, 31]


def test_multiple_targets_can_share_the_same_source_column() -> None:
    specs = (
        FundamentalTargetSpec("future_roa_1y", "roa", "level"),
        FundamentalTargetSpec("delta_roa_1y", "roa", "delta"),
    )

    result = build_annual_fundamental_target_panel(_annual_frame(), specs)
    mask = (result.frame["symbol"] == "A") & (
        result.frame["report_period"] == pd.Timestamp("2022-12-31")
    )
    row = result.frame.loc[mask].iloc[0]

    assert row["future_roa_1y"] == pytest.approx(0.12)
    assert row["delta_roa_1y"] == pytest.approx(0.02)


def test_walk_forward_runner_compares_persistence_and_ridge_without_future_labels() -> None:
    rows = []
    periods = pd.to_datetime(["2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31"])
    for period_index, period in enumerate(periods):
        feature_date = pd.Timestamp(year=period.year + 1, month=3, day=31)
        for symbol_index, symbol in enumerate(["A", "B", "C"]):
            feature = float(period_index * 3 + symbol_index + 1)
            rows.append(
                {
                    "symbol": symbol,
                    "report_period": period,
                    "feature_as_of_date": feature_date,
                    "fundamental_label_end_date": (
                        feature_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)
                    ),
                    "roa": feature / 100.0,
                    "feature_x": feature,
                    "delta_roa_1y": 0.02 * feature + 0.01,
                }
            )
    frame = pd.DataFrame(rows)
    # This prior-period row has a label that is not known at the 2023-03-31 test cutoff.
    poisoned = frame.index[
        (frame["report_period"] == pd.Timestamp("2021-12-31")) & (frame["symbol"] == "C")
    ][0]
    frame.loc[poisoned, "fundamental_label_end_date"] = pd.Timestamp("2023-06-30")
    frame.loc[poisoned, "delta_roa_1y"] = 999.0

    result = run_walk_forward_fundamental_forecast(
        frame,
        target_spec=FundamentalTargetSpec("delta_roa_1y", "roa", "delta"),
        feature_cols=("feature_x",),
        model_configs={
            "ridge": {"type": "ridge", "params": {"alpha": 1e-9}},
        },
        min_train_rows=5,
        min_train_periods=2,
    )

    test_rows = result.frame[result.frame["report_period"] == pd.Timestamp("2022-12-31")]
    assert len(test_rows) == 3
    assert test_rows["pred_ridge"].notna().all()
    assert test_rows["pred_persistence"].eq(0.0).all()
    assert np.allclose(test_rows["pred_ridge"], test_rows["delta_roa_1y"], atol=1e-5)
    folds = result.audit["folds"]
    assert isinstance(folds, list) and folds
    assert isinstance(folds[-1], dict)
    final_fold = cast(dict[str, Any], folds[-1])
    assert final_fold["training_label_end_max"] < final_fold["test_cutoff"]
    assert final_fold["training_rows"] == 8


def test_walk_forward_runner_supports_pairwise_and_listwise_fundamental_targets() -> None:
    rows = []
    periods = pd.to_datetime(["2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31"])
    for period_index, period in enumerate(periods):
        feature_date = pd.Timestamp(year=period.year + 1, month=3, day=31)
        for symbol_index, symbol in enumerate(["A", "B", "C"]):
            feature = float(period_index * 3 + symbol_index + 1)
            rows.append(
                {
                    "symbol": symbol,
                    "report_period": period,
                    "feature_as_of_date": feature_date,
                    "fundamental_label_end_date": (
                        feature_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)
                    ),
                    "roa": feature / 100.0,
                    "feature_x": feature,
                    "delta_roa_1y": 0.02 * feature + 0.01,
                }
            )
    result = run_walk_forward_fundamental_forecast(
        pd.DataFrame(rows),
        target_spec=FundamentalTargetSpec("delta_roa_1y", "roa", "delta"),
        feature_cols=("feature_x",),
        model_configs={
            "pairwise": {
                "type": "xgb_ranker",
                "params": {
                    "n_estimators": 3,
                    "max_depth": 1,
                    "learning_rate": 0.1,
                    "objective": "rank:pairwise",
                    "random_state": 42,
                },
            },
            "listwise": {
                "type": "xgb_ranker",
                "params": {
                    "n_estimators": 3,
                    "max_depth": 1,
                    "learning_rate": 0.1,
                    "objective": "rank:ndcg",
                    "random_state": 42,
                },
            },
        },
        min_train_rows=5,
        min_train_periods=2,
    )
    assert result.frame["pred_pairwise"].notna().any()
    assert result.frame["pred_listwise"].notna().any()
    folds = result.audit["folds"]
    assert isinstance(folds, list) and folds
    assert isinstance(folds[-1], dict)
    final_fold = cast(dict[str, Any], folds[-1])
    assert final_fold["target_transforms"] == {
        "pairwise": "cross_sectional_rank",
        "listwise": "cross_sectional_rank",
    }


def test_walk_forward_runner_skips_rows_with_unavailable_future_labels() -> None:
    frame = _annual_frame()
    targets = build_annual_fundamental_target_panel(
        frame,
        (FundamentalTargetSpec("future_roa_1y", "roa", "level"),),
    )

    result = run_walk_forward_fundamental_forecast(
        targets.frame,
        target_spec=FundamentalTargetSpec("future_roa_1y", "roa", "level"),
        feature_cols=("roa",),
        model_configs={},
        min_train_rows=1,
        min_train_periods=1,
    )

    assert result.frame["target_available_date"].notna().all()
    assert result.audit["prediction_rows"] == len(result.frame)
    assert result.audit["prediction_rows"] == 0


def test_walk_forward_runner_rejects_future_label_columns_as_features() -> None:
    with pytest.raises(ValueError, match="future label"):
        run_walk_forward_fundamental_forecast(
            _annual_frame().assign(
                feature_as_of_date=pd.to_datetime(["2023-03-20", "2024-03-20", "2025-03-20"] * 2),
                fundamental_label_end_date=pd.to_datetime(
                    ["2024-03-20", "2025-03-20", "2026-03-20"] * 2
                ),
                target_available_date=pd.to_datetime(["2024-03-20", "2025-03-20", None] * 2),
                future_roa_1y=[0.12, 0.11, None, 0.04, 0.06, None],
            ),
            target_spec=FundamentalTargetSpec("future_roa_1y", "roa", "level"),
            feature_cols=("target_available_date",),
            model_configs={},
        )
