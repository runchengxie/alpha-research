from __future__ import annotations

import math

import pandas as pd
import pytest

from alpha_research.contextual import ContextTransformSpec, build_context_features


def _monthly(values: list[float], *, start="2024-01-31") -> pd.DataFrame:
    periods = pd.date_range(start, periods=len(values), freq="ME", tz="UTC")
    return pd.DataFrame(
        {
            "series_id": "activity.demo",
            "period_end": periods,
            "available_at": periods + pd.Timedelta(days=10),
            "source_retrieved_at": periods + pd.Timedelta(days=11),
            "value": values,
        }
    )


def test_context_transforms_are_period_aware_and_do_not_fill_missing_history():
    frame = _monthly([float(i) for i in range(1, 15)])
    specs = [
        ContextTransformSpec(
            series_id="activity.demo",
            transform="level",
            feature_name="ctx__demo_level",
        ),
        ContextTransformSpec(
            series_id="activity.demo",
            transform="change_1p",
            minimum_history=2,
            feature_name="ctx__demo_change1",
        ),
        ContextTransformSpec(
            series_id="activity.demo",
            transform="change_np",
            window=3,
            minimum_history=4,
            feature_name="ctx__demo_change3",
        ),
        ContextTransformSpec(
            series_id="activity.demo",
            transform="yoy",
            minimum_history=13,
            feature_name="ctx__demo_yoy",
        ),
        ContextTransformSpec(
            series_id="activity.demo",
            transform="acceleration",
            minimum_history=3,
            feature_name="ctx__demo_acceleration",
        ),
    ]

    result = build_context_features(frame, specs)
    assert result.iloc[0]["ctx__demo_level"] == pytest.approx(1.0)
    assert math.isnan(result.iloc[0]["ctx__demo_change1"])
    assert result.iloc[3]["ctx__demo_change3"] == pytest.approx(3.0)
    assert math.isnan(result.iloc[11]["ctx__demo_yoy"])
    assert result.iloc[12]["ctx__demo_yoy"] == pytest.approx(12.0)
    assert result.iloc[2]["ctx__demo_acceleration"] == pytest.approx(0.0)


def test_rolling_zscore_and_percentile_use_only_trailing_window():
    frame = _monthly([1.0, 2.0, 4.0, 8.0, 16.0])
    specs = [
        ContextTransformSpec(
            series_id="activity.demo",
            transform="rolling_zscore",
            window=3,
            minimum_history=3,
            feature_name="ctx__demo_z3",
        ),
        ContextTransformSpec(
            series_id="activity.demo",
            transform="rolling_percentile",
            window=3,
            minimum_history=3,
            feature_name="ctx__demo_pct3",
        ),
    ]
    result = build_context_features(frame, specs)

    assert result["ctx__demo_z3"].iloc[:2].isna().all()
    assert result.iloc[2]["ctx__demo_z3"] > 0
    assert result.iloc[2]["ctx__demo_pct3"] == pytest.approx(1.0)
    assert result.iloc[4]["ctx__demo_pct3"] == pytest.approx(1.0)


def test_yoy_matches_same_period_previous_year_not_row_12():
    frame = _monthly([100.0, 110.0, 150.0], start="2024-01-31")
    extra = pd.DataFrame(
        {
            "series_id": ["activity.demo"],
            "period_end": [pd.Timestamp("2025-01-31", tz="UTC")],
            "available_at": [pd.Timestamp("2025-02-10", tz="UTC")],
            "source_retrieved_at": [pd.Timestamp("2025-02-11", tz="UTC")],
            "value": [130.0],
        }
    )
    sparse = pd.concat([frame, extra], ignore_index=True)
    spec = ContextTransformSpec(
        series_id="activity.demo",
        transform="yoy",
        minimum_history=2,
        feature_name="ctx__demo_yoy",
    )
    result = build_context_features(sparse, [spec])
    jan_2025 = result.loc[result["period_end"] == pd.Timestamp("2025-01-31", tz="UTC")].iloc[0]
    assert jan_2025["ctx__demo_yoy"] == pytest.approx(30.0)


def test_transform_specs_reject_ambiguous_or_duplicate_configuration():
    frame = _monthly([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="window"):
        ContextTransformSpec(
            series_id="activity.demo",
            transform="change_np",
            feature_name="ctx__bad",
        )
    with pytest.raises(ValueError, match="duplicate"):
        build_context_features(
            frame,
            [
                ContextTransformSpec(
                    series_id="activity.demo",
                    transform="level",
                    feature_name="ctx__same",
                ),
                ContextTransformSpec(
                    series_id="activity.demo",
                    transform="level",
                    feature_name="ctx__same",
                ),
            ],
        )
