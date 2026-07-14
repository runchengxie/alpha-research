from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_research.event_labeling import TripleBarrierConfig, label_triple_barrier
from alpha_research.fracdiff import (
    fixed_width_fractional_difference,
    fractional_difference_weights,
)
from alpha_research.sample_weighting import (
    SampleWeightConfig,
    average_uniqueness,
    build_event_sample_weights,
    build_indicator_matrix,
    sequential_bootstrap,
)
from alpha_research.split import build_sample_weight
from alpha_research.structural_breaks import sadf_series, symmetric_cusum_filter


def test_triple_barrier_records_exact_event_windows() -> None:
    dates = pd.date_range("2024-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "symbol": ["A"] * 6,
            "trade_date": dates,
            "close": [100.0, 101.0, 103.0, 102.0, 99.0, 100.0],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["up", "down", "vertical"],
            "symbol": ["A", "A", "A"],
            "signal_date": [dates[0], dates[2], dates[4]],
            "target": [0.02, 0.02, 0.05],
            "side": [1.0, 1.0, 1.0],
        }
    )

    result = label_triple_barrier(
        prices,
        events,
        config=TripleBarrierConfig(vertical_horizon=2),
    ).set_index("event_id")

    assert result.loc["up", "barrier"] == "profit_taking"
    assert result.loc["up", "label"] == 1
    assert result.loc["down", "barrier"] == "stop_loss"
    assert result.loc["down", "label"] == -1
    assert result.loc["vertical", "barrier"] == "vertical"
    assert result.loc["vertical", "label_end"] == dates[5]
    assert {"label_start", "label_end", "meta_label"}.issubset(result.columns)


def test_uniqueness_weights_and_sequential_bootstrap_are_deterministic() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "label_start": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
            "label_end": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
        }
    )
    bars = pd.date_range("2024-01-01", "2024-01-05")
    indicator = build_indicator_matrix(events, bar_index=bars)
    uniqueness = average_uniqueness(indicator)

    assert np.isclose(uniqueness["e1"], 2.0 / 3.0)
    assert np.isclose(uniqueness["e2"], 0.5)
    assert np.isclose(uniqueness["e3"], 0.75)
    assert sequential_bootstrap(indicator, random_state=42) == ["e3", "e2", "e3"]

    weighted, receipt = build_event_sample_weights(
        events,
        bar_index=bars,
        config=SampleWeightConfig(
            mode="uniqueness_time_decay",
            time_decay_halflife=2,
        ),
    )
    assert np.isclose(weighted["sample_weight"].mean(), 1.0)
    assert receipt.effective_sample_size < len(events)
    assert receipt.events_sha256


def test_panel_uniqueness_does_not_mix_independent_symbols() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "symbol": ["A", "B"],
            "label_start": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "label_end": pd.to_datetime(["2024-01-03", "2024-01-03"]),
        }
    )
    weighted, receipt = build_event_sample_weights(
        events,
        bar_index=pd.date_range("2024-01-01", "2024-01-03"),
        config=SampleWeightConfig(mode="uniqueness"),
    )

    assert weighted["average_uniqueness"].tolist() == [1.0, 1.0]
    assert receipt.group_count == 2


def test_uniqueness_mode_is_wired_into_training_weight_builder() -> None:
    dates = pd.date_range("2024-01-01", periods=5)
    frame = pd.DataFrame(
        {
            "trade_date": [dates[0], dates[1], dates[0], dates[1]],
            "symbol": ["A", "A", "B", "B"],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    weights = build_sample_weight(
        frame,
        "uniqueness_time_decay",
        params={
            "all_trade_dates": dates,
            "label_horizon_days": 2,
            "label_shift_days": 0,
            "halflife": 2,
            "equalize_dates": False,
        },
    )

    assert weights is not None
    assert len(weights) == len(frame)
    assert np.isclose(np.mean(weights), 1.0)
    assert np.all(np.isfinite(weights))


def test_fixed_width_fractional_difference_has_stable_warmup() -> None:
    series = pd.Series(np.arange(100, dtype=float))
    weights = fractional_difference_weights(0.5, threshold=1e-3)
    transformed = fixed_width_fractional_difference(series, 0.5, threshold=1e-3)

    assert weights[-1] == 1.0
    assert transformed.notna().sum() == len(series) - len(weights) + 1
    assert transformed.iloc[: len(weights) - 1].isna().all()


def test_structural_break_diagnostics_emit_point_in_time_series() -> None:
    index = pd.date_range("2024-01-01", periods=80)
    increments = pd.Series([0.0] * 35 + [0.5] * 10 + [0.0] * 35, index=index)
    events = symmetric_cusum_filter(increments, threshold=1.0)
    assert len(events) >= 3

    explosive = pd.Series(np.exp(np.linspace(0.0, 2.0, 80)), index=index)
    sadf = sadf_series(explosive, min_window=30, lags=0, step=5)
    assert sadf.index.equals(explosive.index)
    assert sadf.notna().sum() > 0
