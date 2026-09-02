from __future__ import annotations

import numpy as np
import pytest

from alpha_research.signal_drift import summarize_signal_drift


def test_identical_populations_have_zero_drift() -> None:
    values = np.linspace(-2.0, 2.0, 200)

    report = summarize_signal_drift(values, values.copy(), bins=10)

    assert report.psi == pytest.approx(0.0, abs=1e-12)
    assert report.ks_statistic == pytest.approx(0.0, abs=1e-12)
    assert report.mean_shift_std == pytest.approx(0.0, abs=1e-12)
    assert report.std_ratio == pytest.approx(1.0, rel=1e-12)


def test_shifted_live_population_is_detected() -> None:
    rng = np.random.default_rng(7)
    reference = rng.normal(0.0, 1.0, 2000)
    current = rng.normal(0.8, 1.2, 2000)

    report = summarize_signal_drift(reference, current, bins=10)

    assert report.psi > 0.1
    assert report.ks_statistic > 0.2
    assert report.mean_shift_std > 0.5
    assert report.std_ratio > 1.0


def test_nonfinite_values_are_removed_but_minimum_sample_is_enforced() -> None:
    report = summarize_signal_drift(
        [0.0, 1.0, 2.0, float("nan"), float("inf")],
        [0.1, 1.1, 2.1, float("nan")],
        bins=3,
        min_observations=3,
    )
    assert report.reference_count == 3
    assert report.current_count == 3

    with pytest.raises(ValueError, match="min_observations"):
        summarize_signal_drift([0.0, 1.0], [0.0, 1.0], min_observations=3)


def test_constant_reference_reports_mean_shift_as_unavailable_when_current_moves() -> None:
    report = summarize_signal_drift([1.0] * 20, [2.0] * 20, min_observations=10)

    assert report.mean_shift_std is None
    assert report.reference_constant is True
    assert report.ks_statistic == pytest.approx(1.0)
