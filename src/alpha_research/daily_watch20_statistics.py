"""Small-sample inference helpers for DailyWatch20 rolling OOS evidence."""

from __future__ import annotations

from math import erfc, floor, sqrt
from typing import Any

import numpy as np


def newey_west_mean_inference(
    values: object,
    *,
    max_lag: int | None = None,
    minimum_lag: int = 0,
) -> dict[str, Any]:
    """Estimate a mean and two-sided normal p-value with a HAC standard error."""

    sample = np.asarray(values, dtype=float).reshape(-1)
    sample = sample[np.isfinite(sample)]
    observations = int(sample.size)
    if max_lag is not None and int(max_lag) != max_lag:
        raise ValueError("max_lag must be an integer or None")
    if max_lag is not None and max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    if int(minimum_lag) != minimum_lag or minimum_lag < 0:
        raise ValueError("minimum_lag must be a non-negative integer")
    if observations == 0:
        return {
            "observations": 0,
            "mean": np.nan,
            "max_lag": 0,
            "standard_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_95_low": np.nan,
            "ci_95_high": np.nan,
        }
    mean = float(sample.mean())
    if observations == 1:
        return {
            "observations": 1,
            "mean": mean,
            "max_lag": 0,
            "standard_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_95_low": np.nan,
            "ci_95_high": np.nan,
        }
    automatic = floor(4 * (observations / 100) ** (2 / 9))
    lag = min(
        observations - 1,
        max(int(minimum_lag), automatic) if max_lag is None else int(max_lag),
    )
    centered = sample - mean
    long_run_variance = float(np.dot(centered, centered) / observations)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / observations)
        weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    variance_of_mean = max(long_run_variance, 0.0) / observations
    standard_error = sqrt(variance_of_mean)
    if standard_error == 0.0:
        t_stat = 0.0 if mean == 0.0 else np.sign(mean) * np.inf
        p_value = 1.0 if mean == 0.0 else 0.0
    else:
        t_stat = mean / standard_error
        p_value = erfc(abs(t_stat) / sqrt(2.0))
    return {
        "observations": observations,
        "mean": mean,
        "max_lag": lag,
        "standard_error": standard_error,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_95_low": mean - 1.96 * standard_error,
        "ci_95_high": mean + 1.96 * standard_error,
    }


def holm_adjust(p_values: object) -> np.ndarray:
    """Return Holm-adjusted p-values while preserving input order and NaNs."""

    values = np.asarray(p_values, dtype=float).reshape(-1)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not finite_indices.size:
        return adjusted
    ordered = finite_indices[np.argsort(values[finite_indices], kind="mergesort")]
    family_size = len(ordered)
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, (family_size - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


__all__ = ["holm_adjust", "newey_west_mean_inference"]
