"""Fixed-width fractional differentiation with point-in-time order selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FractionalDifferenceSelection:
    d: float
    adf_t_stat: float
    correlation: float
    observations: int
    threshold: float


def fractional_difference_weights(
    d: float,
    *,
    threshold: float = 1e-5,
    max_size: int = 10000,
) -> np.ndarray:
    """Return fixed-width fractional-difference weights, oldest first."""

    if d < 0:
        raise ValueError("d must be >= 0")
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    if max_size <= 0:
        raise ValueError("max_size must be > 0")
    weights = [1.0]
    for k in range(1, max_size):
        next_weight = -weights[-1] * (d - k + 1.0) / k
        if abs(next_weight) < threshold:
            break
        weights.append(float(next_weight))
    return np.asarray(weights[::-1], dtype=float)


def fixed_width_fractional_difference(
    series: pd.Series,
    d: float,
    *,
    threshold: float = 1e-5,
) -> pd.Series:
    """Apply fixed-width fractional differentiation without expanding-window drift."""

    values = pd.to_numeric(series, errors="coerce").astype(float)
    weights = fractional_difference_weights(d, threshold=threshold)
    width = len(weights)
    output = pd.Series(np.nan, index=values.index, dtype=float, name=series.name)
    if width > len(values):
        return output
    raw = values.to_numpy(dtype=float)
    for end in range(width - 1, len(raw)):
        window = raw[end - width + 1 : end + 1]
        if np.isnan(window).any():
            continue
        output.iloc[end] = float(np.dot(weights, window))
    return output


def augmented_dickey_fuller_t_stat(
    series: pd.Series,
    *,
    lags: int = 1,
) -> float:
    """Estimate the ADF t-statistic for the lagged level coefficient."""

    if lags < 0:
        raise ValueError("lags must be >= 0")
    y = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if y.size < max(8, lags + 5):
        return float("nan")
    delta = np.diff(y)
    rows: list[list[float]] = []
    target: list[float] = []
    for t in range(lags, len(delta)):
        row = [1.0, y[t]]
        for lag in range(1, lags + 1):
            row.append(delta[t - lag])
        rows.append(row)
        target.append(delta[t])
    x = np.asarray(rows, dtype=float)
    response = np.asarray(target, dtype=float)
    if x.shape[0] <= x.shape[1]:
        return float("nan")
    beta, *_ = np.linalg.lstsq(x, response, rcond=None)
    residuals = response - x @ beta
    degrees = x.shape[0] - x.shape[1]
    sigma2 = float(residuals @ residuals / degrees)
    covariance = sigma2 * np.linalg.pinv(x.T @ x)
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    return float(beta[1] / standard_error) if standard_error > 0 else float("nan")


def select_fractional_difference_order(
    series: pd.Series,
    *,
    d_values: Iterable[float] | None = None,
    threshold: float = 1e-5,
    adf_critical_value: float = -2.86,
    lags: int = 1,
    min_observations: int = 50,
) -> FractionalDifferenceSelection:
    """Choose the smallest stationary ``d`` while retaining maximum memory."""

    candidates = list(d_values if d_values is not None else np.linspace(0.0, 1.0, 11))
    if not candidates:
        raise ValueError("d_values must not be empty")
    original = pd.to_numeric(series, errors="coerce")
    results: list[FractionalDifferenceSelection] = []
    for d in sorted(float(value) for value in candidates):
        transformed = fixed_width_fractional_difference(original, d, threshold=threshold)
        aligned = pd.concat(
            [original.rename("original"), transformed.rename("transformed")], axis=1
        ).dropna()
        if len(aligned) < min_observations:
            continue
        statistic = augmented_dickey_fuller_t_stat(aligned["transformed"], lags=lags)
        correlation = float(aligned["original"].corr(aligned["transformed"]))
        results.append(
            FractionalDifferenceSelection(
                d=d,
                adf_t_stat=statistic,
                correlation=correlation,
                observations=len(aligned),
                threshold=threshold,
            )
        )
    if not results:
        raise ValueError("No fractional-difference candidate has enough observations")
    stationary = [
        result
        for result in results
        if np.isfinite(result.adf_t_stat) and result.adf_t_stat <= adf_critical_value
    ]
    if stationary:
        return min(stationary, key=lambda result: (result.d, -result.correlation))
    return min(results, key=lambda result: result.adf_t_stat)


__all__ = [
    "FractionalDifferenceSelection",
    "augmented_dickey_fuller_t_stat",
    "fixed_width_fractional_difference",
    "fractional_difference_weights",
    "select_fractional_difference_order",
]
