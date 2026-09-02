"""Small framework-neutral population drift diagnostics for research/live monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np

SIGNAL_DRIFT_SCHEMA = "alpha_research.signal_drift.v1"


def _finite_array(values: Iterable[float], *, label: str, min_observations: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) < min_observations:
        raise ValueError(
            f"{label} has {len(array)} finite observations; min_observations={min_observations}"
        )
    return array


def _population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int,
    epsilon: float = 1e-8,
) -> float:
    reference_span = float(np.ptp(reference))
    scale = max(abs(float(reference[0])), 1.0)
    if reference_span <= np.finfo(float).eps * scale:
        center = float(reference[0])
        half_width = max(abs(center) * 1e-6, 1e-12)
        edges = np.array([-np.inf, center - half_width, center + half_width, np.inf])
    else:
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        raw_edges = np.quantile(reference, quantiles)
        interior = np.unique(raw_edges[1:-1])
        edges = np.concatenate(([-np.inf], interior, [np.inf]))
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    reference_share = reference_counts.astype(float) / len(reference)
    current_share = current_counts.astype(float) / len(current)
    reference_share = np.clip(reference_share, epsilon, None)
    current_share = np.clip(current_share, epsilon, None)
    value = np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    return float(value)


def _ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    support = np.sort(np.unique(np.concatenate((reference, current))))
    ref_cdf = np.searchsorted(np.sort(reference), support, side="right") / len(reference)
    cur_cdf = np.searchsorted(np.sort(current), support, side="right") / len(current)
    return float(np.max(np.abs(ref_cdf - cur_cdf))) if len(support) else 0.0


@dataclass(frozen=True)
class SignalDriftReport:
    reference_count: int
    current_count: int
    psi: float
    ks_statistic: float
    mean_shift_std: float | None
    std_ratio: float | None
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    reference_constant: bool
    bins: int
    schema_version: str = SIGNAL_DRIFT_SCHEMA

    def receipt(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "reference_count": self.reference_count,
            "current_count": self.current_count,
            "psi": self.psi,
            "ks_statistic": self.ks_statistic,
            "mean_shift_std": self.mean_shift_std,
            "std_ratio": self.std_ratio,
            "reference_mean": self.reference_mean,
            "current_mean": self.current_mean,
            "reference_std": self.reference_std,
            "current_std": self.current_std,
            "reference_constant": self.reference_constant,
            "bins": self.bins,
        }


def summarize_signal_drift(
    reference: Iterable[float],
    current: Iterable[float],
    *,
    bins: int = 10,
    min_observations: int = 20,
) -> SignalDriftReport:
    """Summarize distribution shift without assigning a strategy lifecycle verdict.

    PSI and KS provide complementary distribution diagnostics. ``mean_shift_std``
    expresses the mean change in reference-standard-deviation units when the
    reference population has non-zero variance. Thresholds and promotion/live
    decisions deliberately stay outside this module.
    """

    if bins < 2:
        raise ValueError("bins must be >= 2")
    if min_observations < 2:
        raise ValueError("min_observations must be >= 2")
    ref = _finite_array(reference, label="reference", min_observations=min_observations)
    cur = _finite_array(current, label="current", min_observations=min_observations)

    reference_mean = float(np.mean(ref))
    current_mean = float(np.mean(cur))
    reference_std = float(np.std(ref, ddof=1))
    current_std = float(np.std(cur, ddof=1))
    reference_constant = reference_std <= np.finfo(float).eps
    mean_shift_std = None if reference_constant else (current_mean - reference_mean) / reference_std
    std_ratio = None if reference_constant else current_std / reference_std

    report = SignalDriftReport(
        reference_count=len(ref),
        current_count=len(cur),
        psi=_population_stability_index(ref, cur, bins=bins),
        ks_statistic=_ks_statistic(ref, cur),
        mean_shift_std=float(mean_shift_std) if mean_shift_std is not None else None,
        std_ratio=float(std_ratio) if std_ratio is not None else None,
        reference_mean=reference_mean,
        current_mean=current_mean,
        reference_std=reference_std,
        current_std=current_std,
        reference_constant=reference_constant,
        bins=bins,
    )
    for value in (report.psi, report.ks_statistic, report.reference_mean, report.current_mean):
        if not isfinite(value):
            raise ValueError("drift diagnostics must be finite")
    return report


__all__ = ["SIGNAL_DRIFT_SCHEMA", "SignalDriftReport", "summarize_signal_drift"]
