from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DynamicSignalEnsembleConfig:
    min_history: int = 12
    evaluation_window: int = 12
    covariance_window: int = 12
    regime_window: int = 24
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    min_icir: float | None = None
    min_long_short_sharpe: float | None = None
    min_stability: float | None = 0.45
    min_coverage_ratio: float | None = 0.55
    min_signal_dispersion: float | None = None
    min_rank_ic_mean: float | None = None
    min_direction_consistency: float | None = 0.55
    selection_threshold: float | None = None
    correlation_threshold: float = 0.75
    fallback_factor_count: int = 1
    factor_weight_mode: str = "strength"
    max_factor_weight: float = 1.0
    max_factor_turnover: float = 1.0
    stock_selection_count: int = 20
    stock_buffer_count: int = 0
    max_stock_weight: float | None = None
    max_stock_turnover: float = 2.0
    stock_weight_mode: str = "equal"
    risk_penalty_scale: float = 0.0
    flip_mean_threshold: float = 0.02
    flip_consistency_threshold: float = 0.60
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "icir": 0.40,
            "long_short": 0.30,
            "stability": 0.20,
            "regime": 0.10,
        }
    )


@dataclass(frozen=True)
class FactorMetricBundle:
    rank_ic: pd.DataFrame
    long_short: pd.DataFrame
    long_leg: pd.DataFrame
    short_leg: pd.DataFrame
    coverage: pd.DataFrame
    coverage_ratio: pd.DataFrame
    dispersion: pd.DataFrame


@dataclass(frozen=True)
class DynamicSignalEnsembleResult:
    stock_scores: pd.DataFrame
    stock_weights: pd.DataFrame
    factor_weights: pd.DataFrame
    factor_monitor: pd.DataFrame
    portfolio_monitor: pd.DataFrame
    direction_calibration: pd.DataFrame
    factor_metrics: FactorMetricBundle
    summary: dict[str, Any]
