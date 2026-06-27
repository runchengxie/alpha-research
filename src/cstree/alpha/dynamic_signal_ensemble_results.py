from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .dynamic_signal_ensemble_types import DynamicSignalEnsembleConfig, FactorMetricBundle


@dataclass(frozen=True)
class _DynamicEnsembleInputs:
    cfg: DynamicSignalEnsembleConfig
    returns: pd.DataFrame
    risk_panels: dict[str, pd.DataFrame]
    oriented_panels: dict[str, pd.DataFrame]
    metrics: FactorMetricBundle
    rolling: dict[str, pd.DataFrame]
    direction_report: pd.DataFrame


@dataclass(frozen=True)
class _DynamicEnsembleRebalance:
    snapshot: dict[str, pd.Series]
    strength: pd.Series
    factor_weights: pd.Series
    stock_weights: pd.Series
    scores: pd.Series
    selected: list[str]
    reasons: dict[str, str]
    factor_turnover: float
    stock_turnover: float
    realized: float


@dataclass(frozen=True)
class _DynamicEnsembleFrames:
    stock_scores: pd.DataFrame
    stock_weights: pd.DataFrame
    factor_weights: pd.DataFrame
    factor_monitor: pd.DataFrame
    portfolio_monitor: pd.DataFrame


def _dynamic_ensemble_frames_from_history(
    *,
    assets: pd.Index,
    factors: list[str],
    stock_weight_history: dict[pd.Timestamp, pd.Series],
    factor_weight_history: dict[pd.Timestamp, pd.Series],
    score_history: dict[pd.Timestamp, pd.Series],
    factor_monitor_rows: list[dict[str, Any]],
    portfolio_monitor_rows: list[dict[str, Any]],
) -> _DynamicEnsembleFrames:
    stock_scores = pd.DataFrame(score_history).T.reindex(columns=assets)
    stock_weights = pd.DataFrame(stock_weight_history).T.reindex(columns=assets)
    factor_weights = pd.DataFrame(factor_weight_history).T.reindex(columns=factors)
    factor_monitor = pd.DataFrame(factor_monitor_rows)
    portfolio_monitor = pd.DataFrame(portfolio_monitor_rows)
    if not factor_monitor.empty:
        factor_monitor = factor_monitor.sort_values(["date", "factor"]).reset_index(drop=True)
    if not portfolio_monitor.empty:
        portfolio_monitor = portfolio_monitor.sort_values("date").reset_index(drop=True)
    return _DynamicEnsembleFrames(
        stock_scores=stock_scores,
        stock_weights=stock_weights,
        factor_weights=factor_weights,
        factor_monitor=factor_monitor,
        portfolio_monitor=portfolio_monitor,
    )


def _build_dynamic_ensemble_summary(
    inputs: _DynamicEnsembleInputs,
    frames: _DynamicEnsembleFrames,
) -> dict[str, Any]:
    cfg = inputs.cfg
    return {
        "schema_version": 1,
        "artifact_type": "cstree.dynamic_signal_ensemble",
        "no_level2": True,
        "rolling_metrics_shifted": True,
        "date_count": len(inputs.returns.index),
        "signal_count": len(inputs.oriented_panels),
        "stock_score_dates": len(frames.stock_scores),
        "risk_penalty_enabled": bool(cfg.risk_penalty_scale > 0 and inputs.risk_panels),
        "correlation_threshold": float(cfg.correlation_threshold),
        "avg_active_factor_count": (
            float(frames.portfolio_monitor["active_factor_count"].mean())
            if not frames.portfolio_monitor.empty
            else None
        ),
        "avg_factor_turnover": (
            float(frames.portfolio_monitor["factor_turnover"].mean())
            if not frames.portfolio_monitor.empty
            else None
        ),
        "avg_stock_turnover": (
            float(frames.portfolio_monitor["stock_turnover"].mean())
            if not frames.portfolio_monitor.empty
            else None
        ),
        "config": asdict(cfg),
    }
