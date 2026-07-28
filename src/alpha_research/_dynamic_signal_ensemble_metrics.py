"""Dynamic signal ensemble: per-factor metrics and rolling diagnostics.

Private helpers that compute single-factor Rank-IC / long-short metrics,
regime-conditioned scores, rolling diagnostics, and combined selection
strength. Split out of the historical single-file
:mod:`alpha_research.dynamic_signal_ensemble` implementation to keep
individual files smaller while preserving the exact public/private symbol
surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .dynamic_signal_ensemble_math import (
    _cross_sectional_zscore_frame,
    _zscore_series,
)
from .dynamic_signal_ensemble_types import DynamicSignalEnsembleConfig, FactorMetricBundle


def _compute_single_factor_metrics(
    factor_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    config: DynamicSignalEnsembleConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_assets = max(len(factor_panel.columns), 1)
    for date in factor_panel.index:
        joined = (
            pd.concat(
                [
                    factor_panel.loc[date].rename("factor"),
                    forward_returns.loc[date].rename("return"),
                ],
                axis=1,
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(joined) < 5:
            rows.append(
                {
                    "date": date,
                    "rank_ic": np.nan,
                    "long_short": np.nan,
                    "long_leg": np.nan,
                    "short_leg": np.nan,
                    "coverage": float(len(joined)),
                    "coverage_ratio": float(len(joined)) / float(total_assets),
                    "dispersion": np.nan,
                }
            )
            continue
        rank_ic = joined["factor"].corr(joined["return"], method="spearman")
        high_count = max(int(np.ceil(len(joined) * config.top_quantile)), 1)
        low_count = max(int(np.ceil(len(joined) * config.bottom_quantile)), 1)
        ranked = joined.sort_values("factor")
        short_leg = ranked.head(low_count)["return"].mean()
        long_leg = ranked.tail(high_count)["return"].mean()
        rows.append(
            {
                "date": date,
                "rank_ic": float(rank_ic),
                "long_short": float(long_leg - short_leg),
                "long_leg": float(long_leg),
                "short_leg": float(short_leg),
                "coverage": float(len(joined)),
                "coverage_ratio": float(len(joined)) / float(total_assets),
                "dispersion": float(joined["factor"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).set_index("date").sort_index()


def compute_factor_metrics(
    panels: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    config: DynamicSignalEnsembleConfig,
) -> FactorMetricBundle:
    stores: dict[str, dict[str, pd.Series]] = {
        "rank_ic": {},
        "long_short": {},
        "long_leg": {},
        "short_leg": {},
        "coverage": {},
        "coverage_ratio": {},
        "dispersion": {},
    }
    for name, panel in panels.items():
        metrics = _compute_single_factor_metrics(panel, forward_returns, config)
        for key in stores:
            stores[key][name] = metrics[key]
    return FactorMetricBundle(
        rank_ic=pd.DataFrame(stores["rank_ic"]).sort_index(),
        long_short=pd.DataFrame(stores["long_short"]).sort_index(),
        long_leg=pd.DataFrame(stores["long_leg"]).sort_index(),
        short_leg=pd.DataFrame(stores["short_leg"]).sort_index(),
        coverage=pd.DataFrame(stores["coverage"]).sort_index(),
        coverage_ratio=pd.DataFrame(stores["coverage_ratio"]).sort_index(),
        dispersion=pd.DataFrame(stores["dispersion"]).sort_index(),
    )


def _compute_regime_scores(
    factor_returns: pd.DataFrame,
    regime_features: pd.DataFrame | None,
    config: DynamicSignalEnsembleConfig,
) -> pd.DataFrame:
    scores = pd.DataFrame(0.0, index=factor_returns.index, columns=factor_returns.columns)
    if regime_features is None or regime_features.empty:
        return scores
    features = regime_features.reindex(factor_returns.index)
    for idx, date in enumerate(factor_returns.index):
        if idx < config.regime_window:
            continue
        hist_dates = factor_returns.index[max(0, idx - config.regime_window) : idx]
        current = features.loc[date]
        factor_scores: list[pd.Series] = []
        for feature in features.columns:
            hist_feature = features.loc[hist_dates, feature].dropna()
            if len(hist_feature) < max(3, config.regime_window // 3):
                continue
            current_value = current.get(feature, np.nan)
            if pd.isna(current_value):
                continue
            low = hist_feature.quantile(1.0 / 3.0)
            high = hist_feature.quantile(2.0 / 3.0)
            if current_value <= low:
                bucket = hist_feature[hist_feature <= low].index
            elif current_value >= high:
                bucket = hist_feature[hist_feature >= high].index
            else:
                bucket = hist_feature[(hist_feature > low) & (hist_feature < high)].index
            if len(bucket):
                factor_scores.append(factor_returns.loc[bucket].mean())
        if factor_scores:
            scores.loc[date] = pd.concat(factor_scores, axis=1).mean(axis=1)
    return _cross_sectional_zscore_frame(scores).fillna(0.0)


def compute_rolling_diagnostics(
    metrics: FactorMetricBundle,
    config: DynamicSignalEnsembleConfig,
    *,
    regime_features: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    window = config.evaluation_window
    rolling_ic_mean = metrics.rank_ic.rolling(window, min_periods=window).mean()
    rolling_ic_std = metrics.rank_ic.rolling(window, min_periods=window).std()
    rolling_ls_mean = metrics.long_short.rolling(window, min_periods=window).mean()
    rolling_ls_std = metrics.long_short.rolling(window, min_periods=window).std()
    stability = 0.5 * (
        metrics.rank_ic.gt(0).rolling(window, min_periods=window).mean()
        + metrics.long_short.gt(0).rolling(window, min_periods=window).mean()
    )
    return {
        "rank_ic_mean": rolling_ic_mean.shift(1),
        "icir": rolling_ic_mean.divide(rolling_ic_std.replace(0.0, np.nan)).shift(1),
        "long_short_sharpe": rolling_ls_mean.divide(rolling_ls_std.replace(0.0, np.nan)).shift(1),
        "stability": stability.shift(1),
        "direction_consistency": metrics.rank_ic.gt(0)
        .rolling(window, min_periods=window)
        .mean()
        .shift(1),
        "coverage_ratio": metrics.coverage_ratio.rolling(
            window,
            min_periods=window,
        )
        .mean()
        .shift(1),
        "dispersion": metrics.dispersion.rolling(window, min_periods=window).mean().shift(1),
        "regime_score": _compute_regime_scores(metrics.long_short, regime_features, config),
    }


def _combine_strength(
    diagnostics: dict[str, pd.Series],
    config: DynamicSignalEnsembleConfig,
) -> pd.Series:
    components = {
        "icir": _zscore_series(diagnostics["icir"]),
        "long_short": _zscore_series(diagnostics["long_short_sharpe"]),
        "stability": _zscore_series(diagnostics["stability"]),
        "regime": _zscore_series(diagnostics["regime_score"]),
    }
    strength = pd.Series(0.0, index=diagnostics["icir"].index, dtype=float)
    total_weight = 0.0
    for name, component in components.items():
        weight = float(config.score_weights.get(name, 0.0))
        if weight == 0:
            continue
        strength = strength.add(component.fillna(0.0) * weight, fill_value=0.0)
        total_weight += abs(weight)
    return strength / total_weight if total_weight > 0 else strength


def _passes_min(value: float | None, threshold: float | None) -> bool:
    return threshold is None or (value is not None and np.isfinite(value) and value >= threshold)
