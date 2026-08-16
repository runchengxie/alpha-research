"""Dynamic signal ensemble: factor selection and weight construction.

Private helpers that select factors by rolling diagnostics and correlation,
build factor / stock weights, and aggregate per-date stock scores. Split out
of the historical single-file :mod:`alpha_research.dynamic_signal_ensemble`
implementation to keep individual files smaller while preserving the exact
public/private symbol surface.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ._dynamic_signal_ensemble_metrics import _passes_min
from .dynamic_signal_ensemble_math import _cap_positive_weights, _zscore_series
from .dynamic_signal_ensemble_types import DynamicSignalEnsembleConfig, FactorMetricBundle


def _rolling_factor_correlation(
    factor_panels: dict[str, pd.DataFrame],
    factors: list[str],
    date: pd.Timestamp,
    config: DynamicSignalEnsembleConfig,
) -> pd.DataFrame:
    if not factors:
        return pd.DataFrame()
    first_panel = next(iter(factor_panels.values()))
    hist_dates = first_panel.index[first_panel.index < date][-config.covariance_window :]
    if len(hist_dates) == 0:
        return pd.DataFrame(
            np.eye(len(factors)),
            index=pd.Index(factors),
            columns=pd.Index(factors),
        )
    flattened = {}
    for factor in factors:
        panel = factor_panels[factor].reindex(hist_dates)
        flattened[factor] = panel.stack(future_stack=True).reset_index(drop=True)
    history = pd.DataFrame(flattened).replace([np.inf, -np.inf], np.nan)
    corr = history.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for column in corr.columns:
        corr.loc[column, column] = 1.0
    return corr


def _select_factors(
    *,
    date: pd.Timestamp,
    factors: list[str],
    strength: pd.Series,
    diagnostics: dict[str, pd.Series],
    factor_panels: dict[str, pd.DataFrame],
    metrics: FactorMetricBundle,
    config: DynamicSignalEnsembleConfig,
) -> tuple[list[str], dict[str, str], pd.DataFrame]:
    reasons: dict[str, str] = {}
    candidates: list[str] = []
    checks = {
        "icir": ("icir_below_threshold", config.min_icir),
        "rank_ic_mean": ("rank_ic_mean_below_threshold", config.min_rank_ic_mean),
        "long_short_sharpe": ("long_short_below_threshold", config.min_long_short_sharpe),
        "stability": ("stability_below_threshold", config.min_stability),
        "direction_consistency": (
            "direction_inconsistent",
            config.min_direction_consistency,
        ),
        "coverage_ratio": ("coverage_below_threshold", config.min_coverage_ratio),
        "dispersion": ("dispersion_below_threshold", config.min_signal_dispersion),
    }
    for factor in factors:
        values = {name: diagnostics[name].get(factor, np.nan) for name in checks}
        failed = False
        for name, (reason, threshold) in checks.items():
            if threshold is None:
                continue
            if pd.isna(values[name]):
                reasons[factor] = "insufficient_history"
                failed = True
                break
            if not _passes_min(float(values[name]), threshold):
                reasons[factor] = reason
                failed = True
                break
        if failed:
            continue
        threshold = config.selection_threshold
        if threshold is not None and strength.get(factor, np.nan) < threshold:
            reasons[factor] = "strength_below_threshold"
            continue
        candidates.append(factor)

    if not candidates:
        fallback_count = max(int(config.fallback_factor_count), 1)
        candidates = (
            strength.dropna().sort_values(ascending=False).head(fallback_count).index.tolist()
        )
        for factor in candidates:
            reasons.pop(factor, None)

    corr = _rolling_factor_correlation(factor_panels, candidates, date, config)
    selected: list[str] = []
    for factor in strength.reindex(candidates).sort_values(ascending=False).index:
        if not selected:
            selected.append(factor)
            continue
        corr_slice = corr.reindex(index=[factor], columns=selected)
        max_corr = corr_slice.abs().max(axis=1).iloc[0]
        if pd.notna(max_corr) and float(max_corr) > config.correlation_threshold:
            reasons[factor] = "correlation_filtered"
            continue
        selected.append(factor)
    for factor in factors:
        reasons.setdefault(factor, "")
    return selected, reasons, corr


def _factor_weights(
    *,
    alpha: pd.Series,
    selected: list[str],
    corr: pd.DataFrame,
    config: DynamicSignalEnsembleConfig,
) -> pd.Series:
    if not selected:
        return pd.Series(0.0, index=alpha.index, dtype=float)
    mode = str(config.factor_weight_mode or "strength").strip().lower()
    if mode == "equal":
        raw = pd.Series(1.0, index=selected, dtype=float)
    else:
        raw = alpha.reindex(selected).fillna(0.0).clip(lower=0.0)
        if mode == "optimized" and len(selected) > 1:
            avg_corr = corr.reindex(index=selected, columns=selected).abs().mean(axis=1).fillna(0.0)
            raw = raw.subtract(0.10 * avg_corr, fill_value=0.0).clip(lower=0.0)
        if raw.sum() <= 0:
            raw = pd.Series(1.0, index=selected, dtype=float)
    capped = _cap_positive_weights(raw, config.max_factor_weight)
    return capped.reindex(alpha.index).fillna(0.0)


def _aggregate_stock_scores(
    *,
    date: pd.Timestamp,
    factor_panels: dict[str, pd.DataFrame],
    weights: pd.Series,
) -> pd.Series:
    score: pd.Series | None = None
    for factor, weight in weights.items():
        factor = cast(str, factor)
        if weight <= 0 or factor not in factor_panels:
            continue
        values = factor_panels[factor].loc[date].astype(float)
        values = _zscore_series(values)
        contribution = values * float(weight)
        score = contribution if score is None else score.add(contribution, fill_value=0.0)
    return pd.Series(dtype=float) if score is None else score.sort_values(ascending=False)


def _risk_penalty_for_date(
    *,
    date: pd.Timestamp,
    risk_panels: dict[str, pd.DataFrame],
    assets: pd.Index,
) -> pd.Series:
    if not risk_panels:
        return pd.Series(0.0, index=assets, dtype=float)
    penalties = []
    for panel in risk_panels.values():
        if date not in panel.index:
            continue
        exposures = _zscore_series(panel.loc[date].reindex(assets).astype(float)).abs()
        penalties.append(exposures)
    if not penalties:
        return pd.Series(0.0, index=assets, dtype=float)
    return pd.concat(penalties, axis=1).mean(axis=1).fillna(0.0)


def _stock_weights(
    scores: pd.Series,
    *,
    previous_holdings: list[str],
    config: DynamicSignalEnsembleConfig,
) -> tuple[pd.Series, list[str]]:
    clean = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or config.stock_selection_count <= 0:
        return pd.Series(dtype=float), []
    ranked = clean.sort_values(ascending=False)
    k = min(int(config.stock_selection_count), len(ranked))
    keep_limit = min(len(ranked), k + max(int(config.stock_buffer_count), 0))
    keep = set(ranked.head(keep_limit).index) & set(previous_holdings)
    holdings = [symbol for symbol in ranked.index if symbol in keep]
    for symbol in ranked.index:
        if len(holdings) >= k:
            break
        if symbol not in holdings:
            holdings.append(symbol)
    selected_scores = ranked.reindex(holdings)
    if str(config.stock_weight_mode).lower() == "score":
        zscores = _zscore_series(selected_scores).clip(-5.0, 5.0)
        raw = pd.Series(np.exp(zscores), index=selected_scores.index, dtype=float)
    else:
        raw = pd.Series(1.0, index=selected_scores.index, dtype=float)
    return _cap_positive_weights(raw, config.max_stock_weight), holdings
