from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dynamic_signal_ensemble_artifacts import (
    stock_scores_to_long,
    write_dynamic_ensemble_artifacts,
)
from .dynamic_signal_ensemble_calibration import (
    _apply_direction_panels,
    _compute_raw_rank_ic,
    calibrate_signal_directions,
)
from .dynamic_signal_ensemble_io import (
    _coerce_date_column,
    _config_from_mapping,
    _load_from_signal_files,
    _load_regime_features,
    _load_table,
    _load_yaml,
    _normalize_long_frame,
    _panel_from_long,
    _pivot_panel,
    _resolve_path,
    _section,
)
from .dynamic_signal_ensemble_math import (
    _apply_turnover_budget,
    _cap_positive_weights,
    _cross_sectional_zscore_frame,
    _zscore_series,
)
from .dynamic_signal_ensemble_results import (
    _build_dynamic_ensemble_summary,
    _dynamic_ensemble_frames_from_history,
    _DynamicEnsembleFrames,
    _DynamicEnsembleInputs,
    _DynamicEnsembleRebalance,
)
from .dynamic_signal_ensemble_types import (
    DynamicSignalEnsembleConfig,
    DynamicSignalEnsembleResult,
    FactorMetricBundle,
)


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
            index=factors,
            columns=factors,
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


def _align_inputs(
    panels: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    common_index = returns.index
    common_columns = returns.columns
    for panel in panels.values():
        common_index = common_index.intersection(panel.index)
        common_columns = common_columns.intersection(panel.columns)
    if len(common_index) == 0 or len(common_columns) == 0:
        raise SystemExit("Dynamic ensemble inputs have no overlapping dates or symbols.")
    aligned_returns = returns.reindex(index=common_index, columns=common_columns).sort_index()
    aligned_panels = {
        name: panel.reindex(
            index=aligned_returns.index,
            columns=aligned_returns.columns,
        ).sort_index()
        for name, panel in panels.items()
    }
    return aligned_panels, aligned_returns


def _prepare_dynamic_ensemble_inputs(
    data: pd.DataFrame,
    *,
    signal_cols: list[str],
    target_col: str,
    date_col: str,
    symbol_col: str,
    risk_cols: list[str] | None,
    regime_features: pd.DataFrame | None,
    config: DynamicSignalEnsembleConfig | None,
) -> _DynamicEnsembleInputs:
    cfg = config or DynamicSignalEnsembleConfig()
    frame = _normalize_long_frame(data, date_col=date_col, symbol_col=symbol_col)
    if not signal_cols:
        raise SystemExit("dynamic_signal_ensemble.signal_cols must be non-empty.")
    if target_col not in frame.columns:
        raise SystemExit(f"Missing target column for dynamic ensemble: {target_col}")
    panels = _panel_from_long(frame, date_col=date_col, symbol_col=symbol_col, columns=signal_cols)
    returns = _pivot_panel(frame, date_col=date_col, symbol_col=symbol_col, value_col=target_col)
    panels, returns = _align_inputs(panels, returns)
    risk_panels = _panel_from_long(
        frame,
        date_col=date_col,
        symbol_col=symbol_col,
        columns=list(risk_cols or []),
    )
    risk_panels = {
        name: panel.reindex(index=returns.index, columns=returns.columns)
        for name, panel in risk_panels.items()
    }
    raw_rank_ic = _compute_raw_rank_ic(panels, returns)
    directions, direction_report = calibrate_signal_directions(raw_rank_ic, cfg)
    oriented_panels = _apply_direction_panels(panels, directions)
    metrics = compute_factor_metrics(oriented_panels, returns, cfg)
    rolling = compute_rolling_diagnostics(metrics, cfg, regime_features=regime_features)
    return _DynamicEnsembleInputs(
        cfg=cfg,
        returns=returns,
        risk_panels=risk_panels,
        oriented_panels=oriented_panels,
        metrics=metrics,
        rolling=rolling,
        direction_report=direction_report,
    )


def _run_dynamic_ensemble_rebalance(
    inputs: _DynamicEnsembleInputs,
    *,
    date: pd.Timestamp,
    factors: list[str],
    assets: pd.Index,
    prev_factor_weights: pd.Series,
    prev_stock_weights: pd.Series,
    prev_holdings: list[str],
) -> _DynamicEnsembleRebalance:
    cfg = inputs.cfg
    snapshot = {
        name: frame_.loc[date].reindex(factors).astype(float)
        for name, frame_ in inputs.rolling.items()
    }
    strength = _combine_strength(snapshot, cfg)
    alpha = strength.clip(lower=0.0)
    selected, reasons, corr = _select_factors(
        date=date,
        factors=factors,
        strength=strength,
        diagnostics=snapshot,
        factor_panels=inputs.oriented_panels,
        metrics=inputs.metrics,
        config=cfg,
    )
    target_factor_weights = _factor_weights(alpha=alpha, selected=selected, corr=corr, config=cfg)
    factor_weights = _apply_turnover_budget(
        previous=prev_factor_weights,
        target=target_factor_weights.reindex(factors).fillna(0.0),
        max_l1_turnover=cfg.max_factor_turnover,
    )
    scores = _aggregate_stock_scores(
        date=date,
        factor_panels=inputs.oriented_panels,
        weights=factor_weights,
    )
    if cfg.risk_penalty_scale > 0 and inputs.risk_panels:
        penalty = _risk_penalty_for_date(
            date=date,
            risk_panels=inputs.risk_panels,
            assets=scores.index,
        )
        scores = scores.subtract(penalty * float(cfg.risk_penalty_scale), fill_value=0.0)
    target_stock_weights, _ = _stock_weights(
        scores,
        previous_holdings=prev_holdings,
        config=cfg,
    )
    stock_weights = _apply_turnover_budget(
        previous=prev_stock_weights,
        target=target_stock_weights.reindex(assets).fillna(0.0),
        max_l1_turnover=cfg.max_stock_turnover,
    )
    realized = float(
        stock_weights.reindex(assets).fillna(0.0).dot(inputs.returns.loc[date].fillna(0.0))
    )
    return _DynamicEnsembleRebalance(
        snapshot=snapshot,
        strength=strength,
        factor_weights=factor_weights,
        stock_weights=stock_weights,
        scores=scores,
        selected=selected,
        reasons=reasons,
        factor_turnover=float((factor_weights - prev_factor_weights).abs().sum()),
        stock_turnover=float((stock_weights - prev_stock_weights).abs().sum()),
        realized=realized,
    )


def _append_factor_monitor_rows(
    rows: list[dict[str, Any]],
    *,
    date: pd.Timestamp,
    factors: list[str],
    rebalance: _DynamicEnsembleRebalance,
) -> None:
    snapshot = rebalance.snapshot
    for factor in factors:
        rows.append(
            {
                "date": date,
                "factor": factor,
                "raw_strength": rebalance.strength.get(factor, np.nan),
                "selected": factor in rebalance.selected,
                "drop_reason": rebalance.reasons.get(factor, ""),
                "factor_weight": rebalance.factor_weights.get(factor, 0.0),
                "rank_ic_mean": snapshot["rank_ic_mean"].get(factor, np.nan),
                "icir": snapshot["icir"].get(factor, np.nan),
                "long_short_sharpe": snapshot["long_short_sharpe"].get(factor, np.nan),
                "coverage_ratio": snapshot["coverage_ratio"].get(factor, np.nan),
                "dispersion": snapshot["dispersion"].get(factor, np.nan),
                "stability": snapshot["stability"].get(factor, np.nan),
                "direction_consistency": snapshot["direction_consistency"].get(factor, np.nan),
                "regime_score": snapshot["regime_score"].get(factor, np.nan),
            }
        )


def _append_portfolio_monitor_row(
    rows: list[dict[str, Any]],
    *,
    date: pd.Timestamp,
    rebalance: _DynamicEnsembleRebalance,
) -> None:
    rows.append(
        {
            "date": date,
            "portfolio_return": rebalance.realized,
            "factor_turnover": rebalance.factor_turnover,
            "stock_turnover": rebalance.stock_turnover,
            "active_factor_count": int((rebalance.factor_weights > 0).sum()),
            "holding_count": int((rebalance.stock_weights > 0).sum()),
        }
    )


def _run_dynamic_ensemble_rebalances(inputs: _DynamicEnsembleInputs) -> _DynamicEnsembleFrames:
    factors = list(inputs.oriented_panels)
    assets = inputs.returns.columns
    prev_factor_weights = pd.Series(0.0, index=factors)
    prev_stock_weights = pd.Series(0.0, index=assets)
    prev_holdings: list[str] = []
    factor_weight_history: dict[pd.Timestamp, pd.Series] = {}
    stock_weight_history: dict[pd.Timestamp, pd.Series] = {}
    score_history: dict[pd.Timestamp, pd.Series] = {}
    factor_monitor_rows: list[dict[str, Any]] = []
    portfolio_monitor_rows: list[dict[str, Any]] = []

    start_idx = max(inputs.cfg.min_history, inputs.cfg.evaluation_window)
    for idx in range(start_idx, len(inputs.returns.index)):
        date = inputs.returns.index[idx]
        rebalance = _run_dynamic_ensemble_rebalance(
            inputs,
            date=date,
            factors=factors,
            assets=assets,
            prev_factor_weights=prev_factor_weights,
            prev_stock_weights=prev_stock_weights,
            prev_holdings=prev_holdings,
        )
        factor_weight_history[date] = rebalance.factor_weights.reindex(factors).fillna(0.0)
        stock_weight_history[date] = rebalance.stock_weights.reindex(assets).fillna(0.0)
        score_history[date] = rebalance.scores.reindex(assets)
        _append_factor_monitor_rows(
            factor_monitor_rows,
            date=date,
            factors=factors,
            rebalance=rebalance,
        )
        _append_portfolio_monitor_row(portfolio_monitor_rows, date=date, rebalance=rebalance)
        prev_factor_weights = rebalance.factor_weights.reindex(factors).fillna(0.0)
        prev_stock_weights = rebalance.stock_weights.reindex(assets).fillna(0.0)
        prev_holdings = list(rebalance.stock_weights[rebalance.stock_weights > 0].index)

    return _dynamic_ensemble_frames_from_history(
        assets=assets,
        factors=factors,
        stock_weight_history=stock_weight_history,
        factor_weight_history=factor_weight_history,
        score_history=score_history,
        factor_monitor_rows=factor_monitor_rows,
        portfolio_monitor_rows=portfolio_monitor_rows,
    )


def build_dynamic_signal_ensemble(
    data: pd.DataFrame,
    *,
    signal_cols: list[str],
    target_col: str,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    risk_cols: list[str] | None = None,
    regime_features: pd.DataFrame | None = None,
    config: DynamicSignalEnsembleConfig | None = None,
) -> DynamicSignalEnsembleResult:
    inputs = _prepare_dynamic_ensemble_inputs(
        data,
        signal_cols=signal_cols,
        target_col=target_col,
        date_col=date_col,
        symbol_col=symbol_col,
        risk_cols=risk_cols,
        regime_features=regime_features,
        config=config,
    )
    frames = _run_dynamic_ensemble_rebalances(inputs)
    return DynamicSignalEnsembleResult(
        stock_scores=frames.stock_scores,
        stock_weights=frames.stock_weights,
        factor_weights=frames.factor_weights,
        factor_monitor=frames.factor_monitor,
        portfolio_monitor=frames.portfolio_monitor,
        direction_calibration=inputs.direction_report,
        factor_metrics=inputs.metrics,
        summary=_build_dynamic_ensemble_summary(inputs, frames),
    )


def attach_dynamic_ensemble_score(
    data: pd.DataFrame,
    *,
    spec: dict[str, Any],
    target_col: str,
    default_date_col: str = "trade_date",
    default_symbol_col: str = "symbol",
) -> tuple[pd.DataFrame, str, DynamicSignalEnsembleResult]:
    date_col = str(spec.get("date_col") or default_date_col)
    symbol_col = str(spec.get("symbol_col") or default_symbol_col)
    signal_cols = [str(col) for col in spec.get("signal_cols", [])]
    risk_cols = [str(col) for col in spec.get("risk_cols", spec.get("risk_columns", []))]
    output_col = str(spec.get("output_col") or "__dynamic_ensemble_score")
    cfg = _config_from_mapping(spec.get("config", spec))
    result = build_dynamic_signal_ensemble(
        data,
        signal_cols=signal_cols,
        target_col=str(spec.get("target_col") or target_col),
        date_col=date_col,
        symbol_col=symbol_col,
        risk_cols=risk_cols,
        config=cfg,
    )
    scores_long = stock_scores_to_long(result.stock_scores, score_col=output_col)
    out = data.copy()
    out[date_col] = _coerce_date_column(out[date_col])
    merged = out.merge(scores_long, on=[date_col, symbol_col], how="left")
    fallback_col = str(spec.get("fallback_col") or (signal_cols[0] if signal_cols else ""))
    if fallback_col and fallback_col in merged.columns:
        merged[output_col] = merged[output_col].fillna(
            pd.to_numeric(merged[fallback_col], errors="coerce")
        )
    return merged, output_col, result


def _build_from_config(
    config: dict[str, Any],
    *,
    config_dir: Path,
) -> tuple[DynamicSignalEnsembleResult, dict[str, Any]]:
    cfg = _section(config)
    input_path = _resolve_path(cfg.get("input_file"), base_dir=config_dir)
    if input_path is not None:
        data = _load_table(input_path)
        date_col = str(cfg.get("date_col") or "trade_date")
        symbol_col = str(cfg.get("symbol_col") or "symbol")
        signal_cols = [str(col) for col in cfg.get("signal_cols", [])]
        target_col = str(cfg.get("target_col") or "future_return")
    else:
        data = _load_from_signal_files(cfg, config_dir=config_dir)
        date_col = "trade_date"
        symbol_col = "symbol"
        target_col = str(cfg.get("target_col") or cfg.get("returns_col") or "future_return")
        signal_cols = [
            str((item if isinstance(item, dict) else {}).get("name") or Path(str(item)).parent.name)
            for item in cfg.get("signal_files", [])
        ]
    risk_cols = [str(col) for col in cfg.get("risk_cols", cfg.get("risk_columns", []))]
    result = build_dynamic_signal_ensemble(
        data,
        signal_cols=signal_cols,
        target_col=target_col,
        date_col=date_col,
        symbol_col=symbol_col,
        risk_cols=risk_cols,
        regime_features=_load_regime_features(cfg, config_dir=config_dir),
        config=_config_from_mapping(cfg.get("config", cfg)),
    )
    return result, {
        "date_col": date_col,
        "symbol_col": symbol_col,
        "score_col": str(cfg.get("score_col_out") or "dynamic_ensemble_score"),
    }


def add_dynamic_signal_ensemble_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Dynamic signal ensemble YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override output artifact directory.")
    return parser


def run(args: argparse.Namespace) -> DynamicSignalEnsembleResult:
    config_path = _resolve_path(args.config)
    assert config_path is not None
    config = _load_yaml(config_path)
    result, output_meta = _build_from_config(config, config_dir=config_path.parent)
    cfg = _section(config)
    output_dir = _resolve_path(
        args.output_dir or cfg.get("output_dir") or "artifacts/reports/dynamic_signal_ensemble",
        base_dir=config_path.parent,
    )
    assert output_dir is not None
    write_dynamic_ensemble_artifacts(
        result,
        output_dir=output_dir,
        score_col=str(output_meta["score_col"]),
    )
    return result
