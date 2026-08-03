"""Dynamic signal ensemble: rebalance orchestration and CLI.

Public entry points ``build_dynamic_signal_ensemble`` / ``attach_dynamic_ensemble_score``
plus the ``add_dynamic_signal_ensemble_args`` / ``run`` CLI helpers, backed by the
historical single-file :mod:`alpha_research.dynamic_signal_ensemble`
implementation. The rebalance loop and input wiring live here; per-factor
metrics and factor selection live in the sibling ``_dynamic_signal_ensemble_*``
submodules. Split to keep individual files smaller while preserving the exact
public/private symbol surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ._dynamic_signal_ensemble_metrics import (
    _combine_strength,
    compute_factor_metrics,
    compute_rolling_diagnostics,
)
from ._dynamic_signal_ensemble_select import (
    _aggregate_stock_scores,
    _factor_weights,
    _risk_penalty_for_date,
    _select_factors,
    _stock_weights,
)
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
from .dynamic_signal_ensemble_math import _apply_turnover_budget
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
)


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
        cast(Any, stock_weights.reindex(assets).fillna(0.0).dot(inputs.returns.loc[date].fillna(0.0)))
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
