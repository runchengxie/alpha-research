"""Dynamic signal ensemble orchestration.

This module is a thin public surface for the dynamic signal ensemble
implementation. The historical single-file implementation has been split into
private submodules (``_dynamic_signal_ensemble_metrics`` /
``_dynamic_signal_ensemble_select`` / ``_dynamic_signal_ensemble_run``) plus the
pre-existing ``dynamic_signal_ensemble_*`` helpers, to keep individual files
smaller while preserving the exact public and private symbol surface.
Everything below is re-exported so existing ``alpha_research.dynamic_signal_ensemble``
imports keep working unchanged.
"""

from __future__ import annotations

from ._dynamic_signal_ensemble_metrics import (
    _combine_strength,
    _compute_regime_scores,
    _compute_single_factor_metrics,
    _passes_min,
    compute_factor_metrics,
    compute_rolling_diagnostics,
)
from ._dynamic_signal_ensemble_run import (
    _align_inputs,
    _append_factor_monitor_rows,
    _append_portfolio_monitor_row,
    _build_from_config,
    _prepare_dynamic_ensemble_inputs,
    _run_dynamic_ensemble_rebalance,
    _run_dynamic_ensemble_rebalances,
    add_dynamic_signal_ensemble_args,
    attach_dynamic_ensemble_score,
    build_dynamic_signal_ensemble,
    run,
)
from ._dynamic_signal_ensemble_select import (
    _aggregate_stock_scores,
    _factor_weights,
    _risk_penalty_for_date,
    _rolling_factor_correlation,
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

__all__ = [
    "DynamicSignalEnsembleConfig",
    "DynamicSignalEnsembleResult",
    "FactorMetricBundle",
    "_DynamicEnsembleFrames",
    "_DynamicEnsembleInputs",
    "_DynamicEnsembleRebalance",
    "_aggregate_stock_scores",
    "_align_inputs",
    "_append_factor_monitor_rows",
    "_append_portfolio_monitor_row",
    "_apply_direction_panels",
    "_apply_turnover_budget",
    "_build_dynamic_ensemble_summary",
    "_build_from_config",
    "_cap_positive_weights",
    "_coerce_date_column",
    "_combine_strength",
    "_compute_raw_rank_ic",
    "_compute_regime_scores",
    "_compute_single_factor_metrics",
    "_config_from_mapping",
    "_cross_sectional_zscore_frame",
    "_dynamic_ensemble_frames_from_history",
    "_factor_weights",
    "_load_from_signal_files",
    "_load_regime_features",
    "_load_table",
    "_load_yaml",
    "_normalize_long_frame",
    "_panel_from_long",
    "_passes_min",
    "_pivot_panel",
    "_prepare_dynamic_ensemble_inputs",
    "_resolve_path",
    "_risk_penalty_for_date",
    "_rolling_factor_correlation",
    "_run_dynamic_ensemble_rebalance",
    "_run_dynamic_ensemble_rebalances",
    "_section",
    "_select_factors",
    "_stock_weights",
    "_zscore_series",
    "add_dynamic_signal_ensemble_args",
    "attach_dynamic_ensemble_score",
    "build_dynamic_signal_ensemble",
    "calibrate_signal_directions",
    "compute_factor_metrics",
    "compute_rolling_diagnostics",
    "run",
    "stock_scores_to_long",
    "write_dynamic_ensemble_artifacts",
]
