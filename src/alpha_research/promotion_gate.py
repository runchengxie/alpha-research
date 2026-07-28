"""Promotion-gate orchestration: evidence loading, record building, and CLI.

This module is a thin public surface for the promotion-gate implementation.
The historical single-file implementation has been split into private
submodules (``_promotion_gate_loaders`` / ``_promotion_gate_evidence`` /
``_promotion_gate_record``) plus the configuration helpers in
:mod:`alpha_research.promotion_gate_config` and threshold helpers in
:mod:`alpha_research.promotion_gate_thresholds`, to keep individual files
smaller while preserving the exact public and private symbol surface.
Everything below is re-exported so existing ``alpha_research.promotion_gate``
imports keep working unchanged.
"""

from __future__ import annotations

from ._promotion_gate_evidence import (
    _backtest_stats,
    _comparability,
    _cv_valid_folds,
    _evidence,
    _feature_stability,
    _is_missing_evidence_category,
    _missing_evidence,
    _recency_diagnostics,
    _recency_window_evidence,
    _recency_window_rows,
    _summary_benchmark,
    _walk_forward_test_ic_mean,
)
from ._promotion_gate_loaders import (
    _bool,
    _empty_cpcv_summary,
    _empty_dsr_summary,
    _empty_dynamic_ensemble_summary,
    _get_nested,
    _load_benchmark_report,
    _load_cpcv_summary,
    _load_dsr_summary,
    _load_dynamic_ensemble_report,
    _load_exposure_screen_report,
    _load_json,
    _load_run,
    _norm,
    _to_float,
)
from ._promotion_gate_record import (
    add_promotion_gate_args,
    build_promotion_record,
    flatten_promotion_record,
    run,
    write_promotion_report,
)
from .promotion_gate_config import (
    DEFAULT_COMPARABILITY_KEYS,
    DEFAULT_REQUIRED_EVIDENCE,
    PROMOTION_STATUSES,
    PromotionCPCVConfig,
    PromotionDSRConfig,
    PromotionDynamicEnsembleConfig,
    PromotionGateConfig,
    PromotionHardRejections,
    PromotionSoftThresholds,
    _first_non_empty,
    _resolve_path,
    load_promotion_gate_config,
)
from .promotion_gate_thresholds import soft_failures as _soft_failures

__all__ = [
    "DEFAULT_COMPARABILITY_KEYS",
    "DEFAULT_REQUIRED_EVIDENCE",
    "PROMOTION_STATUSES",
    "PromotionCPCVConfig",
    "PromotionDSRConfig",
    "PromotionDynamicEnsembleConfig",
    "PromotionGateConfig",
    "PromotionHardRejections",
    "PromotionSoftThresholds",
    "_backtest_stats",
    "_bool",
    "_comparability",
    "_cv_valid_folds",
    "_empty_cpcv_summary",
    "_empty_dsr_summary",
    "_empty_dynamic_ensemble_summary",
    "_evidence",
    "_feature_stability",
    "_first_non_empty",
    "_get_nested",
    "_is_missing_evidence_category",
    "_load_benchmark_report",
    "_load_cpcv_summary",
    "_load_dsr_summary",
    "_load_dynamic_ensemble_report",
    "_load_exposure_screen_report",
    "_load_json",
    "_load_run",
    "_missing_evidence",
    "_norm",
    "_recency_diagnostics",
    "_recency_window_evidence",
    "_recency_window_rows",
    "_resolve_path",
    "_soft_failures",
    "_summary_benchmark",
    "_to_float",
    "_walk_forward_test_ic_mean",
    "add_promotion_gate_args",
    "build_promotion_record",
    "flatten_promotion_record",
    "load_promotion_gate_config",
    "run",
    "write_promotion_report",
]
