"""Promotion-gate configuration: dataclasses, coercion, and config loading.

Extracted from ``promotion_gate.py`` to separate the self-contained config
concern (no dependency on evidence loading or record building) from the rest of
the promotion-gate module. The public API is re-exported by ``promotion_gate``
so existing ``alpha_research.promotion_gate`` imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROMOTION_STATUSES = ("promotable", "reviewable", "rejected", "non-comparable")

DEFAULT_COMPARABILITY_KEYS = (
    "market",
    "data.provider",
    "universe.mode",
    "universe.by_date_file",
    "label.target_col",
    "label.horizon_days",
    "label.shift_days",
    "label.train_target_transform",
    "features.list",
    "features.exclude",
    "model.type",
    "eval.rebalance_frequency",
    "backtest.rebalance_frequency",
    "backtest.transaction_cost_bps",
    "backtest.execution",
    "data.benchmark_returns_file",
)

DEFAULT_REQUIRED_EVIDENCE = (
    "main_eval",
    "backtest",
    "walk_forward",
    "final_oos",
    "cost_turnover",
)


@dataclass(frozen=True)
class PromotionHardRejections:
    constant_prediction: bool = True
    zero_feature_importance: bool = True
    require_final_oos: bool = True
    min_cv_ic_valid_folds: int = 0
    min_cpcv_path_count: int = 0
    min_dsr_n_trials: int = 0


@dataclass(frozen=True)
class PromotionSoftThresholds:
    min_eval_ic_ir: float | None = 0.0
    min_eval_long_short: float | None = 0.0
    min_walk_forward_test_ic_mean: float | None = 0.0
    min_final_oos_ic_mean: float | None = 0.0
    min_final_oos_long_short: float | None = 0.0
    min_backtest_sharpe_delta: float | None = 0.0
    min_final_oos_sharpe_delta: float | None = None
    max_backtest_drawdown: float | None = 0.30
    max_backtest_avg_turnover: float | None = 0.70
    max_backtest_avg_cost_drag: float | None = 0.02
    min_cpcv_sharpe_median: float | None = None
    min_cpcv_sharpe_p25: float | None = None
    min_cpcv_positive_sharpe_ratio: float | None = None
    min_cpcv_ic_median: float | None = None
    min_cpcv_long_short_median: float | None = None
    max_cpcv_drawdown_p10: float | None = None
    min_cpcv_sharpe_median_delta: float | None = None
    min_cpcv_sharpe_p25_delta: float | None = None
    max_exposure_screen_breach_count: float | None = None
    min_dsr: float | None = None


@dataclass(frozen=True)
class PromotionCPCVConfig:
    baseline_report: Path | None = None
    candidate_report: Path | None = None


@dataclass(frozen=True)
class PromotionDSRConfig:
    baseline_report: Path | None = None
    candidate_report: Path | None = None


@dataclass(frozen=True)
class PromotionDynamicEnsembleConfig:
    baseline_report: Path | None = None
    candidate_report: Path | None = None


@dataclass(frozen=True)
class PromotionGateConfig:
    baseline_run: Path | None = None
    candidate_run: Path | None = None
    benchmark_report: Path | None = None
    baseline_exposure_screen_report: Path | None = None
    candidate_exposure_screen_report: Path | None = None
    comparability_keys: tuple[str, ...] = DEFAULT_COMPARABILITY_KEYS
    required_evidence: tuple[str, ...] = DEFAULT_REQUIRED_EVIDENCE
    hard_rejections: PromotionHardRejections = field(default_factory=PromotionHardRejections)
    soft_thresholds: PromotionSoftThresholds = field(default_factory=PromotionSoftThresholds)
    cpcv: PromotionCPCVConfig = field(default_factory=PromotionCPCVConfig)
    dsr: PromotionDSRConfig = field(default_factory=PromotionDSRConfig)
    dynamic_ensemble: PromotionDynamicEnsembleConfig = field(
        default_factory=PromotionDynamicEnsembleConfig
    )


def _resolve_path(
    path_text: str | Path | None,
    *,
    base_dir: Path | None = None,
) -> Path | None:
    if path_text is None:
        return None
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root = Path.cwd() if base_dir is None else base_dir
    return (root / candidate).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to parse YAML config: {path} ({exc})") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Promotion gate config must be a mapping: {path}")
    return payload


def _coerce_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise SystemExit(f"Missing boolean value: {key}")
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise SystemExit(f"Invalid boolean value for {key}: {value}")


def _coerce_float_or_none(value: Any, *, key: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{key} must be numeric or null.") from exc
    if not np.isfinite(number):
        raise SystemExit(f"{key} must be finite or null.")
    return number


def _coerce_int(value: Any, *, key: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{key} must be an integer.") from exc
    return number


def _coerce_str_tuple(value: Any, *, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        values = [str(item).strip() for item in value]
    else:
        raise SystemExit(f"{key} must be a string or list.")
    cleaned = tuple(item for item in values if item)
    if not cleaned:
        raise SystemExit(f"{key} cannot be empty.")
    return cleaned


def _promotion_gate_payload(
    path_or_payload: str | Path | dict[str, Any],
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(path_or_payload, dict):
        payload = path_or_payload
        base_dir = None
    else:
        path = _resolve_path(path_or_payload)
        assert path is not None
        payload = _load_yaml(path)
        base_dir = path.parent
    gate_payload = payload.get("promotion_gate", payload)
    if not isinstance(gate_payload, dict):
        raise SystemExit("promotion_gate must be a mapping.")
    return gate_payload, base_dir


def _mapping_section(gate_payload: dict[str, Any], name: str) -> dict[str, Any]:
    section = gate_payload.get(name) or {}
    if not isinstance(section, dict):
        raise SystemExit(f"promotion_gate.{name} must be a mapping.")
    return section


def _load_hard_rejections(gate_payload: dict[str, Any]) -> PromotionHardRejections:
    hard_raw = _mapping_section(gate_payload, "hard_rejections")
    default_hard = PromotionHardRejections()
    return PromotionHardRejections(
        constant_prediction=_coerce_bool(
            hard_raw.get("constant_prediction", default_hard.constant_prediction),
            key="hard_rejections.constant_prediction",
        ),
        zero_feature_importance=_coerce_bool(
            hard_raw.get("zero_feature_importance", default_hard.zero_feature_importance),
            key="hard_rejections.zero_feature_importance",
        ),
        require_final_oos=_coerce_bool(
            hard_raw.get("require_final_oos", default_hard.require_final_oos),
            key="hard_rejections.require_final_oos",
        ),
        min_cv_ic_valid_folds=max(
            0,
            _coerce_int(
                hard_raw.get("min_cv_ic_valid_folds", default_hard.min_cv_ic_valid_folds),
                key="hard_rejections.min_cv_ic_valid_folds",
            ),
        ),
        min_cpcv_path_count=max(
            0,
            _coerce_int(
                hard_raw.get("min_cpcv_path_count", default_hard.min_cpcv_path_count),
                key="hard_rejections.min_cpcv_path_count",
            ),
        ),
        min_dsr_n_trials=max(
            0,
            _coerce_int(
                hard_raw.get("min_dsr_n_trials", default_hard.min_dsr_n_trials),
                key="hard_rejections.min_dsr_n_trials",
            ),
        ),
    )


def _load_soft_thresholds(gate_payload: dict[str, Any]) -> PromotionSoftThresholds:
    soft_raw = _mapping_section(gate_payload, "soft_thresholds")
    default_soft = PromotionSoftThresholds()
    return PromotionSoftThresholds(
        **{
            field_name: _coerce_float_or_none(
                soft_raw.get(field_name, getattr(default_soft, field_name)),
                key=f"soft_thresholds.{field_name}",
            )
            for field_name in default_soft.__dataclass_fields__
        }
    )


def _load_cpcv_config(
    gate_payload: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> PromotionCPCVConfig:
    cpcv_raw = _mapping_section(gate_payload, "cpcv")
    return PromotionCPCVConfig(
        baseline_report=_resolve_path(
            cpcv_raw.get("baseline_report", gate_payload.get("baseline_cpcv_report")),
            base_dir=base_dir,
        ),
        candidate_report=_resolve_path(
            cpcv_raw.get("candidate_report", gate_payload.get("candidate_cpcv_report")),
            base_dir=base_dir,
        ),
    )


def _load_dsr_config(
    gate_payload: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> PromotionDSRConfig:
    dsr_raw = _mapping_section(gate_payload, "dsr")
    return PromotionDSRConfig(
        baseline_report=_resolve_path(
            dsr_raw.get("baseline_report", gate_payload.get("baseline_dsr_report")),
            base_dir=base_dir,
        ),
        candidate_report=_resolve_path(
            dsr_raw.get("candidate_report", gate_payload.get("candidate_dsr_report")),
            base_dir=base_dir,
        ),
    )


def _load_dynamic_ensemble_config(
    gate_payload: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> PromotionDynamicEnsembleConfig:
    dynamic_raw = _mapping_section(gate_payload, "dynamic_ensemble")
    return PromotionDynamicEnsembleConfig(
        baseline_report=_resolve_path(
            dynamic_raw.get(
                "baseline_report",
                gate_payload.get("baseline_dynamic_ensemble_report"),
            ),
            base_dir=base_dir,
        ),
        candidate_report=_resolve_path(
            dynamic_raw.get(
                "candidate_report",
                gate_payload.get("candidate_dynamic_ensemble_report"),
            ),
            base_dir=base_dir,
        ),
    )


def load_promotion_gate_config(
    path_or_payload: str | Path | dict[str, Any],
) -> PromotionGateConfig:
    gate_payload, base_dir = _promotion_gate_payload(path_or_payload)
    return PromotionGateConfig(
        baseline_run=_resolve_path(gate_payload.get("baseline_run"), base_dir=base_dir),
        candidate_run=_resolve_path(gate_payload.get("candidate_run"), base_dir=base_dir),
        benchmark_report=_resolve_path(
            gate_payload.get("benchmark_report"),
            base_dir=base_dir,
        ),
        baseline_exposure_screen_report=_resolve_path(
            gate_payload.get("baseline_exposure_screen_report"),
            base_dir=base_dir,
        ),
        candidate_exposure_screen_report=_resolve_path(
            _first_non_empty(
                gate_payload.get("candidate_exposure_screen_report"),
                gate_payload.get("exposure_screen_report"),
            ),
            base_dir=base_dir,
        ),
        comparability_keys=_coerce_str_tuple(
            gate_payload.get("comparability_keys"),
            key="comparability_keys",
            default=DEFAULT_COMPARABILITY_KEYS,
        ),
        required_evidence=_coerce_str_tuple(
            gate_payload.get("required_evidence"),
            key="required_evidence",
            default=DEFAULT_REQUIRED_EVIDENCE,
        ),
        hard_rejections=_load_hard_rejections(gate_payload),
        soft_thresholds=_load_soft_thresholds(gate_payload),
        cpcv=_load_cpcv_config(gate_payload, base_dir=base_dir),
        dsr=_load_dsr_config(gate_payload, base_dir=base_dir),
        dynamic_ensemble=_load_dynamic_ensemble_config(gate_payload, base_dir=base_dir),
    )


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None
