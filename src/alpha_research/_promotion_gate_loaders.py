"""Promotion-gate evidence loaders and scalar helpers.

Private helpers for reading the various on-disk reports (CPCV, DSR/PBO,
benchmark ladder, exposure screen, dynamic ensemble) that feed the promotion
gate, plus small scalar coercion helpers. Split out of the historical
single-file :mod:`alpha_research.promotion_gate` implementation to keep
individual files smaller while preserving the exact public/private symbol
surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from .promotion_gate_config import _first_non_empty, _resolve_path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _load_json(run_dir / "summary.json")
    config_path = run_dir / "config.used.yml"
    config = {}
    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = payload if isinstance(payload, dict) else {}
    return summary, config


def _empty_cpcv_summary(path: Path | None) -> dict[str, Any]:
    return {
        "available": False,
        "path": str(path) if path else None,
        "path_count": None,
        "valid_path_count": None,
        "sharpe_median": None,
        "sharpe_p25": None,
        "sharpe_min": None,
        "positive_sharpe_ratio": None,
        "ic_median": None,
        "long_short_median": None,
        "max_drawdown_p10": None,
        "turnover_median": None,
        "cost_drag_median": None,
    }


def _load_cpcv_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _empty_cpcv_summary(None)
    payload = _load_json(path)
    if not payload:
        return _empty_cpcv_summary(path)
    return {
        "available": True,
        "path": str(path),
        "path_count": _to_float(payload.get("path_count")),
        "valid_path_count": _to_float(payload.get("valid_path_count")),
        "sharpe_median": _to_float(payload.get("sharpe_median")),
        "sharpe_p25": _to_float(payload.get("sharpe_p25")),
        "sharpe_min": _to_float(payload.get("sharpe_min")),
        "positive_sharpe_ratio": _to_float(payload.get("positive_sharpe_ratio")),
        "ic_median": _to_float(payload.get("ic_median")),
        "long_short_median": _to_float(payload.get("long_short_median")),
        "max_drawdown_p10": _to_float(payload.get("max_drawdown_p10")),
        "turnover_median": _to_float(payload.get("turnover_median")),
        "cost_drag_median": _to_float(payload.get("cost_drag_median")),
    }


def _empty_dsr_summary(path: Path | None) -> dict[str, Any]:
    return {
        "available": False,
        "path": str(path) if path else None,
        "dsr": None,
        "dsr_z": None,
        "n_trials": None,
        "n_obs": None,
        "selected_candidate": None,
        "selected_sharpe": None,
        "pbo": None,
    }


def _load_dsr_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _empty_dsr_summary(None)
    payload = _load_json(path)
    if not payload:
        return _empty_dsr_summary(path)
    summary = cast(dict[str, Any], payload.get("summary") if isinstance(payload.get("summary"), dict) else payload)
    return {
        "available": True,
        "path": str(path),
        "dsr": _to_float(summary.get("dsr")),
        "dsr_z": _to_float(summary.get("dsr_z")),
        "n_trials": _to_float(
            _first_non_empty(
                summary.get("dsr_n_trials"),
                summary.get("n_trials"),
                summary.get("candidate_count"),
            )
        ),
        "n_obs": _to_float(_first_non_empty(summary.get("dsr_n_obs"), summary.get("n_obs"))),
        "selected_candidate": summary.get("selected_candidate"),
        "selected_sharpe": _to_float(summary.get("selected_sharpe")),
        "pbo": _to_float(summary.get("pbo")),
    }


def _load_benchmark_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "active_information_ratio": None,
            "benchmark_compare_file": None,
            "benchmark_name": None,
        }
    if not path.exists():
        payload: Any = {}
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            rows = [row for row in payload["rows"] if isinstance(row, dict)]
        elif isinstance(payload.get("benchmarks"), list):
            rows = [row for row in payload["benchmarks"] if isinstance(row, dict)]
    primary = next((row for row in rows if row.get("role") == "primary"), None)
    selected = primary or next((row for row in rows if row.get("status") == "ok"), None)
    if selected is None:
        return {
            "active_information_ratio": None,
            "benchmark_compare_file": str(path),
            "benchmark_name": None,
        }
    return {
        "active_information_ratio": _to_float(selected.get("information_ratio")),
        "benchmark_compare_file": str(path),
        "benchmark_name": selected.get("benchmark_name"),
    }


def _load_exposure_screen_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "available": False,
            "path": None,
            "status": None,
            "breach_count": None,
            "rows_checked": None,
        }
    if not path.exists():
        return {
            "available": False,
            "path": str(path),
            "status": None,
            "breach_count": None,
            "rows_checked": None,
        }
    payload = _load_json(path)
    if not payload:
        return {
            "available": False,
            "path": str(path),
            "status": None,
            "breach_count": None,
            "rows_checked": None,
        }
    return {
        "available": True,
        "path": str(path),
        "status": payload.get("status"),
        "breach_count": _to_float(payload.get("breach_count")),
        "rows_checked": _to_float(payload.get("rows_checked")),
    }


def _empty_dynamic_ensemble_summary(path: Path | None) -> dict[str, Any]:
    return {
        "available": False,
        "path": str(path) if path else None,
        "rolling_metrics_shifted": None,
        "no_level2": None,
        "signal_count": None,
        "stock_score_dates": None,
        "avg_active_factor_count": None,
        "avg_factor_turnover": None,
        "avg_stock_turnover": None,
        "risk_penalty_enabled": None,
        "correlation_threshold": None,
    }


def _load_dynamic_ensemble_report(path: Path | None, summary: dict[str, Any]) -> dict[str, Any]:
    report_path = path
    if report_path is None:
        summary_file = _get_nested(summary, "dynamic_signal_ensemble.summary_file")
        if summary_file:
            report_path = _resolve_path(summary_file)
    if report_path is None or not report_path.exists():
        return _empty_dynamic_ensemble_summary(report_path)
    payload = _load_json(report_path)
    if not payload:
        return _empty_dynamic_ensemble_summary(report_path)
    return {
        "available": True,
        "path": str(report_path),
        "rolling_metrics_shifted": payload.get("rolling_metrics_shifted"),
        "no_level2": payload.get("no_level2"),
        "signal_count": _to_float(payload.get("signal_count")),
        "stock_score_dates": _to_float(payload.get("stock_score_dates")),
        "avg_active_factor_count": _to_float(payload.get("avg_active_factor_count")),
        "avg_factor_turnover": _to_float(payload.get("avg_factor_turnover")),
        "avg_stock_turnover": _to_float(payload.get("avg_stock_turnover")),
        "risk_penalty_enabled": payload.get("risk_penalty_enabled"),
        "correlation_threshold": _to_float(payload.get("correlation_threshold")),
    }


def _get_nested(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
