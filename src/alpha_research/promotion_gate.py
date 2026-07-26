"""Promotion-gate orchestration: evidence loading, record building, and CLI.

The configuration dataclasses, coercion helpers, and ``load_promotion_gate_config``
live in :mod:`alpha_research.promotion_gate_config`. This module re-exports the
public config API so that existing ``alpha_research.promotion_gate`` imports keep
working unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

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
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
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


def _cv_valid_folds(summary: dict[str, Any]) -> int | None:
    scores = _get_nested(summary, "eval.cv_ic.scores")
    if isinstance(scores, list):
        return sum(1 for score in scores if _to_float(score) is not None)
    return 1 if _to_float(_get_nested(summary, "eval.cv_ic.mean")) is not None else 0


def _walk_forward_test_ic_mean(summary: dict[str, Any]) -> float | None:
    results = _get_nested(summary, "walk_forward.results")
    if not isinstance(results, list):
        return None
    values: list[float] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if str(item.get("status", "")).lower() != "ok":
            continue
        value = _to_float(_get_nested(item, "test_ic.mean"))
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else None


def _feature_stability(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    path_text = _get_nested(summary, "walk_forward.feature_stability_file")
    candidates: list[Path] = []
    if path_text:
        raw = Path(str(path_text)).expanduser()
        candidates.extend([raw if raw.is_absolute() else (run_dir / raw), (Path.cwd() / raw)])
    candidates.append(run_dir / "walk_forward_feature_stability.csv")
    for path in candidates:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty:
            return {
                "available": False,
                "path": str(path),
                "top_k_hit_rate": None,
                "nonzero_hit_rate": None,
            }
        return {
            "available": True,
            "path": str(path),
            "top_k_hit_rate": _to_float(frame.get("top_k_hit_rate", pd.Series(dtype=float)).max()),
            "nonzero_hit_rate": _to_float(
                frame.get("nonzero_hit_rate", pd.Series(dtype=float)).max()
            ),
        }
    return {"available": False, "path": None, "top_k_hit_rate": None, "nonzero_hit_rate": None}


def _recency_window_rows(summary: dict[str, Any], scope: str) -> dict[str, dict[str, Any]]:
    rows = _get_nested(summary, f"recency_diagnostics.{scope}.rows")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("window")): row
        for row in rows
        if isinstance(row, dict) and row.get("window") is not None
    }


def _recency_window_evidence(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "status": None,
            "role": None,
            "ic_mean": None,
            "total_return": None,
            "sharpe": None,
            "max_drawdown": None,
            "active_total_return": None,
            "avg_turnover": None,
        }
    return {
        "status": row.get("status"),
        "role": row.get("role"),
        "ic_mean": _to_float(row.get("ic_mean")),
        "total_return": _to_float(row.get("total_return")),
        "sharpe": _to_float(row.get("sharpe")),
        "max_drawdown": _to_float(row.get("max_drawdown")),
        "active_total_return": _to_float(row.get("active_total_return")),
        "avg_turnover": _to_float(row.get("avg_turnover")),
    }


def _recency_diagnostics(summary: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in ("test", "final_oos"):
        rows = _recency_window_rows(summary, scope)
        result[scope] = {
            window: _recency_window_evidence(rows.get(window)) for window in ("6m", "1m", "1w")
        }
    return result


def _backtest_stats(summary: dict[str, Any]) -> dict[str, Any]:
    nested = _get_nested(summary, "backtest.stats")
    if isinstance(nested, dict):
        return nested
    flat = _get_nested(summary, "backtest")
    if isinstance(flat, dict) and any(
        key in flat for key in ("sharpe", "max_drawdown", "avg_turnover", "avg_cost_drag")
    ):
        return flat
    position_stats = _get_nested(summary, "position_backtest.stats")
    return position_stats if isinstance(position_stats, dict) else {}


def _summary_benchmark(summary: dict[str, Any]) -> dict[str, Any]:
    benchmark = {
        "active_information_ratio": _to_float(
            _get_nested(summary, "backtest.active.information_ratio")
        ),
        "benchmark_compare_file": _get_nested(
            summary,
            "backtest.benchmark_compare.summary_file",
        ),
        "benchmark_name": _get_nested(summary, "backtest.benchmark.name"),
    }
    if benchmark["active_information_ratio"] is None:
        benchmark["active_information_ratio"] = _to_float(
            _get_nested(summary, "benchmark_primary.information_ratio")
        )
        benchmark["benchmark_name"] = _get_nested(summary, "benchmark_primary.benchmark_name")
    return benchmark


def _evidence(
    run_dir: Path,
    summary: dict[str, Any],
    *,
    cpcv_report: Path | None = None,
    dsr_report: Path | None = None,
    dynamic_ensemble_report: Path | None = None,
    benchmark_report: Path | None = None,
    exposure_screen_report: Path | None = None,
) -> dict[str, Any]:
    bt_stats = _backtest_stats(summary)
    final_bt_stats = _get_nested(summary, "final_oos.backtest.stats") or {}
    benchmark = _summary_benchmark(summary)
    if benchmark_report is not None:
        external_benchmark = _load_benchmark_report(benchmark_report)
        if external_benchmark["active_information_ratio"] is not None:
            benchmark = external_benchmark
    elif benchmark["active_information_ratio"] is None:
        benchmark = _load_benchmark_report(None)
    return {
        "main_eval": {
            "eval_ic_mean": _to_float(_get_nested(summary, "eval.ic.mean")),
            "eval_ic_ir": _to_float(_get_nested(summary, "eval.ic.ir")),
            "eval_long_short": _to_float(_get_nested(summary, "eval.long_short")),
            "cv_ic_valid_folds": _cv_valid_folds(summary),
        },
        "backtest": {
            "sharpe": _to_float(bt_stats.get("sharpe")) if isinstance(bt_stats, dict) else None,
            "max_drawdown": _to_float(bt_stats.get("max_drawdown"))
            if isinstance(bt_stats, dict)
            else None,
            "avg_turnover": _to_float(bt_stats.get("avg_turnover"))
            if isinstance(bt_stats, dict)
            else None,
            "avg_cost_drag": _to_float(bt_stats.get("avg_cost_drag"))
            if isinstance(bt_stats, dict)
            else None,
        },
        "walk_forward": {
            "enabled": bool(_get_nested(summary, "walk_forward.enabled")),
            "test_ic_mean": _walk_forward_test_ic_mean(summary),
            "actual_windows": _get_nested(summary, "walk_forward.actual_windows"),
        },
        "final_oos": {
            "enabled": bool(_get_nested(summary, "final_oos.enabled")),
            "dates": _get_nested(summary, "final_oos.dates"),
            "ic_mean": _to_float(_get_nested(summary, "final_oos.ic.mean")),
            "long_short": _to_float(_get_nested(summary, "final_oos.long_short")),
            "sharpe": _to_float(final_bt_stats.get("sharpe"))
            if isinstance(final_bt_stats, dict)
            else None,
            "avg_turnover": _to_float(final_bt_stats.get("avg_turnover"))
            if isinstance(final_bt_stats, dict)
            else None,
            "avg_cost_drag": _to_float(final_bt_stats.get("avg_cost_drag"))
            if isinstance(final_bt_stats, dict)
            else None,
        },
        "feature_stability": _feature_stability(run_dir, summary),
        "recency_diagnostics": _recency_diagnostics(summary),
        "benchmark": benchmark,
        "cpcv": _load_cpcv_summary(cpcv_report),
        "dsr": _load_dsr_summary(dsr_report),
        "dynamic_ensemble": _load_dynamic_ensemble_report(dynamic_ensemble_report, summary),
        "exposure_screen": _load_exposure_screen_report(exposure_screen_report),
    }


def _is_missing_evidence_category(evidence: dict[str, Any], category: str) -> bool:
    if category == "main_eval":
        return (
            evidence["main_eval"]["eval_ic_ir"] is None
            and evidence["main_eval"]["eval_ic_mean"] is None
        )
    if category == "backtest":
        return evidence["backtest"]["sharpe"] is None
    if category == "walk_forward":
        return evidence["walk_forward"]["test_ic_mean"] is None
    if category == "final_oos":
        return not evidence["final_oos"]["enabled"] or evidence["final_oos"]["ic_mean"] is None
    if category == "cost_turnover":
        return (
            evidence["backtest"]["avg_turnover"] is None
            or evidence["backtest"]["avg_cost_drag"] is None
        )
    if category == "feature_stability":
        return not evidence["feature_stability"]["available"]
    if category == "recency_diagnostics":
        return not any(
            evidence["recency_diagnostics"]["test"][window]["status"] is not None
            for window in ("6m", "1m", "1w")
        )
    if category == "benchmark":
        return evidence["benchmark"]["active_information_ratio"] is None
    if category == "cpcv":
        return not evidence["cpcv"]["available"]
    if category == "dsr":
        return not evidence["dsr"]["available"] or evidence["dsr"]["dsr"] is None
    if category == "dynamic_ensemble":
        return (
            not evidence["dynamic_ensemble"]["available"]
            or evidence["dynamic_ensemble"]["rolling_metrics_shifted"] is not True
            or evidence["dynamic_ensemble"]["no_level2"] is not True
        )
    if category == "exposure_screen":
        return not evidence["exposure_screen"]["available"]
    return False


def _missing_evidence(evidence: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    return [category for category in required if _is_missing_evidence_category(evidence, category)]


def _comparability(
    baseline_config: dict[str, Any],
    candidate_config: dict[str, Any],
    keys: tuple[str, ...],
) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for key in keys:
        left = _get_nested(baseline_config, key)
        right = _get_nested(candidate_config, key)
        if _norm(left) != _norm(right):
            mismatches.append(key)
    return not mismatches, mismatches


def build_promotion_record(config: PromotionGateConfig) -> dict[str, Any]:
    if config.baseline_run is None or config.candidate_run is None:
        raise SystemExit("Promotion gate requires baseline_run and candidate_run.")
    baseline_summary, baseline_config = _load_run(config.baseline_run)
    candidate_summary, candidate_config = _load_run(config.candidate_run)

    comparable, mismatches = _comparability(
        baseline_config,
        candidate_config,
        config.comparability_keys,
    )
    baseline_evidence = _evidence(
        config.baseline_run,
        baseline_summary,
        cpcv_report=config.cpcv.baseline_report,
        dsr_report=config.dsr.baseline_report,
        dynamic_ensemble_report=config.dynamic_ensemble.baseline_report,
        benchmark_report=config.benchmark_report,
        exposure_screen_report=config.baseline_exposure_screen_report,
    )
    candidate_evidence = _evidence(
        config.candidate_run,
        candidate_summary,
        cpcv_report=config.cpcv.candidate_report,
        dsr_report=config.dsr.candidate_report,
        dynamic_ensemble_report=config.dynamic_ensemble.candidate_report,
        benchmark_report=config.benchmark_report,
        exposure_screen_report=config.candidate_exposure_screen_report,
    )
    missing = _missing_evidence(candidate_evidence, config.required_evidence)

    hard_failures: list[str] = []
    hard = config.hard_rejections
    if hard.constant_prediction and _bool(
        _get_nested(candidate_summary, "eval.constant_prediction")
    ):
        hard_failures.append("constant_prediction")
    if hard.zero_feature_importance and _bool(
        _get_nested(candidate_summary, "eval.zero_feature_importance")
    ):
        hard_failures.append("zero_feature_importance")
    if hard.require_final_oos and "final_oos" in missing:
        hard_failures.append("missing_final_oos")
    valid_folds = candidate_evidence["main_eval"]["cv_ic_valid_folds"]
    if hard.min_cv_ic_valid_folds > 0 and (valid_folds or 0) < hard.min_cv_ic_valid_folds:
        hard_failures.append("insufficient_cv_ic_valid_folds")
    if hard.min_cpcv_path_count > 0:
        cpcv_paths = candidate_evidence["cpcv"]["valid_path_count"]
        if cpcv_paths is None or cpcv_paths < hard.min_cpcv_path_count:
            hard_failures.append("insufficient_cpcv_path_count")
    if hard.min_dsr_n_trials > 0:
        dsr_trials = candidate_evidence["dsr"]["n_trials"]
        if dsr_trials is None or dsr_trials < hard.min_dsr_n_trials:
            hard_failures.append("insufficient_dsr_trial_count")

    soft_failures = _soft_failures(baseline_evidence, candidate_evidence, config.soft_thresholds)

    if not comparable:
        status = "non-comparable"
    elif hard_failures or missing:
        status = "rejected"
    elif soft_failures:
        status = "reviewable"
    else:
        status = "promotable"

    return {
        "baseline_run": str(config.baseline_run),
        "candidate_run": str(config.candidate_run),
        "promotion_status": status,
        "is_comparable": comparable,
        "comparability_mismatches": mismatches,
        "missing_evidence": missing,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "config": {
            **asdict(config),
            "baseline_run": str(config.baseline_run),
            "candidate_run": str(config.candidate_run),
        },
        "baseline_evidence": baseline_evidence,
        "candidate_evidence": candidate_evidence,
    }


def flatten_promotion_record(record: dict[str, Any]) -> dict[str, Any]:
    cand = record.get("candidate_evidence") or {}
    base = record.get("baseline_evidence") or {}
    row = {
        "baseline_run": record.get("baseline_run"),
        "candidate_run": record.get("candidate_run"),
        "promotion_status": record.get("promotion_status"),
        "is_comparable": record.get("is_comparable"),
        "comparability_mismatches": "|".join(record.get("comparability_mismatches") or []),
        "missing_evidence": "|".join(record.get("missing_evidence") or []),
        "hard_failures": "|".join(record.get("hard_failures") or []),
        "soft_failures": "|".join(record.get("soft_failures") or []),
        "baseline_backtest_sharpe": _get_nested(base, "backtest.sharpe"),
        "candidate_backtest_sharpe": _get_nested(cand, "backtest.sharpe"),
        "candidate_eval_ic_ir": _get_nested(cand, "main_eval.eval_ic_ir"),
        "candidate_walk_forward_test_ic_mean": _get_nested(cand, "walk_forward.test_ic_mean"),
        "candidate_final_oos_ic_mean": _get_nested(cand, "final_oos.ic_mean"),
        "candidate_final_oos_long_short": _get_nested(cand, "final_oos.long_short"),
        "candidate_backtest_avg_turnover": _get_nested(cand, "backtest.avg_turnover"),
        "candidate_backtest_avg_cost_drag": _get_nested(cand, "backtest.avg_cost_drag"),
        "baseline_benchmark_active_ir": _get_nested(base, "benchmark.active_information_ratio"),
        "candidate_benchmark_active_ir": _get_nested(cand, "benchmark.active_information_ratio"),
        "baseline_cpcv_sharpe_median": _get_nested(base, "cpcv.sharpe_median"),
        "baseline_cpcv_sharpe_p25": _get_nested(base, "cpcv.sharpe_p25"),
        "candidate_cpcv_path_count": _get_nested(cand, "cpcv.path_count"),
        "candidate_cpcv_valid_path_count": _get_nested(cand, "cpcv.valid_path_count"),
        "candidate_cpcv_sharpe_median": _get_nested(cand, "cpcv.sharpe_median"),
        "candidate_cpcv_sharpe_p25": _get_nested(cand, "cpcv.sharpe_p25"),
        "candidate_cpcv_sharpe_min": _get_nested(cand, "cpcv.sharpe_min"),
        "candidate_cpcv_positive_sharpe_ratio": _get_nested(cand, "cpcv.positive_sharpe_ratio"),
        "candidate_cpcv_ic_median": _get_nested(cand, "cpcv.ic_median"),
        "candidate_cpcv_long_short_median": _get_nested(cand, "cpcv.long_short_median"),
        "candidate_cpcv_max_drawdown_p10": _get_nested(cand, "cpcv.max_drawdown_p10"),
        "candidate_cpcv_turnover_median": _get_nested(cand, "cpcv.turnover_median"),
        "candidate_cpcv_cost_drag_median": _get_nested(cand, "cpcv.cost_drag_median"),
        "baseline_dsr": _get_nested(base, "dsr.dsr"),
        "baseline_dsr_n_trials": _get_nested(base, "dsr.n_trials"),
        "candidate_dsr": _get_nested(cand, "dsr.dsr"),
        "candidate_dsr_z": _get_nested(cand, "dsr.dsr_z"),
        "candidate_dsr_n_trials": _get_nested(cand, "dsr.n_trials"),
        "candidate_dsr_n_obs": _get_nested(cand, "dsr.n_obs"),
        "candidate_dsr_selected_candidate": _get_nested(cand, "dsr.selected_candidate"),
        "candidate_dsr_selected_sharpe": _get_nested(cand, "dsr.selected_sharpe"),
        "candidate_pbo": _get_nested(cand, "dsr.pbo"),
        "candidate_dynamic_ensemble_report": _get_nested(cand, "dynamic_ensemble.path"),
        "candidate_dynamic_ensemble_signal_count": _get_nested(
            cand, "dynamic_ensemble.signal_count"
        ),
        "candidate_dynamic_ensemble_avg_active_factor_count": _get_nested(
            cand, "dynamic_ensemble.avg_active_factor_count"
        ),
        "candidate_dynamic_ensemble_avg_factor_turnover": _get_nested(
            cand, "dynamic_ensemble.avg_factor_turnover"
        ),
        "candidate_dynamic_ensemble_avg_stock_turnover": _get_nested(
            cand, "dynamic_ensemble.avg_stock_turnover"
        ),
        "candidate_exposure_screen_status": _get_nested(cand, "exposure_screen.status"),
        "candidate_exposure_screen_breach_count": _get_nested(cand, "exposure_screen.breach_count"),
        "candidate_exposure_screen_report": _get_nested(cand, "exposure_screen.path"),
    }
    for scope in ("test", "final_oos"):
        for window in ("6m", "1m", "1w"):
            for side, evidence in (("baseline", base), ("candidate", cand)):
                prefix = f"{side}_recency_{scope}_{window}"
                row[f"{prefix}_status"] = _get_nested(
                    evidence,
                    f"recency_diagnostics.{scope}.{window}.status",
                )
                row[f"{prefix}_total_return"] = _get_nested(
                    evidence,
                    f"recency_diagnostics.{scope}.{window}.total_return",
                )
                row[f"{prefix}_sharpe"] = _get_nested(
                    evidence,
                    f"recency_diagnostics.{scope}.{window}.sharpe",
                )
                row[f"{prefix}_ic_mean"] = _get_nested(
                    evidence,
                    f"recency_diagnostics.{scope}.{window}.ic_mean",
                )
    return row


def write_promotion_report(
    record: dict[str, Any],
    *,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> None:
    if output_json:
        path = _resolve_path(output_json)
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record, ensure_ascii=True, indent=2, default=str), encoding="utf-8"
        )
    if output_csv:
        path = _resolve_path(output_csv)
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        row = flatten_promotion_record(record)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)


def add_promotion_gate_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Promotion gate YAML config.")
    parser.add_argument("--baseline-run", default=None, help="Override baseline run directory.")
    parser.add_argument("--candidate-run", default=None, help="Override candidate run directory.")
    parser.add_argument(
        "--benchmark-report",
        default=None,
        help="Override benchmark-ladder JSON report.",
    )
    parser.add_argument(
        "--baseline-cpcv-report",
        default=None,
        help="Override baseline CPCV summary.",
    )
    parser.add_argument(
        "--candidate-cpcv-report",
        default=None,
        help="Override candidate CPCV summary.",
    )
    parser.add_argument(
        "--baseline-dsr-report",
        default=None,
        help="Override baseline DSR/PBO summary.",
    )
    parser.add_argument(
        "--candidate-dsr-report",
        default=None,
        help="Override candidate DSR/PBO summary.",
    )
    parser.add_argument(
        "--baseline-dynamic-ensemble-report",
        default=None,
        help="Override baseline dynamic signal ensemble summary.",
    )
    parser.add_argument(
        "--candidate-dynamic-ensemble-report",
        default=None,
        help="Override candidate dynamic signal ensemble summary.",
    )
    parser.add_argument(
        "--baseline-exposure-screen-report",
        default=None,
        help="Override baseline exposure screen JSON report.",
    )
    parser.add_argument(
        "--candidate-exposure-screen-report",
        default=None,
        help="Override candidate exposure screen JSON report.",
    )
    parser.add_argument("--output-json", default=None, help="Output JSON report path.")
    parser.add_argument("--output-csv", default=None, help="Output CSV report path.")
    return parser


def run(args: argparse.Namespace) -> int:
    cfg = load_promotion_gate_config(args.config)
    payload = asdict(cfg)
    if args.baseline_run:
        payload["baseline_run"] = args.baseline_run
    if args.candidate_run:
        payload["candidate_run"] = args.candidate_run
    if args.benchmark_report:
        payload["benchmark_report"] = args.benchmark_report
    cpcv_payload = payload.setdefault("cpcv", {})
    if args.baseline_cpcv_report:
        cpcv_payload["baseline_report"] = args.baseline_cpcv_report
    if args.candidate_cpcv_report:
        cpcv_payload["candidate_report"] = args.candidate_cpcv_report
    dsr_payload = payload.setdefault("dsr", {})
    if args.baseline_dsr_report:
        dsr_payload["baseline_report"] = args.baseline_dsr_report
    if args.candidate_dsr_report:
        dsr_payload["candidate_report"] = args.candidate_dsr_report
    dynamic_payload = payload.setdefault("dynamic_ensemble", {})
    if args.baseline_dynamic_ensemble_report:
        dynamic_payload["baseline_report"] = args.baseline_dynamic_ensemble_report
    if args.candidate_dynamic_ensemble_report:
        dynamic_payload["candidate_report"] = args.candidate_dynamic_ensemble_report
    if args.baseline_exposure_screen_report:
        payload["baseline_exposure_screen_report"] = args.baseline_exposure_screen_report
    if args.candidate_exposure_screen_report:
        payload["candidate_exposure_screen_report"] = args.candidate_exposure_screen_report
    cfg = load_promotion_gate_config(payload)
    record = build_promotion_record(cfg)
    write_promotion_report(record, output_json=args.output_json, output_csv=args.output_csv)
    if not args.output_json and not args.output_csv:
        print(json.dumps(record, ensure_ascii=True, indent=2, default=str))
    return 0


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
    "add_promotion_gate_args",
    "build_promotion_record",
    "flatten_promotion_record",
    "load_promotion_gate_config",
    "run",
    "write_promotion_report",
]
