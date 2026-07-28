"""Promotion-gate evidence extraction helpers.

Private helpers that turn a run's summary/config into the normalized
``evidence`` dict consumed by the promotion-record builder, plus the
missing-evidence and comparability checks. Split out of the historical
single-file :mod:`alpha_research.promotion_gate` implementation to keep
individual files smaller while preserving the exact public/private symbol
surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._promotion_gate_loaders import (
    _get_nested,
    _load_benchmark_report,
    _load_cpcv_summary,
    _load_dsr_summary,
    _load_dynamic_ensemble_report,
    _load_exposure_screen_report,
    _norm,
    _to_float,
)


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
