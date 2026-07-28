"""Promotion-gate record building, reporting, and CLI.

Public entry points ``build_promotion_record`` / ``flatten_promotion_record`` /
``write_promotion_report`` plus the ``add_promotion_gate_args`` / ``run`` CLI
helpers. Split out of the historical single-file
:mod:`alpha_research.promotion_gate` implementation to keep individual files
smaller while preserving the exact public/private symbol surface.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ._promotion_gate_evidence import (
    _comparability,
    _evidence,
    _missing_evidence,
)
from ._promotion_gate_loaders import _bool, _get_nested, _load_run
from .promotion_gate_config import (
    PromotionGateConfig,
    _resolve_path,
    load_promotion_gate_config,
)
from .promotion_gate_thresholds import soft_failures as _soft_failures


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
