from __future__ import annotations

from typing import Any


def _nested(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


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
        "baseline_backtest_sharpe": _nested(base, "backtest.sharpe"),
        "candidate_backtest_sharpe": _nested(cand, "backtest.sharpe"),
        "candidate_eval_ic_ir": _nested(cand, "main_eval.eval_ic_ir"),
        "candidate_walk_forward_test_ic_mean": _nested(cand, "walk_forward.test_ic_mean"),
        "candidate_final_oos_ic_mean": _nested(cand, "final_oos.ic_mean"),
        "candidate_final_oos_long_short": _nested(cand, "final_oos.long_short"),
        "candidate_backtest_avg_turnover": _nested(cand, "backtest.avg_turnover"),
        "candidate_backtest_avg_cost_drag": _nested(cand, "backtest.avg_cost_drag"),
        "baseline_benchmark_active_ir": _nested(base, "benchmark.active_information_ratio"),
        "candidate_benchmark_active_ir": _nested(cand, "benchmark.active_information_ratio"),
        "baseline_cpcv_sharpe_median": _nested(base, "cpcv.sharpe_median"),
        "baseline_cpcv_sharpe_p25": _nested(base, "cpcv.sharpe_p25"),
        "candidate_cpcv_path_count": _nested(cand, "cpcv.path_count"),
        "candidate_cpcv_valid_path_count": _nested(cand, "cpcv.valid_path_count"),
        "candidate_cpcv_sharpe_median": _nested(cand, "cpcv.sharpe_median"),
        "candidate_cpcv_sharpe_p25": _nested(cand, "cpcv.sharpe_p25"),
        "candidate_cpcv_sharpe_min": _nested(cand, "cpcv.sharpe_min"),
        "candidate_cpcv_positive_sharpe_ratio": _nested(cand, "cpcv.positive_sharpe_ratio"),
        "candidate_cpcv_ic_median": _nested(cand, "cpcv.ic_median"),
        "candidate_cpcv_long_short_median": _nested(cand, "cpcv.long_short_median"),
        "candidate_cpcv_max_drawdown_p10": _nested(cand, "cpcv.max_drawdown_p10"),
        "candidate_cpcv_turnover_median": _nested(cand, "cpcv.turnover_median"),
        "candidate_cpcv_cost_drag_median": _nested(cand, "cpcv.cost_drag_median"),
        "baseline_dsr": _nested(base, "dsr.dsr"),
        "baseline_dsr_n_trials": _nested(base, "dsr.n_trials"),
        "candidate_dsr": _nested(cand, "dsr.dsr"),
        "candidate_dsr_z": _nested(cand, "dsr.dsr_z"),
        "candidate_dsr_n_trials": _nested(cand, "dsr.n_trials"),
        "candidate_dsr_n_obs": _nested(cand, "dsr.n_obs"),
        "candidate_dsr_selected_candidate": _nested(cand, "dsr.selected_candidate"),
        "candidate_dsr_selected_sharpe": _nested(cand, "dsr.selected_sharpe"),
        "candidate_pbo": _nested(cand, "dsr.pbo"),
        "candidate_dynamic_ensemble_report": _nested(cand, "dynamic_ensemble.path"),
        "candidate_dynamic_ensemble_signal_count": _nested(cand, "dynamic_ensemble.signal_count"),
        "candidate_dynamic_ensemble_avg_active_factor_count": _nested(
            cand, "dynamic_ensemble.avg_active_factor_count"
        ),
        "candidate_dynamic_ensemble_avg_factor_turnover": _nested(
            cand, "dynamic_ensemble.avg_factor_turnover"
        ),
        "candidate_dynamic_ensemble_avg_stock_turnover": _nested(
            cand, "dynamic_ensemble.avg_stock_turnover"
        ),
        "candidate_exposure_screen_status": _nested(cand, "exposure_screen.status"),
        "candidate_exposure_screen_breach_count": _nested(cand, "exposure_screen.breach_count"),
        "candidate_exposure_screen_report": _nested(cand, "exposure_screen.path"),
        "candidate_backend_comparison_status": _nested(cand, "backend_comparison.status"),
        "candidate_backend_comparison_replay_verified": _nested(
            cand, "backend_comparison.replay_verified"
        ),
        "candidate_backend_comparison_overlap_ratio": _nested(
            cand, "backend_comparison.overlap_ratio"
        ),
        "candidate_backend_comparison_prediction_pearson": _nested(
            cand, "backend_comparison.prediction_pearson"
        ),
        "candidate_backend_comparison_prediction_mae": _nested(
            cand, "backend_comparison.prediction_mae"
        ),
    }
    for scope in ("test", "final_oos"):
        for window in ("6m", "1m", "1w"):
            for side, evidence in (("baseline", base), ("candidate", cand)):
                prefix = f"{side}_recency_{scope}_{window}"
                row[f"{prefix}_status"] = _nested(
                    evidence,
                    f"recency_diagnostics.{scope}.{window}.status",
                )
                row[f"{prefix}_total_return"] = _nested(
                    evidence,
                    f"recency_diagnostics.{scope}.{window}.total_return",
                )
                row[f"{prefix}_sharpe"] = _nested(
                    evidence,
                    f"recency_diagnostics.{scope}.{window}.sharpe",
                )
                row[f"{prefix}_ic_mean"] = _nested(
                    evidence,
                    f"recency_diagnostics.{scope}.{window}.ic_mean",
                )
    return row


__all__ = ["flatten_promotion_record"]
