import json

import yaml

from cstree.alpha import promotion_gate


def _summary(*, sharpe=1.0, wf=0.03, final_ic=0.02, constant=False):
    return {
        "eval": {
            "ic": {"mean": 0.02, "ir": 0.4},
            "long_short": 0.01,
            "cv_ic": {"scores": [0.01, 0.02, None]},
            "constant_prediction": constant,
            "zero_feature_importance": False,
        },
        "backtest": {
            "stats": {
                "sharpe": sharpe,
                "max_drawdown": -0.10,
                "avg_turnover": 0.20,
                "avg_cost_drag": 0.002,
            }
        },
        "walk_forward": {
            "enabled": True,
            "results": [{"status": "ok", "test_ic": {"mean": wf}}],
        },
        "final_oos": {
            "enabled": True,
            "dates": ["2025-01-01", "2025-03-31"],
            "ic": {"mean": final_ic},
            "long_short": 0.01,
            "backtest": {"stats": {"sharpe": 0.8, "avg_turnover": 0.2, "avg_cost_drag": 0.002}},
        },
        "recency_diagnostics": {
            "test": {
                "rows": [
                    {
                        "window": "6m",
                        "role": "current_effectiveness",
                        "status": "ok",
                        "ic_mean": 0.03,
                        "total_return": 0.06,
                        "sharpe": 1.1,
                        "max_drawdown": -0.03,
                        "active_total_return": 0.02,
                        "avg_turnover": 0.18,
                    },
                    {
                        "window": "1m",
                        "role": "watch_signal",
                        "status": "ok",
                        "ic_mean": 0.02,
                        "total_return": 0.015,
                        "sharpe": 0.7,
                    },
                    {
                        "window": "1w",
                        "role": "monitoring_only",
                        "status": "limited_sample",
                        "ic_mean": 0.01,
                        "total_return": 0.004,
                    },
                ]
            },
            "final_oos": {
                "rows": [
                    {
                        "window": "6m",
                        "role": "current_effectiveness",
                        "status": "ok",
                        "ic_mean": final_ic,
                        "total_return": 0.04,
                        "sharpe": 0.8,
                    }
                ]
            },
        },
    }


def _summary_with_builtin_benchmark_ir(ir):
    payload = _summary()
    payload["backtest"]["active"] = {"information_ratio": ir}
    payload["backtest"]["benchmark"] = {"name": "builtin"}
    return payload


def _write_run(path, summary, *, horizon=20):
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(summary, ensure_ascii=True), encoding="utf-8")
    (path / "config.used.yml").write_text(
        yaml.safe_dump({"label": {"horizon_days": horizon}}, sort_keys=False),
        encoding="utf-8",
    )


def test_promotion_gate_marks_promotable_candidate(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))

    cfg = promotion_gate.load_promotion_gate_config(
        {
            "baseline_run": str(baseline),
            "candidate_run": str(candidate),
            "comparability_keys": ["label.horizon_days"],
        }
    )
    record = promotion_gate.build_promotion_record(cfg)

    assert record["promotion_status"] == "promotable"
    assert record["is_comparable"] is True
    flat = promotion_gate.flatten_promotion_record(record)
    assert flat["candidate_backtest_sharpe"] == 1.2
    assert flat["candidate_recency_test_6m_total_return"] == 0.06
    assert flat["candidate_recency_test_1w_status"] == "limited_sample"


def test_promotion_gate_accepts_flat_materialized_backtest_summary(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, _summary(sharpe=1.0))
    materialized = _summary(sharpe=1.2)
    materialized["backtest"] = {
        "sharpe": 1.2,
        "max_drawdown": -0.12,
        "avg_turnover": 0.25,
        "avg_cost_drag": 0.003,
    }
    materialized["benchmark_primary"] = {
        "benchmark_name": "csi300",
        "information_ratio": 0.7,
    }
    _write_run(candidate, materialized)

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "required_evidence": ["backtest", "benchmark", "cost_turnover"],
            }
        )
    )

    assert record["promotion_status"] == "promotable"
    flat = promotion_gate.flatten_promotion_record(record)
    assert flat["candidate_backtest_sharpe"] == 1.2
    assert flat["candidate_benchmark_active_ir"] == 0.7


def test_promotion_gate_config_accepts_tuple_values_after_dataclass_roundtrip(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))
    cfg = promotion_gate.load_promotion_gate_config(
        {
            "baseline_run": str(baseline),
            "candidate_run": str(candidate),
            "comparability_keys": ["label.horizon_days"],
            "required_evidence": ["main_eval", "backtest"],
        }
    )

    roundtripped = promotion_gate.load_promotion_gate_config(
        {
            "baseline_run": str(cfg.baseline_run),
            "candidate_run": str(cfg.candidate_run),
            "comparability_keys": cfg.comparability_keys,
            "required_evidence": cfg.required_evidence,
        }
    )

    assert roundtripped.comparability_keys == ("label.horizon_days",)
    assert roundtripped.required_evidence == ("main_eval", "backtest")


def test_promotion_gate_rejects_hard_failure(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2, constant=True))

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
            }
        )
    )

    assert record["promotion_status"] == "rejected"
    assert record["hard_failures"] == ["constant_prediction"]


def test_promotion_gate_distinguishes_reviewable_and_non_comparable(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    other = tmp_path / "other"
    _write_run(baseline, _summary(sharpe=1.0), horizon=20)
    _write_run(candidate, _summary(sharpe=1.2), horizon=20)
    _write_run(other, _summary(sharpe=1.2), horizon=60)

    reviewable = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "soft_thresholds": {"min_backtest_sharpe_delta": 0.5},
            }
        )
    )
    non_comparable = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(other),
                "comparability_keys": ["label.horizon_days"],
            }
        )
    )

    assert reviewable["promotion_status"] == "reviewable"
    assert reviewable["soft_failures"] == ["min_backtest_sharpe_delta"]
    assert non_comparable["promotion_status"] == "non-comparable"
    assert non_comparable["comparability_mismatches"] == ["label.horizon_days"]


def _write_cpcv(path, *, sharpe_median=0.5, sharpe_p25=0.2, valid_path_count=7):
    path.write_text(
        json.dumps(
            {
                "path_count": 7,
                "valid_path_count": valid_path_count,
                "sharpe_median": sharpe_median,
                "sharpe_p25": sharpe_p25,
                "sharpe_min": -0.1,
                "positive_sharpe_ratio": 0.8,
                "ic_median": 0.02,
                "long_short_median": 0.01,
                "max_drawdown_p10": 0.2,
                "turnover_median": 0.3,
                "cost_drag_median": 0.002,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _write_benchmark_ladder(path):
    path.write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "a_share_all_equalw",
                    "role": "primary",
                    "status": "ok",
                    "information_ratio": 0.2,
                }
            ],
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _write_exposure_screen(path, *, status="passed", breach_count=0):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "breach_count": breach_count,
                "rows_checked": 1,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _write_dsr(path, *, dsr=0.85, n_trials=12):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dsr": dsr,
                "dsr_z": 1.1,
                "dsr_n_trials": n_trials,
                "dsr_n_obs": 100,
                "selected_candidate": "candidate_a",
                "selected_sharpe": 1.2,
                "pbo": 0.2,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _write_dynamic_ensemble(path, *, shifted=True, no_level2=True):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "cstree.dynamic_signal_ensemble",
                "rolling_metrics_shifted": shifted,
                "no_level2": no_level2,
                "signal_count": 3,
                "stock_score_dates": 10,
                "avg_active_factor_count": 2.0,
                "avg_factor_turnover": 0.25,
                "avg_stock_turnover": 0.40,
                "risk_penalty_enabled": True,
                "correlation_threshold": 0.75,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def test_promotion_gate_rejects_missing_required_cpcv(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "required_evidence": ["main_eval", "backtest", "cpcv"],
            }
        )
    )

    assert record["promotion_status"] == "rejected"
    assert "cpcv" in record["missing_evidence"]


def test_promotion_gate_rejects_missing_cpcv_when_min_path_count_is_required(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "required_evidence": ["main_eval", "backtest", "cpcv"],
                "hard_rejections": {"min_cpcv_path_count": 5},
            }
        )
    )

    assert record["promotion_status"] == "rejected"
    assert "cpcv" in record["missing_evidence"]
    assert "insufficient_cpcv_path_count" in record["hard_failures"]


def test_promotion_gate_accepts_external_benchmark_ladder(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    benchmark_path = tmp_path / "benchmark_ladder.json"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))
    _write_benchmark_ladder(benchmark_path)

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "required_evidence": ["main_eval", "backtest", "benchmark"],
                "benchmark_report": str(benchmark_path),
            }
        )
    )

    assert record["promotion_status"] == "promotable"
    assert record["missing_evidence"] == []
    assert record["candidate_evidence"]["benchmark"]["active_information_ratio"] == 0.2


def test_promotion_gate_prefers_external_benchmark_ladder_over_summary_ir(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    benchmark_path = tmp_path / "benchmark_ladder.json"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary_with_builtin_benchmark_ir(9.9))
    _write_benchmark_ladder(benchmark_path)

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "required_evidence": ["main_eval", "backtest", "benchmark"],
                "benchmark_report": str(benchmark_path),
            }
        )
    )

    benchmark = record["candidate_evidence"]["benchmark"]
    assert benchmark["active_information_ratio"] == 0.2
    assert benchmark["benchmark_name"] == "a_share_all_equalw"


def test_promotion_gate_accepts_dynamic_ensemble_evidence(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    dynamic_path = tmp_path / "dynamic_signal_ensemble_summary.json"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))
    _write_dynamic_ensemble(dynamic_path)

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "required_evidence": ["main_eval", "backtest", "dynamic_ensemble"],
                "dynamic_ensemble": {"candidate_report": str(dynamic_path)},
            }
        )
    )

    assert record["promotion_status"] == "promotable"
    assert record["candidate_evidence"]["dynamic_ensemble"]["avg_active_factor_count"] == 2.0
    flat = promotion_gate.flatten_promotion_record(record)
    assert flat["candidate_dynamic_ensemble_signal_count"] == 3.0


def test_promotion_gate_rejects_unshifted_dynamic_ensemble_evidence(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    dynamic_path = tmp_path / "dynamic_signal_ensemble_summary.json"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))
    _write_dynamic_ensemble(dynamic_path, shifted=False)

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "required_evidence": ["main_eval", "backtest", "dynamic_ensemble"],
                "candidate_dynamic_ensemble_report": str(dynamic_path),
            }
        )
    )

    assert record["promotion_status"] == "rejected"
    assert "dynamic_ensemble" in record["missing_evidence"]


def test_promotion_gate_accepts_external_exposure_screen(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    exposure_path = tmp_path / "exposure_screen.json"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))
    _write_exposure_screen(exposure_path)

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "required_evidence": ["main_eval", "backtest", "exposure_screen"],
                "candidate_exposure_screen_report": str(exposure_path),
                "soft_thresholds": {"max_exposure_screen_breach_count": 0},
            }
        )
    )

    flat = promotion_gate.flatten_promotion_record(record)
    assert record["promotion_status"] == "promotable"
    assert record["missing_evidence"] == []
    assert flat["candidate_exposure_screen_status"] == "passed"
    assert flat["candidate_exposure_screen_breach_count"] == 0.0


def test_promotion_gate_flags_exposure_screen_breaches_as_reviewable(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    exposure_path = tmp_path / "exposure_screen.json"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))
    _write_exposure_screen(exposure_path, status="breached", breach_count=2)

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "required_evidence": ["main_eval", "backtest", "exposure_screen"],
                "exposure_screen_report": str(exposure_path),
                "soft_thresholds": {"max_exposure_screen_breach_count": 0},
            }
        )
    )

    assert record["promotion_status"] == "reviewable"
    assert record["candidate_evidence"]["exposure_screen"]["breach_count"] == 2.0
    assert record["soft_failures"] == ["max_exposure_screen_breach_count"]


def test_promotion_gate_rejects_missing_required_exposure_screen(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "required_evidence": ["main_eval", "backtest", "exposure_screen"],
            }
        )
    )

    assert record["promotion_status"] == "rejected"
    assert "exposure_screen" in record["missing_evidence"]


def test_promotion_gate_accepts_and_flattens_cpcv_evidence(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    cpcv_path = tmp_path / "candidate_cpcv_summary.json"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))
    _write_cpcv(cpcv_path)

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "required_evidence": ["main_eval", "backtest", "cpcv"],
                "cpcv": {"candidate_report": str(cpcv_path)},
                "hard_rejections": {"min_cpcv_path_count": 5},
                "soft_thresholds": {"min_cpcv_sharpe_median": 0.0, "min_cpcv_sharpe_p25": 0.0},
            }
        )
    )

    flat = promotion_gate.flatten_promotion_record(record)
    assert record["promotion_status"] == "promotable"
    assert flat["candidate_cpcv_valid_path_count"] == 7.0
    assert flat["candidate_cpcv_sharpe_median"] == 0.5


def test_promotion_gate_cpcv_threshold_failures(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    cpcv_path = tmp_path / "candidate_cpcv_summary.json"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))
    _write_cpcv(cpcv_path, sharpe_median=-0.1, valid_path_count=3)

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "cpcv": {"candidate_report": str(cpcv_path)},
                "hard_rejections": {"min_cpcv_path_count": 5},
                "soft_thresholds": {"min_cpcv_sharpe_median": 0.0},
            }
        )
    )

    assert record["promotion_status"] == "rejected"
    assert "insufficient_cpcv_path_count" in record["hard_failures"]
    assert "min_cpcv_sharpe_median" in record["soft_failures"]


def test_promotion_gate_cpcv_baseline_relative_threshold(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline_cpcv = tmp_path / "baseline_cpcv_summary.json"
    candidate_cpcv = tmp_path / "candidate_cpcv_summary.json"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))
    _write_cpcv(baseline_cpcv, sharpe_median=0.6, sharpe_p25=0.4)
    _write_cpcv(candidate_cpcv, sharpe_median=0.55, sharpe_p25=0.3)

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "cpcv": {
                    "baseline_report": str(baseline_cpcv),
                    "candidate_report": str(candidate_cpcv),
                },
                "soft_thresholds": {"min_cpcv_sharpe_median_delta": 0.0},
            }
        )
    )

    assert record["promotion_status"] == "reviewable"
    assert record["soft_failures"] == ["min_cpcv_sharpe_median_delta"]


def test_promotion_gate_accepts_and_flattens_dsr_evidence(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    dsr_path = tmp_path / "candidate_dsr.json"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))
    _write_dsr(dsr_path, dsr=0.91, n_trials=10)

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "required_evidence": ["main_eval", "backtest", "dsr"],
                "dsr": {"candidate_report": str(dsr_path)},
                "hard_rejections": {"min_dsr_n_trials": 5},
                "soft_thresholds": {"min_dsr": 0.8},
            }
        )
    )

    flat = promotion_gate.flatten_promotion_record(record)
    assert record["promotion_status"] == "promotable"
    assert flat["candidate_dsr"] == 0.91
    assert flat["candidate_dsr_n_trials"] == 10.0
    assert flat["candidate_pbo"] == 0.2


def test_promotion_gate_dsr_threshold_failures(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    dsr_path = tmp_path / "candidate_dsr.json"
    _write_run(baseline, _summary(sharpe=1.0))
    _write_run(candidate, _summary(sharpe=1.2))
    _write_dsr(dsr_path, dsr=0.4, n_trials=2)

    record = promotion_gate.build_promotion_record(
        promotion_gate.load_promotion_gate_config(
            {
                "baseline_run": str(baseline),
                "candidate_run": str(candidate),
                "comparability_keys": ["label.horizon_days"],
                "dsr": {"candidate_report": str(dsr_path)},
                "hard_rejections": {"min_dsr_n_trials": 5},
                "soft_thresholds": {"min_dsr": 0.8},
            }
        )
    )

    assert record["promotion_status"] == "rejected"
    assert "insufficient_dsr_trial_count" in record["hard_failures"]
    assert "min_dsr" in record["soft_failures"]
