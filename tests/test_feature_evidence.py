import json

import pandas as pd
import pytest
import yaml

from cstree.alpha import feature_evidence


def test_generate_ablation_jobs_writes_baseline_and_minus_configs(tmp_path):
    base_config = {
        "features": {"list": ["f1", "f2", "f3"]},
        "eval": {"run_name": "base", "output_dir": str(tmp_path / "runs")},
    }
    base_path = tmp_path / "base.yml"
    base_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")

    cfg = {
        "feature_evidence": {
            "base_config": str(base_path),
            "output_dir": str(tmp_path / "ablation"),
            "run_name_prefix": "feat_",
            "families": {"value": ["f1", "f2"], "quality": ["f3"]},
        }
    }

    result = feature_evidence.generate_ablation_jobs(cfg, config_dir=tmp_path)

    jobs = pd.read_csv(result["jobs_csv"])
    assert set(jobs["family"]) == {"baseline", "minus_value", "minus_quality"}
    minus_value = yaml.safe_load(
        (tmp_path / "ablation" / "configs" / "minus_value.yml").read_text(encoding="utf-8")
    )
    assert minus_value["features"]["list"] == ["f3"]
    assert minus_value["metadata"]["feature_ablation"]["removed_features"] == ["f1", "f2"]


def _write_summary(run_dir, *, sharpe, ic_ir, wf):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "eval": {
            "ic": {"ir": ic_ir},
            "long_short": 0.02,
            "constant_prediction": False,
            "zero_feature_importance": False,
        },
        "walk_forward": {
            "results": [{"status": "ok", "test_ic": {"mean": wf}}],
            "feature_stability_file": "walk_forward_feature_stability.csv",
        },
        "final_oos": {"ic": {"mean": 0.01}},
        "backtest": {
            "stats": {
                "sharpe": sharpe,
                "max_drawdown": -0.1,
                "avg_turnover": 0.2,
                "avg_cost_drag": 0.001,
            },
            "active": {"information_ratio": 0.5},
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    pd.DataFrame(
        {
            "feature": ["f1", "f2"],
            "top_k_hit_rate": [0.8, 0.4],
            "nonzero_hit_rate": [1.0, 0.5],
        }
    ).to_csv(run_dir / "walk_forward_feature_stability.csv", index=False)


def test_summarize_ablation_results_adds_deltas_and_stability(tmp_path):
    baseline = tmp_path / "baseline"
    minus = tmp_path / "minus_value"
    _write_summary(baseline, sharpe=1.0, ic_ir=0.4, wf=0.03)
    _write_summary(minus, sharpe=0.7, ic_ir=0.2, wf=0.01)

    cfg = {
        "feature_evidence": {
            "runs": [
                {"family": "baseline", "run_dir": str(baseline)},
                {"family": "minus_value", "run_dir": str(minus)},
            ]
        }
    }

    rows = feature_evidence.summarize_ablation_results(cfg, config_dir=tmp_path)

    minus_row = next(row for row in rows if row["family"] == "minus_value")
    assert minus_row["delta_backtest_sharpe_vs_baseline"] == pytest.approx(-0.3)
    assert minus_row["feature_stability_available"] is True
    assert minus_row["feature_stability_top_k_hit_rate"] == 0.8


def test_permutation_active_return_importance_outputs_feature_and_family_rows(tmp_path):
    rows = []
    dates = pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17"])
    for d_idx, trade_date in enumerate(dates):
        for rank in range(4):
            rows.append(
                {
                    "trade_date": trade_date,
                    "signal_backtest": float(4 - rank),
                    "future_return": 0.01 * (4 - rank),
                    "f_good": float(4 - rank + d_idx * 0.1),
                    "f_bad": float(rank),
                }
            )
    scored = tmp_path / "scored.parquet"
    pd.DataFrame(rows).to_parquet(scored)

    cfg = {
        "feature_evidence": {
            "scored_file": str(scored),
            "features": ["f_good", "f_bad"],
            "families": {"combo": ["f_good", "f_bad"]},
            "score_col": "signal_backtest",
            "target_col": "future_return",
            "top_k": 1,
            "n_repeats": 2,
            "seed": 1,
        }
    }

    result = feature_evidence.permutation_active_return_importance(cfg, config_dir=tmp_path)

    assert {row["kind"] for row in result} == {"feature", "family"}
    good = next(row for row in result if row["name"] == "f_good")
    assert good["baseline_score_metric"] > 0
    assert good["permutation_importance"] >= 0


def test_factor_ic_report_outputs_feature_ic_and_quantile_fields(tmp_path):
    rows = []
    dates = pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17"])
    for trade_date in dates:
        for rank in range(4):
            target = 0.01 * (rank + 1)
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": f"s{rank}",
                    "future_return": target,
                    "f_good": target,
                    "f_bad": -target,
                }
            )
    factor_file = tmp_path / "dataset.parquet"
    pd.DataFrame(rows).set_index(["trade_date", "symbol"]).to_parquet(factor_file)

    cfg = {
        "feature_evidence": {
            "factor_ic_file": str(factor_file),
            "features": ["f_good", "f_bad"],
            "target_col": "future_return",
            "n_quantiles": 2,
        }
    }

    result = feature_evidence.factor_ic_report(cfg, config_dir=tmp_path)

    assert [row["feature"] for row in result] == ["f_bad", "f_good"]
    good = next(row for row in result if row["feature"] == "f_good")
    bad = next(row for row in result if row["feature"] == "f_bad")
    assert good["n"] == 3
    assert good["ic_mean"] == pytest.approx(1.0)
    assert good["pearson_ic_mean"] == pytest.approx(1.0)
    assert good["q1_return"] == pytest.approx(0.015)
    assert good["qN_return"] == pytest.approx(0.035)
    assert good["long_short"] == pytest.approx(0.02)
    assert good["coverage"] == pytest.approx(1.0)
    assert good["positive_ic_ratio"] == pytest.approx(1.0)
    assert bad["ic_mean"] == pytest.approx(-1.0)
    assert bad["long_short"] == pytest.approx(-0.02)


def test_correlation_audit_report_clusters_highly_correlated_features(tmp_path):
    data = pd.DataFrame(
        {
            "trade_date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "f1": [1, 2, 3, 4, 5],
            "f2": [2, 4, 6, 8, 10],
            "f3": [5, 1, 4, 2, 3],
        }
    )
    path = tmp_path / "features.parquet"
    data.to_parquet(path)

    rows = feature_evidence.correlation_audit_report(
        {
            "feature_evidence": {
                "correlation_file": str(path),
                "features": ["f1", "f2", "f3"],
                "correlation_threshold": 0.95,
            }
        },
        config_dir=tmp_path,
    )

    pair = next(
        row
        for row in rows
        if row["row_type"] == "pair" and row["feature_a"] == "f1" and row["feature_b"] == "f2"
    )
    assert pair["is_high_corr"] is True
    assert pair["cluster_size"] == 2


def test_drop_column_importance_report_flips_ablation_delta_sign(tmp_path):
    baseline = tmp_path / "baseline"
    minus = tmp_path / "minus_value"
    _write_summary(baseline, sharpe=1.0, ic_ir=0.4, wf=0.03)
    _write_summary(minus, sharpe=0.7, ic_ir=0.2, wf=0.01)

    rows = feature_evidence.drop_column_importance_report(
        {
            "feature_evidence": {
                "runs": [
                    {"family": "baseline", "run_dir": str(baseline)},
                    {"family": "minus_value", "run_dir": str(minus)},
                ]
            }
        },
        config_dir=tmp_path,
    )

    assert len(rows) == 1
    assert rows[0]["dropped_group"] == "value"
    assert rows[0]["drop_importance_backtest_sharpe"] == pytest.approx(0.3)
