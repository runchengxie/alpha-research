import json

import pandas as pd

from alpha_research import overfitting_diagnostics


def test_uniqueness_report_outputs_sample_weights_and_bootstrap_ids():
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"]),
            "label_end": pd.to_datetime(["2020-01-03", "2020-01-04", "2020-01-05"]),
        }
    )

    report = overfitting_diagnostics.uniqueness_report(
        frame,
        label_end_col="label_end",
        bootstrap_samples=4,
        seed=1,
    )

    assert report["summary"]["event_count"] == 3
    assert len(report["summary"]["bootstrap_event_ids"]) == 4
    assert all(row["sample_weight"] > 0 for row in report["rows"])


def test_negative_control_report_includes_shift_random_and_sentinel_rows():
    rows = []
    for trade_date in pd.date_range("2020-01-01", periods=4, freq="D"):
        for rank in range(3):
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": f"s{rank}",
                    "target": float(rank),
                    "f1": float(rank),
                    "sentinel_future_return": float(rank),
                }
            )
    result = overfitting_diagnostics.negative_control_report(
        pd.DataFrame(rows),
        features=["f1"],
        target_col="target",
        sentinel_features=["sentinel_future_return"],
        random_features=1,
        random_universe_frac=0.5,
        seed=7,
    )

    controls = {row["control"] for row in result}
    assert {
        "baseline_feature_proxy",
        "shifted_label",
        "random_feature",
        "random_universe",
        "future_feature_sentinel",
    }.issubset(controls)


def test_scenario_backtest_report_and_candidate_freeze_manifest(tmp_path):
    returns = pd.DataFrame({"ret": [0.01, -0.02, 0.03, 0.00, 0.01]})
    scenario = overfitting_diagnostics.scenario_backtest_report(
        returns,
        return_col="ret",
        n_scenarios=5,
        block_size=2,
        seed=1,
    )

    assert scenario["summary"]["scenario_count"] == 5
    assert len(scenario["rows"]) == 5

    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    manifest = overfitting_diagnostics.candidate_freeze_manifest(
        run_dir=run,
        lifecycle_stage="paper_trading",
        paper_start_date="2026-01-01",
    )

    assert manifest["lifecycle_stage"] == "paper_trading"
    assert manifest["files"]["summary"]["exists"] is True
    assert manifest["files"]["summary"]["sha256"]
