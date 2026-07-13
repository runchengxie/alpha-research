import csv
import json
from dataclasses import replace
from pathlib import Path

import yaml

from cstree.alpha.research_artifacts import ArtifactHandle, write_strict_json
from cstree.alpha.tuning_service import (
    TuningApplicationService,
    TuningRequest,
    TuningTrialOutcome,
    parse_search_space,
)


class _Runner:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self.config_artifacts: list[ArtifactHandle] = []

    def run(self, job):
        self.config_artifacts.append(job.config_artifact)
        config = yaml.safe_load(job.config_artifact.verify().read_text(encoding="utf-8"))
        learning_rate = float(config["model"]["params"]["learning_rate"])
        sample_weight = config["model"]["sample_weight_mode"]
        eval_ic_ir = 0.20 if sample_weight == "exp_decay" else 0.10
        if learning_rate >= 0.05:
            eval_ic_ir += 0.03
        wf_ic = 0.06 if sample_weight == "exp_decay" else 0.02
        sharpe = 0.80 if learning_rate >= 0.05 else 0.50
        summary_path = write_strict_json(
            self.runs_dir / job.run_name / "summary.json",
            {
                "eval": {
                    "ic": {"ir": eval_ic_ir},
                    "cv_ic": {"mean": 0.02, "scores": [0.02, 0.01, 0.03]},
                    "constant_prediction": False,
                    "zero_feature_importance": False,
                },
                "walk_forward": {
                    "results": [
                        {"status": "ok", "test_ic": {"mean": wf_ic}},
                        {"status": "ok", "test_ic": {"mean": wf_ic / 2}},
                    ]
                },
                "backtest": {
                    "stats": {
                        "sharpe": sharpe,
                        "max_drawdown": -0.10,
                        "avg_turnover": 0.25,
                        "avg_cost_drag": 0.002,
                    }
                },
            },
        )
        return TuningTrialOutcome(
            summary_artifact=ArtifactHandle.from_file(
                summary_path,
                artifact_type="train_eval_summary",
                schema_version="v1",
            )
        )


def _request(tmp_path: Path, *, dry_run: bool = False) -> TuningRequest:
    dimensions = parse_search_space(
        [
            {"name": "lr", "path": "model.params.learning_rate", "values": [0.03, 0.05]},
            {
                "name": "sample_weight",
                "values": [
                    {
                        "label": "date_equal",
                        "overrides": {"model.sample_weight_mode": "date_equal"},
                    },
                    {
                        "label": "exp_h12",
                        "overrides": {
                            "model.sample_weight_mode": "exp_decay",
                            "model.sample_weight_params.halflife": 12,
                        },
                    },
                ],
            },
        ]
    )
    return TuningRequest(
        base_config={
            "model": {
                "type": "xgb_regressor",
                "params": {"learning_rate": 0.05},
                "sample_weight_mode": "date_equal",
            },
            "eval": {"output_dir": "unused", "run_name": "base"},
        },
        dimensions=dimensions,
        sweep_dir=str(tmp_path / "sweep"),
        sweep_tag="unit_tune",
        run_name_prefix="demo_",
        dry_run=dry_run,
    )


def test_tuning_service_preserves_trial_artifact_contract(tmp_path: Path) -> None:
    runner = _Runner(tmp_path / "runs")

    receipt = TuningApplicationService().apply(_request(tmp_path), runner)

    assert receipt.status == "completed"
    assert receipt.job_count == 4
    assert receipt.best_run_name == "demo_unit_tune_trial_004"
    sweep = tmp_path / "sweep"
    assert (sweep / "jobs.csv").exists()
    assert (sweep / "trial_results.csv").exists()
    assert (sweep / "best_trial.json").exists()
    assert (sweep / "best_config.yml").exists()
    assert (sweep / "tuning_receipt.json").exists()
    with (sweep / "trial_results.csv").open(newline="", encoding="utf-8") as handle:
        results = list(csv.DictReader(handle))
    assert len(results) == 4
    assert {row["status"] for row in results} == {"ok"}
    best = json.loads((sweep / "best_trial.json").read_text(encoding="utf-8"))
    assert best["objective_score"] == 0.5925
    assert all(
        artifact.artifact_type == "tuning_trial_config" for artifact in runner.config_artifacts
    )


def test_tuning_service_dry_run_needs_no_pipeline_runner(tmp_path: Path) -> None:
    receipt = TuningApplicationService().apply(_request(tmp_path, dry_run=True))

    assert receipt.status == "planned"
    assert receipt.job_count == 4
    assert receipt.completed_count == 0
    with (tmp_path / "sweep" / "trial_results.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []


def test_tuning_service_records_system_exit_and_stops_cleanly(tmp_path: Path) -> None:
    successful_runner = _Runner(tmp_path / "runs")

    class _StopsOnSecondTrial:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, job):
            self.calls += 1
            if self.calls == 2:
                raise SystemExit("pipeline failed")
            return successful_runner.run(job)

    request = replace(_request(tmp_path), continue_on_error=False)
    receipt = TuningApplicationService().apply(request, _StopsOnSecondTrial())

    assert receipt.status == "failed"
    assert receipt.completed_count == 1
    assert receipt.failed_count == 1
    with (tmp_path / "sweep" / "trial_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["ok", "failed"]
    assert rows[1]["error"] == "pipeline failed"
