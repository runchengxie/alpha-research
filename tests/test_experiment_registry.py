from __future__ import annotations

import json

from alpha_research.experiment_registry import ExperimentRegistry, ExperimentTrial


def test_registry_deduplicates_trials_and_round_trips(tmp_path):
    registry = ExperimentRegistry()
    trial = ExperimentTrial(
        candidate_id="model-a",
        feature_set="base",
        universe="a-share",
        holding_period=5,
        parameters={"depth": 4},
        status="completed",
    )

    assert registry.record(trial) is True
    assert registry.record(trial) is False
    assert registry.trial_count == 1

    path = tmp_path / "trials.json"
    registry.write(path)
    restored = ExperimentRegistry.read(path)
    assert restored.trial_count == 1
    assert restored.trials[0].trial_id == trial.trial_id
    assert json.loads(path.read_text())["schema_version"] == 1
