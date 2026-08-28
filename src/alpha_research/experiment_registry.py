"""Append-only registry for all research trials used by overfitting controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class ExperimentTrial:
    candidate_id: str
    feature_set: str
    universe: str
    holding_period: int
    parameters: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    trial_id: str = field(init=False)

    def __post_init__(self) -> None:
        identity = {
            "candidate_id": self.candidate_id,
            "feature_set": self.feature_set,
            "universe": self.universe,
            "holding_period": self.holding_period,
            "parameters": self.parameters,
        }
        digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
        object.__setattr__(self, "trial_id", digest)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ExperimentTrial:
        return cls(
            candidate_id=str(payload["candidate_id"]),
            feature_set=str(payload["feature_set"]),
            universe=str(payload["universe"]),
            holding_period=int(payload["holding_period"]),
            parameters=dict(payload.get("parameters", {})),
            status=str(payload.get("status", "completed")),
        )


@dataclass
class ExperimentRegistry:
    trials: list[ExperimentTrial] = field(default_factory=list)

    @property
    def trial_count(self) -> int:
        return len(self.trials)

    def record(self, trial: ExperimentTrial) -> bool:
        if any(item.trial_id == trial.trial_id for item in self.trials):
            return False
        self.trials.append(trial)
        return True

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "trial_count": self.trial_count,
            "trials": [trial.to_mapping() for trial in self.trials],
        }

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_canonical_json(self.to_mapping()) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> ExperimentRegistry:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        registry = cls()
        for raw_trial in payload.get("trials", []):
            registry.record(ExperimentTrial.from_mapping(raw_trial))
        return registry
