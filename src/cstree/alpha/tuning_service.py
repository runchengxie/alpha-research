from __future__ import annotations

import csv
import itertools
import json
import math
import random
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from .backends import ExperimentRecorder, NullExperimentRecorder
from .research_artifacts import (
    ArtifactHandle,
    JsonValue,
    load_json_mapping,
    strict_json_mapping,
    strict_json_value,
    write_strict_json,
)

TUNING_SCHEMA = "alpha_tuning_receipt.v1"


@dataclass(frozen=True)
class TuneChoice:
    label: str
    overrides: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("TuneChoice.label cannot be empty")
        normalized = strict_json_mapping(self.overrides, field=f"choice[{self.label}].overrides")
        if not normalized:
            raise ValueError("TuneChoice.overrides cannot be empty")
        object.__setattr__(self, "overrides", normalized)


@dataclass(frozen=True)
class TuneDimension:
    name: str
    choices: tuple[TuneChoice, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("TuneDimension.name cannot be empty")
        if not self.choices:
            raise ValueError(f"TuneDimension {self.name!r} must contain choices")


@dataclass(frozen=True)
class ObjectiveSpec:
    eval_ic_ir_weight: float = 1.0
    walk_forward_test_ic_mean_weight: float = 0.5
    backtest_sharpe_weight: float = 0.5
    drawdown_penalty_weight: float = 0.25
    cost_drag_penalty_weight: float = 5.0
    turnover_penalty_weight: float = 0.1
    drop_degenerate: bool = True
    min_cv_ic_valid_folds: int = 0

    def __post_init__(self) -> None:
        if self.min_cv_ic_valid_folds < 0:
            raise ValueError("min_cv_ic_valid_folds must be >= 0")
        for name in (
            "eval_ic_ir_weight",
            "walk_forward_test_ic_mean_weight",
            "backtest_sharpe_weight",
            "drawdown_penalty_weight",
            "cost_drag_penalty_weight",
            "turnover_penalty_weight",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class TuningRequest:
    base_config: Mapping[str, JsonValue]
    dimensions: tuple[TuneDimension, ...]
    sweep_dir: str
    sweep_tag: str
    run_name_prefix: str
    sampler: str = "grid"
    n_trials: int | None = None
    seed: int = 42
    runs_dir_override: str | None = None
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    dry_run: bool = False
    continue_on_error: bool = False
    experiment_name: str = "alpha-tuning"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_config",
            strict_json_mapping(self.base_config, field="tuning.base_config"),
        )
        if not self.dimensions:
            raise ValueError("TuningRequest.dimensions cannot be empty")
        names = [dimension.name for dimension in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError("Tuning dimension names must be unique")
        if not self.sweep_tag.strip() or not self.run_name_prefix.strip():
            raise ValueError("sweep_tag and run_name_prefix cannot be empty")
        if self.sampler not in {"grid", "random"}:
            raise ValueError("sampler must be 'grid' or 'random'")
        if self.n_trials is not None and self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")


@dataclass(frozen=True)
class TuningJob:
    order: int
    run_name: str
    config_artifact: ArtifactHandle
    dimension_labels: Mapping[str, str]
    overrides: Mapping[str, JsonValue]

    @property
    def config_path(self) -> Path:
        return Path(self.config_artifact.path)


@dataclass(frozen=True)
class TuningTrialOutcome:
    summary_artifact: ArtifactHandle
    artifacts: tuple[ArtifactHandle, ...] = ()


@runtime_checkable
class TuningTrialRunner(Protocol):
    """Strategy-pipeline adapter boundary; implementations may invoke orchestration."""

    def run(self, job: TuningJob) -> TuningTrialOutcome: ...


@dataclass(frozen=True)
class TuningApplicationReceipt:
    status: str
    artifact: ArtifactHandle
    job_count: int
    completed_count: int
    failed_count: int
    best_run_name: str | None

    def to_metadata(self) -> dict[str, JsonValue]:
        return {
            "status": self.status,
            "artifact": self.artifact.to_metadata(),
            "job_count": self.job_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "best_run_name": self.best_run_name,
        }


def _choice_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value).strip() or "choice"


def parse_search_space(raw: Any) -> tuple[TuneDimension, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("search_space must be a non-empty list")
    dimensions: list[TuneDimension] = []
    for dimension_index, raw_dimension in enumerate(raw, start=1):
        if not isinstance(raw_dimension, Mapping):
            raise ValueError(f"search_space item #{dimension_index} must be a mapping")
        raw_path = raw_dimension.get("path")
        path = str(raw_path).strip() if raw_path is not None else None
        if path == "":
            path = None
        fallback = path.split(".")[-1] if path else f"dim_{dimension_index}"
        name = str(raw_dimension.get("name") or fallback).strip()
        raw_values = raw_dimension.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"search_space[{name}] values must be a non-empty list")
        choices = tuple(
            _parse_choice(choice, path=path, dimension=name, index=index)
            for index, choice in enumerate(raw_values, start=1)
        )
        dimensions.append(TuneDimension(name=name, choices=choices))
    return tuple(dimensions)


def _parse_choice(raw: Any, *, path: str | None, dimension: str, index: int) -> TuneChoice:
    if not isinstance(raw, Mapping):
        if path is None:
            raise ValueError(f"search_space[{dimension}] scalar choice requires path")
        value = strict_json_value(raw, field=f"search_space[{dimension}][{index}]")
        return TuneChoice(label=_choice_label(value), overrides={path: value})
    overrides_raw = raw.get("overrides") or {}
    if not isinstance(overrides_raw, Mapping):
        raise ValueError(f"search_space[{dimension}] overrides must be a mapping")
    overrides = strict_json_mapping(overrides_raw, field=f"search_space[{dimension}].overrides")
    has_value = "value" in raw
    if path is not None and has_value:
        overrides = {
            path: strict_json_value(raw.get("value"), field=f"search_space[{dimension}].value"),
            **overrides,
        }
    if not overrides:
        raise ValueError(f"search_space[{dimension}] choice #{index} has no override")
    label = str(raw.get("label") or "").strip()
    if not label:
        if has_value:
            label = _choice_label(raw.get("value"))
        elif len(overrides) == 1:
            label = _choice_label(next(iter(overrides.values())))
        else:
            label = f"choice_{index}"
    return TuneChoice(label=label, overrides=overrides)


def objective_from_mapping(raw: Mapping[str, Any] | None) -> ObjectiveSpec:
    values = raw or {}

    def number(name: str, default: float) -> float:
        try:
            return float(values.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"objective.{name} must be numeric") from exc

    drop_raw = values.get("drop_degenerate", True)
    if isinstance(drop_raw, bool):
        drop_degenerate = drop_raw
    elif str(drop_raw).strip().lower() in {"1", "true", "yes", "on"}:
        drop_degenerate = True
    elif str(drop_raw).strip().lower() in {"0", "false", "no", "off"}:
        drop_degenerate = False
    else:
        raise ValueError("objective.drop_degenerate must be boolean")

    return ObjectiveSpec(
        eval_ic_ir_weight=number("eval_ic_ir_weight", 1.0),
        walk_forward_test_ic_mean_weight=number("walk_forward_test_ic_mean_weight", 0.5),
        backtest_sharpe_weight=number("backtest_sharpe_weight", 0.5),
        drawdown_penalty_weight=number("drawdown_penalty_weight", 0.25),
        cost_drag_penalty_weight=number("cost_drag_penalty_weight", 5.0),
        turnover_penalty_weight=number("turnover_penalty_weight", 0.1),
        drop_degenerate=drop_degenerate,
        min_cv_ic_valid_folds=int(values.get("min_cv_ic_valid_folds", 0)),
    )


def _nested(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _walk_forward_mean(summary: Mapping[str, Any]) -> float | None:
    raw_results = _nested(summary, "walk_forward.results")
    if not isinstance(raw_results, list):
        return None
    values = [
        value
        for item in raw_results
        if isinstance(item, Mapping) and str(item.get("status", "")).lower() == "ok"
        if (value := _finite_float(_nested(item, "test_ic.mean"))) is not None
    ]
    return sum(values) / len(values) if values else None


def score_trial_summary(
    summary: Mapping[str, Any], objective: ObjectiveSpec
) -> dict[str, JsonValue]:
    eval_ic_ir = _finite_float(_nested(summary, "eval.ic.ir"))
    raw_scores = _nested(summary, "eval.cv_ic.scores")
    if isinstance(raw_scores, list):
        valid_scores = [value for raw in raw_scores if (value := _finite_float(raw)) is not None]
        cv_total_folds: int | None = len(raw_scores)
        cv_valid_folds = len(valid_scores)
    else:
        cv_total_folds = None
        cv_valid_folds = 1 if _finite_float(_nested(summary, "eval.cv_ic.mean")) is not None else 0
    cv_mean = _finite_float(_nested(summary, "eval.cv_ic.mean"))
    walk_forward_mean = _walk_forward_mean(summary)
    sharpe = _finite_float(_nested(summary, "backtest.stats.sharpe"))
    drawdown = _finite_float(_nested(summary, "backtest.stats.max_drawdown"))
    turnover = _finite_float(_nested(summary, "backtest.stats.avg_turnover"))
    cost_drag = _finite_float(_nested(summary, "backtest.stats.avg_cost_drag"))
    constant = bool(_nested(summary, "eval.constant_prediction") or False)
    zero_importance = bool(_nested(summary, "eval.zero_feature_importance") or False)
    insufficient = cv_valid_folds < objective.min_cv_ic_valid_folds
    components = {
        "objective_component_eval_ic_ir": objective.eval_ic_ir_weight * (eval_ic_ir or 0.0),
        "objective_component_walk_forward_test_ic_mean": (
            objective.walk_forward_test_ic_mean_weight * (walk_forward_mean or 0.0)
        ),
        "objective_component_backtest_sharpe": objective.backtest_sharpe_weight * (sharpe or 0.0),
        "objective_component_drawdown_penalty": objective.drawdown_penalty_weight
        * abs(drawdown or 0.0),
        "objective_component_cost_drag_penalty": objective.cost_drag_penalty_weight
        * (cost_drag or 0.0),
        "objective_component_turnover_penalty": objective.turnover_penalty_weight
        * (turnover or 0.0),
    }
    score: float | None = None
    if (
        eval_ic_ir is not None
        and not insufficient
        and not (objective.drop_degenerate and (constant or zero_importance))
    ):
        score = (
            components["objective_component_eval_ic_ir"]
            + components["objective_component_walk_forward_test_ic_mean"]
            + components["objective_component_backtest_sharpe"]
            - components["objective_component_drawdown_penalty"]
            - components["objective_component_cost_drag_penalty"]
            - components["objective_component_turnover_penalty"]
        )
    return {
        "objective_score": score,
        "eval_ic_ir": eval_ic_ir,
        "eval_cv_ic_mean": cv_mean,
        "eval_cv_ic_valid_folds": cv_valid_folds,
        "eval_cv_ic_total_folds": cv_total_folds,
        "walk_forward_test_ic_mean": walk_forward_mean,
        "backtest_sharpe": sharpe,
        "backtest_max_drawdown": drawdown,
        "backtest_avg_turnover": turnover,
        "backtest_avg_cost_drag": cost_drag,
        **components,
        "flag_constant_prediction": constant,
        "flag_zero_feature_importance": zero_importance,
        "flag_cv_ic_insufficient": insufficient,
    }


def _set_nested(payload: dict[str, JsonValue], path: str, value: JsonValue) -> None:
    parts = [part.strip() for part in path.split(".") if part.strip()]
    if not parts:
        raise ValueError("Tune override path cannot be empty")
    current: dict[str, JsonValue] = payload
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = strict_json_value(value, field=f"override.{path}")


def _combinations(request: TuningRequest) -> list[tuple[dict[str, str], dict[str, JsonValue]]]:
    combinations: list[tuple[dict[str, str], dict[str, JsonValue]]] = []
    for selected in itertools.product(*(dimension.choices for dimension in request.dimensions)):
        labels: dict[str, str] = {}
        overrides: dict[str, JsonValue] = {}
        for dimension, choice in zip(request.dimensions, selected, strict=True):
            labels[dimension.name] = choice.label
            for path, value in choice.overrides.items():
                if path in overrides and overrides[path] != value:
                    raise ValueError(f"Conflicting tune overrides for {path!r}")
                overrides[path] = value
        combinations.append((labels, overrides))
    trial_count = request.n_trials if request.n_trials is not None else 20
    if request.sampler == "random" and trial_count < len(combinations):
        indexes = sorted(random.Random(request.seed).sample(range(len(combinations)), trial_count))
        return [combinations[index] for index in indexes]
    return combinations


def _write_yaml(path: Path, payload: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _build_jobs(request: TuningRequest, sweep_dir: Path) -> list[TuningJob]:
    jobs: list[TuningJob] = []
    config_dir = sweep_dir / "configs"
    for order, (labels, overrides) in enumerate(_combinations(request), start=1):
        config = strict_json_mapping(request.base_config, field="base_config")
        for path, value in overrides.items():
            _set_nested(config, path, value)
        eval_config = config.get("eval")
        if not isinstance(eval_config, dict):
            eval_config = {}
            config["eval"] = eval_config
        run_name = f"{request.run_name_prefix}{request.sweep_tag}_trial_{order:03d}"
        eval_config["run_name"] = run_name
        if request.runs_dir_override is not None:
            eval_config["output_dir"] = request.runs_dir_override
        config_path = config_dir / f"trial_{order:03d}.yml"
        _write_yaml(config_path, config)
        config_artifact = ArtifactHandle.from_file(
            config_path,
            artifact_type="tuning_trial_config",
            schema_version="v1",
            media_type="application/yaml",
        )
        jobs.append(
            TuningJob(
                order=order,
                run_name=run_name,
                config_artifact=config_artifact,
                dimension_labels=labels,
                overrides=overrides,
            )
        )
    return jobs


_METRIC_FIELDS = (
    "summary_path",
    "summary_artifact_json",
    "experiment_receipt_json",
    "objective_score",
    "eval_ic_ir",
    "eval_cv_ic_mean",
    "eval_cv_ic_valid_folds",
    "eval_cv_ic_total_folds",
    "walk_forward_test_ic_mean",
    "backtest_sharpe",
    "backtest_max_drawdown",
    "backtest_avg_turnover",
    "backtest_avg_cost_drag",
    "objective_component_eval_ic_ir",
    "objective_component_walk_forward_test_ic_mean",
    "objective_component_backtest_sharpe",
    "objective_component_drawdown_penalty",
    "objective_component_cost_drag_penalty",
    "objective_component_turnover_penalty",
    "flag_constant_prediction",
    "flag_zero_feature_importance",
    "flag_cv_ic_insufficient",
    "status",
    "error",
    "dimensions_json",
)


def _write_jobs(path: Path, jobs: Sequence[TuningJob], names: Sequence[str]) -> None:
    fields = ["order", "run_name", "config_path", *names, "overrides_json"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            row: dict[str, Any] = {
                "order": job.order,
                "run_name": job.run_name,
                "config_path": str(job.config_path),
                "overrides_json": json.dumps(job.overrides, sort_keys=True, allow_nan=False),
            }
            row.update(job.dimension_labels)
            writer.writerow(row)


def _write_results(path: Path, rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> None:
    fields = ["order", "run_name", "config_path", *names, *_METRIC_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for raw_row in rows:
            row = {key: raw_row.get(key) for key in fields}
            row["dimensions_json"] = json.dumps(
                raw_row.get("dimensions", {}), sort_keys=True, allow_nan=False
            )
            writer.writerow(row)


def _initial_row(job: TuningJob) -> dict[str, Any]:
    return {
        "order": job.order,
        "run_name": job.run_name,
        "config_path": str(job.config_path),
        **job.dimension_labels,
        "status": "ok",
        "error": "",
        "dimensions": dict(job.dimension_labels),
    }


def _best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("status") == "ok" and _finite_float(row.get("objective_score")) is not None
    ]
    return max(candidates, key=lambda row: float(row["objective_score"])) if candidates else None


def _write_best(sweep_dir: Path, best: Mapping[str, Any]) -> None:
    metric_keys = [field for field in _METRIC_FIELDS if field.startswith("objective_")]
    metric_keys.extend(
        field
        for field in _METRIC_FIELDS
        if field.startswith(("eval_", "walk_forward_", "backtest_", "flag_"))
    )
    payload = {
        "run_name": best["run_name"],
        "config_path": best["config_path"],
        "summary_path": best.get("summary_path"),
        "objective_score": _finite_float(best.get("objective_score")),
        "dimensions": best.get("dimensions", {}),
        "metrics": {key: best.get(key) for key in metric_keys},
    }
    write_strict_json(sweep_dir / "best_trial.json", payload)
    shutil.copyfile(str(best["config_path"]), sweep_dir / "best_config.yml")


class TuningApplicationService:
    """Apply alpha-owned tuning policy through a caller-supplied orchestration adapter."""

    def __init__(self, recorder: ExperimentRecorder | None = None) -> None:
        self._recorder = recorder or NullExperimentRecorder()

    def apply(
        self,
        request: TuningRequest,
        runner: TuningTrialRunner | None = None,
    ) -> TuningApplicationReceipt:
        if not request.dry_run and runner is None:
            raise ValueError("A TuningTrialRunner is required unless dry_run=True")
        sweep_dir = Path(request.sweep_dir).expanduser().resolve()
        sweep_dir.mkdir(parents=True, exist_ok=True)
        jobs = _build_jobs(request, sweep_dir)
        dimension_names = [dimension.name for dimension in request.dimensions]
        _write_jobs(sweep_dir / "jobs.csv", jobs, dimension_names)
        rows = self._apply_jobs(request, jobs, runner)
        _write_results(sweep_dir / "trial_results.csv", rows, dimension_names)
        best = _best_row(rows)
        if best is not None:
            _write_best(sweep_dir, best)
        failed_count = sum(row.get("status") == "failed" for row in rows)
        completed_count = sum(row.get("status") == "ok" for row in rows)
        status = "planned" if request.dry_run else ("failed" if failed_count else "completed")
        manifest_path = write_strict_json(
            sweep_dir / "tuning_receipt.json",
            {
                "schema": TUNING_SCHEMA,
                "status": status,
                "sweep_tag": request.sweep_tag,
                "sampler": request.sampler,
                "seed": request.seed,
                "job_count": len(jobs),
                "completed_count": completed_count,
                "failed_count": failed_count,
                "best_run_name": best.get("run_name") if best is not None else None,
                "artifacts": {
                    "configs_dir": str((sweep_dir / "configs").resolve()),
                    "jobs_csv": str((sweep_dir / "jobs.csv").resolve()),
                    "trial_results_csv": str((sweep_dir / "trial_results.csv").resolve()),
                    "best_trial_json": str((sweep_dir / "best_trial.json").resolve())
                    if best is not None
                    else None,
                    "best_config_yml": str((sweep_dir / "best_config.yml").resolve())
                    if best is not None
                    else None,
                },
            },
        )
        artifact = ArtifactHandle.from_file(
            manifest_path,
            artifact_type="alpha_tuning_receipt",
            schema_version="v1",
        )
        return TuningApplicationReceipt(
            status=status,
            artifact=artifact,
            job_count=len(jobs),
            completed_count=completed_count,
            failed_count=failed_count,
            best_run_name=str(best["run_name"]) if best is not None else None,
        )

    def _apply_jobs(
        self,
        request: TuningRequest,
        jobs: Sequence[TuningJob],
        runner: TuningTrialRunner | None,
    ) -> list[dict[str, Any]]:
        if request.dry_run:
            return []
        assert runner is not None
        rows: list[dict[str, Any]] = []
        for job in jobs:
            row = _initial_row(job)
            experiment_receipt = self._recorder.start(
                experiment_name=request.experiment_name,
                run_name=job.run_name,
                metadata={"config_artifact": job.config_artifact.to_metadata()},
            )
            row["experiment_receipt_json"] = json.dumps(
                experiment_receipt.to_metadata(), sort_keys=True, allow_nan=False
            )
            try:
                outcome = runner.run(job)
                summary_path = outcome.summary_artifact.verify()
                summary = load_json_mapping(summary_path)
                row["summary_path"] = str(summary_path)
                row["summary_artifact_json"] = json.dumps(
                    outcome.summary_artifact.to_metadata(), sort_keys=True, allow_nan=False
                )
                row.update(score_trial_summary(summary, request.objective))
                metrics = {
                    key: value
                    for key, raw_value in row.items()
                    if key.startswith(("eval_", "walk_forward_", "backtest_", "objective_"))
                    if (value := _finite_float(raw_value)) is not None
                }
                self._recorder.log_metrics(experiment_receipt, metrics)
                self._recorder.close(experiment_receipt, status="completed")
            except KeyboardInterrupt:
                self._recorder.close(experiment_receipt, status="failed")
                raise
            except (Exception, SystemExit) as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                self._recorder.close(experiment_receipt, status="failed")
            rows.append(row)
            if row["status"] == "failed" and not request.continue_on_error:
                break
        return rows


__all__ = [
    "TUNING_SCHEMA",
    "ObjectiveSpec",
    "TuneChoice",
    "TuneDimension",
    "TuningApplicationReceipt",
    "TuningApplicationService",
    "TuningJob",
    "TuningRequest",
    "TuningTrialOutcome",
    "TuningTrialRunner",
    "objective_from_mapping",
    "parse_search_space",
    "score_trial_summary",
]
