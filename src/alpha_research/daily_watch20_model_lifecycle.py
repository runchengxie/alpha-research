"""Persisted-model cadence and compatibility gates for DailyWatch20.

This module is alpha-owned. Publication directories remain ordinary artifact
inputs; no strategy-pipeline runtime object crosses the package boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

DEFAULT_SELECTION_SCHEMA = "daily_watch20.selection.v2"


class RestorableRanker(Protocol):
    """Minimal runtime capability required to restore a persisted ranker."""

    def restore_from_path(
        self,
        model_path: Path,
        training: dict[str, Any],
        *,
        metadata: dict[str, Any],
    ) -> None: ...


@dataclass(frozen=True)
class ModelLifecycle:
    mode: str
    reason: str
    origin_run: str | None = None
    origin_source_date: str | None = None
    origin_training_as_of: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def passed_prior_runs(
    output_root: Path,
    source_date: str,
    *,
    selection_schema: str = DEFAULT_SELECTION_SCHEMA,
) -> list[tuple[str, str, Path]]:
    """Return passed prior publication runs in deterministic newest-first order."""

    runs_root = output_root / "runs"
    if not runs_root.is_dir():
        return []
    candidates: list[tuple[str, str, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        receipt_path = run_dir / "selection_receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(receipt, dict):
            continue
        prior_source = str(receipt.get("source_date") or "").replace("-", "")
        if (
            receipt.get("schema_version") != selection_schema
            or str(receipt.get("status") or "").lower() != "passed"
            or str(receipt.get("quality_status") or "passed").lower() != "passed"
            or len(prior_source) != 8
            or not prior_source.isdigit()
            or prior_source >= source_date
        ):
            continue
        candidates.append((prior_source, str(receipt.get("generated_at") or ""), run_dir))
    return sorted(candidates, key=lambda item: (item[0], item[1], item[2].name), reverse=True)


def _artifact_hash_matches(receipt: dict[str, Any], run_dir: Path, name: str) -> bool:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    entry = artifacts.get(name)
    if not isinstance(entry, dict):
        return False
    path = run_dir / str(entry.get("path") or name)
    expected = str(entry.get("sha256") or "")
    return path.is_file() and bool(expected) and _sha256_file(path) == expected


def _trade_date_distance(
    open_dates: pd.DatetimeIndex,
    earlier: str,
    later: str,
) -> int | None:
    positions = open_dates.get_indexer([pd.Timestamp(earlier), pd.Timestamp(later)])
    if bool((positions < 0).any()):
        return None
    return int(positions[1] - positions[0])


def restore_prior_ranker(
    ranker: RestorableRanker,
    *,
    output_root: Path,
    source_date: str,
    open_dates: pd.DatetimeIndex,
    max_age_trade_days: int,
    selection_schema: str = DEFAULT_SELECTION_SCHEMA,
) -> ModelLifecycle | None:
    """Restore the newest compatible, integrity-verified prior model."""

    for prior_source, _, run_dir in passed_prior_runs(
        output_root,
        source_date,
        selection_schema=selection_schema,
    ):
        prior_age = _trade_date_distance(open_dates, prior_source, source_date)
        if prior_age is None or prior_age <= 0 or prior_age > max_age_trade_days:
            continue
        receipt_path = run_dir / "selection_receipt.json"
        metadata_path = run_dir / "model_metadata.json"
        model_path = run_dir / "model.ubj"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(receipt, dict) or not isinstance(metadata, dict):
            continue
        if not all(
            _artifact_hash_matches(receipt, run_dir, name)
            for name in ("model.ubj", "model_metadata.json")
        ):
            continue
        persistence = metadata.get("persistence")
        if not isinstance(persistence, dict):
            persistence = {
                "model_version": metadata.get("model_version"),
                "feature_set_id": metadata.get("feature_set_id"),
                "training_policy_id": metadata.get("training_policy_id"),
            }
        training = metadata.get("training")
        if not isinstance(training, dict):
            continue
        training_as_of = str(training.get("as_of_date") or "")[:10].replace("-", "")
        if (
            len(training_as_of) != 8
            or not training_as_of.isdigit()
            or training_as_of > prior_source
        ):
            continue
        training_age = _trade_date_distance(open_dates, training_as_of, source_date)
        if training_age is None or training_age < 0 or training_age > max_age_trade_days:
            continue
        try:
            ranker.restore_from_path(model_path, training, metadata=dict(persistence))
        except (OSError, ValueError, RuntimeError):
            continue
        return ModelLifecycle(
            mode="reused",
            reason="compatible model reused between scheduled retrain weekdays",
            origin_run=str(run_dir),
            origin_source_date=prior_source,
            origin_training_as_of=training_as_of,
        )
    return None


def prepare_ranker_lifecycle(
    ranker: RestorableRanker,
    *,
    force_retrain: bool,
    retrain_weekdays: tuple[int, ...],
    max_age_trade_days: int,
    output_root: Path,
    source_date: str,
    open_dates: pd.DatetimeIndex,
    selection_schema: str = DEFAULT_SELECTION_SCHEMA,
) -> ModelLifecycle:
    """Choose deterministic train-versus-restore behavior for one source date."""

    weekday = pd.Timestamp(source_date).weekday()
    if force_retrain:
        return ModelLifecycle(mode="trained", reason="force_retrain requested")
    if weekday in set(retrain_weekdays):
        return ModelLifecycle(
            mode="trained",
            reason=f"source weekday {weekday} is a scheduled retrain weekday",
        )
    restored = restore_prior_ranker(
        ranker,
        output_root=output_root,
        source_date=source_date,
        open_dates=open_dates,
        max_age_trade_days=max_age_trade_days,
        selection_schema=selection_schema,
    )
    return restored or ModelLifecycle(
        mode="trained",
        reason="no compatible recent persisted model was available",
    )


__all__ = [
    "DEFAULT_SELECTION_SCHEMA",
    "ModelLifecycle",
    "RestorableRanker",
    "passed_prior_runs",
    "prepare_ranker_lifecycle",
    "restore_prior_ranker",
]
