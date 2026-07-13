from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from .backends import ExperimentReceipt, FittedModelHandle
from .research_artifacts import (
    ArtifactHandle,
    ArtifactIntegrityError,
    JsonValue,
    load_json_mapping,
    sha256_file,
    strict_json_mapping,
    write_strict_json,
)

EVALUATION_SCHEMA = "backend_evaluation.v1"
COMPARISON_SCHEMA = "backend_comparison.v1"
REPLAY_RECEIPT_SCHEMA = "backend_comparison_replay_receipt.v1"


@dataclass(frozen=True)
class BackendPromotionThresholds:
    min_overlap_rows: int = 1
    min_overlap_ratio: float = 1.0
    min_prediction_pearson: float = 0.999
    min_prediction_spearman: float = 0.999
    max_prediction_mae: float = 1e-8
    max_prediction_abs_error: float = 1e-7
    metric_abs_tolerances: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_overlap_rows < 1:
            raise ValueError("min_overlap_rows must be >= 1")
        for name in ("min_overlap_ratio", "min_prediction_pearson", "min_prediction_spearman"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("max_prediction_mae", "max_prediction_abs_error"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        tolerances: dict[str, float] = {}
        for key, raw_value in self.metric_abs_tolerances.items():
            value = float(raw_value)
            if not str(key).strip() or not math.isfinite(value) or value < 0.0:
                raise ValueError("metric_abs_tolerances must use non-empty keys and finite values")
            tolerances[str(key)] = value
        object.__setattr__(self, "metric_abs_tolerances", tolerances)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> BackendPromotionThresholds:
        tolerances = payload.get("metric_abs_tolerances") or {}
        if not isinstance(tolerances, Mapping):
            raise ArtifactIntegrityError("metric_abs_tolerances must be a mapping")
        return cls(
            min_overlap_rows=int(payload.get("min_overlap_rows", 1)),
            min_overlap_ratio=float(payload.get("min_overlap_ratio", 1.0)),
            min_prediction_pearson=float(payload.get("min_prediction_pearson", 0.999)),
            min_prediction_spearman=float(payload.get("min_prediction_spearman", 0.999)),
            max_prediction_mae=float(payload.get("max_prediction_mae", 1e-8)),
            max_prediction_abs_error=float(payload.get("max_prediction_abs_error", 1e-7)),
            metric_abs_tolerances={str(key): float(value) for key, value in tolerances.items()},
        )

    def to_metadata(self) -> dict[str, JsonValue]:
        return {
            "min_overlap_rows": self.min_overlap_rows,
            "min_overlap_ratio": self.min_overlap_ratio,
            "min_prediction_pearson": self.min_prediction_pearson,
            "min_prediction_spearman": self.min_prediction_spearman,
            "max_prediction_mae": self.max_prediction_mae,
            "max_prediction_abs_error": self.max_prediction_abs_error,
            "metric_abs_tolerances": dict(self.metric_abs_tolerances),
        }


@dataclass(frozen=True)
class _EvaluationBundle:
    handle: ArtifactHandle
    manifest: dict[str, Any]
    predictions: pd.DataFrame


def _finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw_value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError("evaluation metric names must be non-empty strings")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"evaluation metric {key!r} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"evaluation metric {key!r} must be finite")
        result[key] = value
    return result


def _canonical_predictions(
    frame: pd.DataFrame,
    *,
    date_col: str,
    symbol_col: str,
    prediction_col: str,
    target_col: str | None,
) -> pd.DataFrame:
    columns = [date_col, symbol_col, prediction_col]
    if target_col is not None:
        columns.append(target_col)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction frame is missing required columns: {missing}")
    result = cast(pd.DataFrame, frame.loc[:, columns].copy())
    date_series = cast(pd.Series, result.loc[:, date_col])
    parsed_dates = pd.to_datetime(date_series, errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError(f"{date_col} contains invalid dates")
    result[date_col] = parsed_dates.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    symbol_series = cast(pd.Series, result.loc[:, symbol_col]).astype("string")
    result[symbol_col] = symbol_series
    if symbol_series.isna().any() or (symbol_series.str.len() == 0).any():
        raise ValueError(f"{symbol_col} contains missing or empty values")
    prediction_series = cast(pd.Series, result.loc[:, prediction_col])
    prediction_series = cast(
        pd.Series,
        pd.to_numeric(prediction_series, errors="coerce"),
    )
    result[prediction_col] = prediction_series
    if not np.isfinite(np.asarray(prediction_series, dtype=np.float64)).all():
        raise ValueError(f"{prediction_col} must contain finite numeric values")
    if target_col is not None:
        target_series = cast(pd.Series, result.loc[:, target_col])
        result[target_col] = pd.to_numeric(target_series, errors="coerce")
    key_columns = [date_col, symbol_col]
    if result.duplicated(subset=key_columns).any():
        raise ValueError(f"prediction frame contains duplicate keys: {key_columns}")
    return result.sort_values(by=key_columns, kind="mergesort").reset_index(drop=True)


def write_backend_evaluation(
    output_dir: str | Path,
    *,
    backend_id: str,
    run_id: str,
    predictions: pd.DataFrame,
    metrics: Mapping[str, Any],
    model_handle: FittedModelHandle | None = None,
    experiment_receipt: ExperimentReceipt | None = None,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    prediction_col: str = "pred",
    target_col: str | None = None,
) -> ArtifactHandle:
    """Persist one backend evaluation without serializing a fitted framework object."""

    if not backend_id.strip() or not run_id.strip():
        raise ValueError("backend_id and run_id cannot be empty")
    if model_handle is not None and model_handle.backend_id != backend_id:
        raise ValueError("model_handle backend_id does not match the evaluation backend_id")
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = _canonical_predictions(
        predictions,
        date_col=date_col,
        symbol_col=symbol_col,
        prediction_col=prediction_col,
        target_col=target_col,
    )
    predictions_path = out_dir / "predictions.csv"
    canonical.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
        float_format="%.17g",
        lineterminator="\n",
    )
    manifest_path = out_dir / "evaluation.json"
    payload: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "backend_id": backend_id,
        "run_id": run_id,
        "predictions": {
            "file": predictions_path.name,
            "sha256": sha256_file(predictions_path),
            "row_count": len(canonical),
            "date_col": date_col,
            "symbol_col": symbol_col,
            "prediction_col": prediction_col,
            "target_col": target_col,
        },
        "metrics": _finite_metrics(metrics),
        "model_handle": model_handle.to_metadata() if model_handle is not None else None,
        "experiment_receipt": (
            experiment_receipt.to_metadata() if experiment_receipt is not None else None
        ),
    }
    write_strict_json(manifest_path, payload)
    return ArtifactHandle.from_file(
        manifest_path,
        artifact_type="backend_evaluation",
        schema_version="v1",
    )


def _load_evaluation(handle: ArtifactHandle) -> _EvaluationBundle:
    manifest_path = handle.verify()
    if handle.artifact_type != "backend_evaluation" or handle.schema_version != "v1":
        raise ArtifactIntegrityError("Expected a backend_evaluation v1 artifact handle")
    manifest = load_json_mapping(manifest_path)
    if manifest.get("schema") != EVALUATION_SCHEMA:
        raise ArtifactIntegrityError(f"Unsupported evaluation schema: {manifest.get('schema')!r}")
    prediction_spec = manifest.get("predictions")
    if not isinstance(prediction_spec, dict):
        raise ArtifactIntegrityError("Evaluation artifact is missing predictions metadata")
    relative_file = prediction_spec.get("file")
    if not isinstance(relative_file, str) or Path(relative_file).is_absolute():
        raise ArtifactIntegrityError("predictions.file must be a relative path")
    predictions_path = (manifest_path.parent / relative_file).resolve()
    if predictions_path.parent != manifest_path.parent:
        raise ArtifactIntegrityError("predictions.file must remain inside the evaluation directory")
    if not predictions_path.is_file():
        raise ArtifactIntegrityError(f"Prediction artifact does not exist: {predictions_path}")
    expected_digest = prediction_spec.get("sha256")
    if not isinstance(expected_digest, str) or sha256_file(predictions_path) != expected_digest:
        raise ArtifactIntegrityError(f"Prediction digest mismatch: {predictions_path}")
    predictions = pd.read_csv(
        predictions_path,
        dtype={str(prediction_spec["symbol_col"]): "string"},
    )
    if len(predictions) != int(prediction_spec.get("row_count", -1)):
        raise ArtifactIntegrityError("Prediction row count does not match the evaluation manifest")
    return _EvaluationBundle(handle=handle, manifest=manifest, predictions=predictions)


def _correlation(
    left: pd.Series,
    right: pd.Series,
    method: Literal["pearson", "spearman"],
) -> float | None:
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    if np.array_equal(left_values, right_values):
        return 1.0
    value = left.corr(right, method=method)
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _comparison_values(
    native: _EvaluationBundle,
    candidate: _EvaluationBundle,
    thresholds: BackendPromotionThresholds,
) -> tuple[dict[str, JsonValue], list[str], str]:
    native_backend_id = str(native.manifest.get("backend_id"))
    candidate_backend_id = str(candidate.manifest.get("backend_id"))
    identity_result = _backend_identity_result(
        native,
        candidate,
        native_backend_id=native_backend_id,
        candidate_backend_id=candidate_backend_id,
    )
    if identity_result is not None:
        return identity_result
    native_spec = native.manifest["predictions"]
    candidate_spec = candidate.manifest["predictions"]
    native_keys = (native_spec["date_col"], native_spec["symbol_col"])
    candidate_keys = (candidate_spec["date_col"], candidate_spec["symbol_col"])
    failures: list[str] = []
    if native_keys != candidate_keys:
        return (
            {"overlap_rows": 0, "overlap_ratio": 0.0},
            ["incompatible_key_columns"],
            ("non-comparable"),
        )
    keys = list(native_keys)
    native_prediction = str(native_spec["prediction_col"])
    candidate_prediction = str(candidate_spec["prediction_col"])
    native_frame = cast(
        pd.DataFrame,
        native.predictions.loc[:, [*keys, native_prediction]],
    ).rename(columns={native_prediction: "prediction_native"})
    candidate_frame = cast(
        pd.DataFrame,
        candidate.predictions.loc[:, [*keys, candidate_prediction]],
    ).rename(columns={candidate_prediction: "prediction_candidate"})
    joined = native_frame.merge(
        candidate_frame,
        on=keys,
        how="inner",
        suffixes=("_native", "_candidate"),
        validate="one_to_one",
    )
    overlap_rows = len(joined)
    denominator = max(len(native.predictions), len(candidate.predictions), 1)
    overlap_ratio = overlap_rows / denominator
    if overlap_rows == 0:
        return (
            {"overlap_rows": 0, "overlap_ratio": overlap_ratio},
            ["no_overlapping_rows"],
            ("non-comparable"),
        )
    left = cast(pd.Series, joined.loc[:, "prediction_native"])
    right = cast(pd.Series, joined.loc[:, "prediction_candidate"])
    absolute_error = (left - right).abs()
    pearson = _correlation(left, right, "pearson")
    spearman = _correlation(left, right, "spearman")
    metric_deltas: dict[str, JsonValue] = {}
    native_metrics = native.manifest.get("metrics") or {}
    candidate_metrics = candidate.manifest.get("metrics") or {}
    for metric, tolerance in thresholds.metric_abs_tolerances.items():
        if metric not in native_metrics or metric not in candidate_metrics:
            metric_deltas[metric] = None
            failures.append(f"missing_metric:{metric}")
            continue
        delta = abs(float(candidate_metrics[metric]) - float(native_metrics[metric]))
        metric_deltas[metric] = delta
        if delta > tolerance:
            failures.append(f"metric_delta_above_threshold:{metric}")
    mae = float(absolute_error.mean())
    max_abs_error = float(absolute_error.max())
    if overlap_rows < thresholds.min_overlap_rows:
        failures.append("insufficient_overlap_rows")
    if overlap_ratio < thresholds.min_overlap_ratio:
        failures.append("insufficient_overlap_ratio")
    if pearson is None or pearson < thresholds.min_prediction_pearson:
        failures.append("prediction_pearson_below_threshold")
    if spearman is None or spearman < thresholds.min_prediction_spearman:
        failures.append("prediction_spearman_below_threshold")
    if mae > thresholds.max_prediction_mae:
        failures.append("prediction_mae_above_threshold")
    if max_abs_error > thresholds.max_prediction_abs_error:
        failures.append("prediction_abs_error_above_threshold")
    values: dict[str, JsonValue] = {
        "native_backend_id": native_backend_id,
        "candidate_backend_id": candidate_backend_id,
        "native_rows": len(native.predictions),
        "candidate_rows": len(candidate.predictions),
        "overlap_rows": overlap_rows,
        "overlap_ratio": overlap_ratio,
        "prediction_pearson": pearson,
        "prediction_spearman": spearman,
        "prediction_mae": mae,
        "prediction_max_abs_error": max_abs_error,
        "metric_abs_deltas": metric_deltas,
    }
    return values, failures, "promotable" if not failures else "rejected"


def _backend_identity_result(
    native: _EvaluationBundle,
    candidate: _EvaluationBundle,
    *,
    native_backend_id: str,
    candidate_backend_id: str,
) -> tuple[dict[str, JsonValue], list[str], str] | None:
    failures: list[str] = []
    if native_backend_id != "native":
        failures.append("expected_native_baseline")
    if candidate_backend_id != "qlib":
        failures.append("expected_qlib_candidate")
    if not failures:
        return None
    return (
        {
            "native_backend_id": native_backend_id,
            "candidate_backend_id": candidate_backend_id,
            "native_rows": len(native.predictions),
            "candidate_rows": len(candidate.predictions),
            "overlap_rows": 0,
            "overlap_ratio": 0.0,
        },
        failures,
        "non-comparable",
    )


def _comparison_payload(
    native_handle: ArtifactHandle,
    candidate_handle: ArtifactHandle,
    thresholds: BackendPromotionThresholds,
) -> dict[str, JsonValue]:
    native = _load_evaluation(native_handle)
    candidate = _load_evaluation(candidate_handle)
    comparison, failures, status = _comparison_values(native, candidate, thresholds)
    return strict_json_mapping(
        {
            "schema": COMPARISON_SCHEMA,
            "source_artifacts": {
                "native": native_handle.to_metadata(),
                "candidate": candidate_handle.to_metadata(),
            },
            "thresholds": thresholds.to_metadata(),
            "comparison": comparison,
            "decision": {"status": status, "failures": failures},
        },
        field="backend_comparison",
    )


def compare_backend_evaluations(
    native_artifact: ArtifactHandle,
    candidate_artifact: ArtifactHandle,
    *,
    thresholds: BackendPromotionThresholds,
    output_path: str | Path,
) -> ArtifactHandle:
    payload = _comparison_payload(native_artifact, candidate_artifact, thresholds)
    report_path = write_strict_json(output_path, payload)
    return ArtifactHandle.from_file(
        report_path,
        artifact_type="backend_comparison",
        schema_version="v1",
    )


def _comparison_report_handle(report: str | Path | ArtifactHandle) -> ArtifactHandle:
    handle = (
        report
        if isinstance(report, ArtifactHandle)
        else ArtifactHandle.from_file(
            report,
            artifact_type="backend_comparison",
            schema_version="v1",
        )
    )
    if handle.artifact_type != "backend_comparison" or handle.schema_version != "v1":
        raise ArtifactIntegrityError("Expected a backend_comparison v1 artifact handle")
    handle.verify()
    return handle


def replay_backend_comparison(report: str | Path | ArtifactHandle) -> dict[str, Any]:
    """Verify every source digest and reproduce a stored promotion decision."""

    report_path = _comparison_report_handle(report).verify()
    stored = load_json_mapping(report_path)
    if stored.get("schema") != COMPARISON_SCHEMA:
        raise ArtifactIntegrityError(f"Unsupported comparison schema: {stored.get('schema')!r}")
    source_artifacts = stored.get("source_artifacts")
    thresholds_payload = stored.get("thresholds")
    if not isinstance(source_artifacts, dict) or not isinstance(thresholds_payload, dict):
        raise ArtifactIntegrityError("Comparison report is missing sources or thresholds")
    native_raw = source_artifacts.get("native")
    candidate_raw = source_artifacts.get("candidate")
    if not isinstance(native_raw, dict) or not isinstance(candidate_raw, dict):
        raise ArtifactIntegrityError("Comparison source handles must be mappings")
    native_handle = ArtifactHandle.from_metadata(native_raw)
    candidate_handle = ArtifactHandle.from_metadata(candidate_raw)
    thresholds = BackendPromotionThresholds.from_mapping(thresholds_payload)
    replayed = _comparison_payload(native_handle, candidate_handle, thresholds)
    if strict_json_mapping(stored, field="stored_report") != replayed:
        raise ArtifactIntegrityError("Stored backend comparison decision does not replay exactly")
    result = dict(stored)
    result["replay_verified"] = True
    return result


def write_backend_comparison_replay_receipt(
    report: str | Path | ArtifactHandle,
    *,
    output_path: str | Path,
) -> ArtifactHandle:
    """Persist a strict, deterministic receipt for a fully replayed comparison."""

    source_handle = _comparison_report_handle(report)
    replayed = replay_backend_comparison(source_handle)
    decision = replayed.get("decision")
    comparison = replayed.get("comparison")
    thresholds = replayed.get("thresholds")
    if not all(isinstance(value, dict) for value in (decision, comparison, thresholds)):
        raise ArtifactIntegrityError("Comparison report is missing replay receipt fields")
    payload = strict_json_mapping(
        {
            "schema": REPLAY_RECEIPT_SCHEMA,
            "source_schema": COMPARISON_SCHEMA,
            "source_report": source_handle.to_metadata(),
            "source_report_sha256": source_handle.sha256,
            "verification_method": "artifact-digest-and-decision-replay",
            "replay_verified": True,
            "decision": decision,
            "comparison": comparison,
            "thresholds": thresholds,
        },
        field="backend_comparison_replay_receipt",
    )
    receipt_path = write_strict_json(output_path, payload)
    return ArtifactHandle.from_file(
        receipt_path,
        artifact_type="backend_comparison_replay_receipt",
        schema_version="v1",
    )


__all__ = [
    "COMPARISON_SCHEMA",
    "EVALUATION_SCHEMA",
    "REPLAY_RECEIPT_SCHEMA",
    "BackendPromotionThresholds",
    "compare_backend_evaluations",
    "replay_backend_comparison",
    "write_backend_comparison_replay_receipt",
    "write_backend_evaluation",
]
