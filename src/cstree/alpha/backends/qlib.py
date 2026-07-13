"""Optional Qlib adapters for the framework-neutral research backend ports.

This module intentionally contains no top-level :mod:`qlib` import.  Importing
``cstree.alpha`` and running the native workflow therefore does not require the
optional runtime. Qlib objects remain in ``_qlib_runtime`` and in a private
in-process registry owned by :class:`QlibTrainerBackend`.
"""

from __future__ import annotations

import hashlib
import json
import uuid
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pandas as pd
from pandas.testing import assert_frame_equal

from ..research_dataset import ResearchDataset
from .base import (
    DatasetBuildRequest,
    ExperimentReceipt,
    FeatureImportanceResult,
    FittedModelHandle,
    TrainerFitRequest,
)
from .native import NativeDatasetBackend


class QlibIntegrationUnavailableError(ImportError):
    """Raised when a Qlib adapter is used without the optional dependency."""


class QlibDatasetParityError(ValueError):
    """Raised when Qlib changes a canonical raw/infer/learn frame."""


class QlibModelUnsupportedError(ValueError):
    """Raised when no explicit Qlib model mapping exists for a model type."""


def _import_runtime_module() -> ModuleType:
    from . import _qlib_runtime

    return _qlib_runtime


def _load_runtime() -> ModuleType:
    try:
        return _import_runtime_module()
    except ImportError as exc:
        missing = exc.name or ""
        if missing == "qlib" or missing.startswith("qlib."):
            raise QlibIntegrationUnavailableError(
                "Qlib is optional. Install alpha-research[qlib] before using a Qlib backend."
            ) from exc
        raise


def _stable_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(
        "Qlib backend metadata must remain framework-neutral and JSON-compatible; "
        f"got {type(value).__name__}."
    )


def _frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(repr(frame.index.names).encode("utf-8"))
    digest.update(repr(frame.columns.tolist()).encode("utf-8"))
    digest.update(repr([str(dtype) for dtype in frame.dtypes]).encode("utf-8"))
    digest.update(
        frame.to_csv(
            index=True,
            date_format="%Y-%m-%dT%H:%M:%S.%f%z",
            float_format="%.17g",
            lineterminator="\n",
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _to_qlib_frame(
    frame: pd.DataFrame,
    *,
    date_col: str,
    symbol_col: str,
    feature_cols: Sequence[str],
    label_cols: Sequence[str],
) -> pd.DataFrame:
    required = [date_col, symbol_col, *feature_cols, *label_cols]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot map canonical frame to Qlib; missing columns: {missing}")
    if frame.empty:
        raise ValueError("Cannot map an empty canonical frame to Qlib.")

    dates = pd.to_datetime(frame[date_col], errors="raise")
    if bool(dates.isna().any()):
        raise ValueError(f"Qlib datetime column {date_col!r} contains null values.")
    instruments = frame[symbol_col].astype(str)
    index = pd.MultiIndex.from_arrays(
        [dates.to_numpy(), instruments.to_numpy()],
        names=["datetime", "instrument"],
    )
    if index.has_duplicates:
        duplicated = [
            value
            for value, is_duplicate in zip(index.tolist(), index.duplicated(), strict=True)
            if is_duplicate
        ][:3]
        raise ValueError(
            "Qlib frames require unique (datetime, instrument) observations; "
            f"duplicates include {duplicated}."
        )

    feature_names = list(dict.fromkeys(str(column) for column in feature_cols))
    label_names = [
        column
        for column in dict.fromkeys(str(column) for column in label_cols)
        if column not in feature_names
    ]
    excluded = {date_col, symbol_col, *feature_names, *label_names}
    meta_names = [str(column) for column in frame.columns if str(column) not in excluded]
    ordered = [*feature_names, *label_names, *meta_names]
    groups = [
        *(["feature"] * len(feature_names)),
        *(["label"] * len(label_names)),
        *(["meta"] * len(meta_names)),
    ]
    mapped = cast(pd.DataFrame, frame[ordered].copy())
    mapped.index = index
    mapped.columns = pd.MultiIndex.from_arrays([groups, ordered])
    return mapped.sort_index(kind="mergesort")


@dataclass(frozen=True)
class QlibFrameSet:
    """Framework-neutral holder for Qlib-shaped raw/infer/learn frames."""

    raw: pd.DataFrame
    infer: pd.DataFrame
    learn: pd.DataFrame
    date_col: str
    symbol_col: str

    @classmethod
    def from_research_dataset(cls, dataset: ResearchDataset) -> QlibFrameSet:
        labels = tuple(dict.fromkeys((dataset.target_col, dataset.train_target_col)))

        def convert(frame: pd.DataFrame) -> pd.DataFrame:
            return _to_qlib_frame(
                frame,
                date_col=dataset.date_col,
                symbol_col=dataset.symbol_col,
                feature_cols=dataset.feature_cols,
                # A derived learning label may not exist at the raw boundary.
                label_cols=tuple(column for column in labels if column in frame.columns),
            )

        return cls(
            # Qlib DK_R starts at the engineered feature/label boundary.  The
            # earlier raw market panel remains source lineage, not model input.
            raw=convert(dataset.raw_feature_label),
            infer=convert(dataset.infer_frame),
            learn=convert(dataset.learn_frame),
            date_col=dataset.date_col,
            symbol_col=dataset.symbol_col,
        )

    def digests(self) -> dict[str, str]:
        return {
            "raw": _frame_digest(self.raw),
            "infer": _frame_digest(self.infer),
            "learn": _frame_digest(self.learn),
        }


class QlibDatasetBackend:
    """Expose canonical research frames through Qlib DataHandlerLP semantics.

    Domain processing, PIT filtering, and leakage controls remain owned by the
    native alpha dataset builder.  This adapter verifies that Qlib's ``DK_R``,
    ``DK_I`` and ``DK_L`` views preserve those already-governed frames.
    """

    backend_id = "qlib"

    def __init__(self, *, source_metadata: Mapping[str, Any] | None = None) -> None:
        self._runtime = _load_runtime()
        self._source_metadata = _json_value(dict(source_metadata or {}))

    def build(self, request: DatasetBuildRequest) -> ResearchDataset:
        canonical = NativeDatasetBackend().build(request)
        frames = QlibFrameSet.from_research_dataset(canonical)
        handler = self._runtime.create_canonical_handler(frames)
        actual_frames = {
            "raw": self._runtime.fetch_handler_frame(handler, "raw"),
            "infer": self._runtime.fetch_handler_frame(handler, "infer"),
            "learn": self._runtime.fetch_handler_frame(handler, "learn"),
        }
        for name, expected in {
            "raw": frames.raw,
            "infer": frames.infer,
            "learn": frames.learn,
        }.items():
            try:
                assert_frame_equal(actual_frames[name], expected, check_exact=True)
            except AssertionError as exc:
                raise QlibDatasetParityError(
                    f"Qlib {name} frame diverged from the canonical alpha frame: {exc}"
                ) from exc

        metadata = dict(canonical.metadata)
        metadata["research_backend"] = {
            "name": "qlib",
            "package": "pyqlib",
            "version": self._runtime.runtime_version(),
            "adapter": "cstree.alpha.backends.qlib",
            "adapter_version": 1,
            "data_keys": {
                "raw_feature_label": "DK_R",
                "infer_frame": "DK_I",
                "learn_frame": "DK_L",
            },
            "canonical_preprocessing_owner": "alpha-research",
            "frame_sha256": frames.digests(),
            "source": self._source_metadata,
        }
        return replace(canonical, metadata=metadata)

    def as_qlib_dataset(
        self,
        dataset: ResearchDataset,
        *,
        segments: Mapping[str, tuple[object | None, object | None]],
    ) -> object:
        """Create an in-process Qlib DatasetH without placing it in artifacts."""

        frames = QlibFrameSet.from_research_dataset(dataset)
        return self._runtime.create_dataset(frames, dict(segments))


class QlibTrainerBackend:
    """Adapt explicitly supported Qlib models to ``TrainerBackend``.

    Canonical ``ridge`` maps to Qlib's ``LinearModel(estimator='ridge')`` and
    ``xgb_regressor`` maps to Qlib's ``XGBModel``. Unsupported mappings fail
    explicitly instead of silently delegating to the native backend.
    """

    backend_id = "qlib"

    def __init__(self) -> None:
        self._runtime = _load_runtime()
        self._models: dict[str, object] = {}

    def fit(self, request: TrainerFitRequest) -> FittedModelHandle:
        supported = {"ridge", "xgb_regressor"}
        if request.model_type not in supported:
            raise QlibModelUnsupportedError(
                "QlibTrainerBackend supports model_type in "
                f"{sorted(supported)!r}; "
                f"got {request.model_type!r}."
            )
        opaque_model, runtime_metadata = self._runtime.fit_model(request)
        model_id = _stable_id(
            {
                "backend": self.backend_id,
                "model_type": request.model_type,
                "model_params": dict(request.model_params),
                "features": list(request.features),
                "target_col": request.target_col,
            }
        )
        runtime_ref = uuid.uuid4().hex
        self._models[runtime_ref] = opaque_model
        handle = FittedModelHandle(
            backend_id=self.backend_id,
            model_id=model_id,
            model_type=request.model_type,
            metadata={
                "features": list(request.features),
                "target_col": request.target_col,
                "runtime": _json_value(runtime_metadata),
            },
            runtime_ref=runtime_ref,
        )
        weakref.finalize(handle, self._models.pop, runtime_ref, None)
        return handle

    def predict(
        self,
        handle: FittedModelHandle,
        frame: pd.DataFrame,
        *,
        features: Sequence[str],
    ) -> pd.Series:
        opaque_model = self._require_model(handle)
        return self._runtime.predict_model(opaque_model, frame, tuple(features))

    def feature_importance(
        self,
        handle: FittedModelHandle,
        *,
        features: Sequence[str],
    ) -> FeatureImportanceResult:
        opaque_model = self._require_model(handle)
        values, source = self._runtime.feature_importance(opaque_model, tuple(features))
        return FeatureImportanceResult(
            frame=pd.DataFrame({"feature": list(features), "importance": values}).sort_values(
                "importance", ascending=False
            ),
            source=str(source),
        )

    def unwrap_legacy_model(self, handle: FittedModelHandle) -> object:
        return self._runtime.unwrap_model(self._require_model(handle))

    def _require_model(self, handle: FittedModelHandle) -> object:
        if handle.backend_id != self.backend_id:
            raise ValueError(
                f"{self.backend_id} backend cannot use {handle.backend_id!r} model handle"
            )
        if handle.runtime_ref is None or handle.runtime_ref not in self._models:
            raise ValueError("Qlib model handle is not active in this backend process")
        return self._models[handle.runtime_ref]


class QlibExperimentRecorder:
    """Record framework-neutral receipts through Qlib's MLflow manager."""

    backend_id = "qlib"

    def __init__(self, tracking_uri: str | Path) -> None:
        value = str(tracking_uri)
        if "://" in value:
            resolved_uri = value
        else:
            tracking_root = Path(value).resolve()
            tracking_root.mkdir(parents=True, exist_ok=True)
            resolved_uri = f"sqlite:///{tracking_root / 'qlib-mlflow.db'}"
        self._runtime = _load_runtime().QlibRecorderRuntime(resolved_uri)

    def start(
        self,
        *,
        experiment_name: str,
        run_name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentReceipt:
        record = self._runtime.start(
            experiment_name=experiment_name,
            run_name=run_name,
            metadata=_json_value(dict(metadata or {})),
        )
        return ExperimentReceipt(
            backend_id=self.backend_id,
            experiment_id=str(record["experiment_id"]),
            run_id=str(record["run_id"]),
            metadata=_json_value(record["metadata"]),
        )

    def log_metrics(self, receipt: ExperimentReceipt, metrics: Mapping[str, float]) -> None:
        self._validate_receipt(receipt)
        self._runtime.log_metrics(receipt.run_id, dict(metrics))

    def close(self, receipt: ExperimentReceipt, *, status: str) -> None:
        self._validate_receipt(receipt)
        self._runtime.close(receipt.run_id, status=status)

    def _validate_receipt(self, receipt: ExperimentReceipt) -> None:
        if receipt.backend_id != self.backend_id:
            raise ValueError(
                f"{self.backend_id} recorder cannot use {receipt.backend_id!r} receipt"
            )


__all__ = [
    "QlibDatasetBackend",
    "QlibDatasetParityError",
    "QlibExperimentRecorder",
    "QlibFrameSet",
    "QlibIntegrationUnavailableError",
    "QlibModelUnsupportedError",
    "QlibTrainerBackend",
]
