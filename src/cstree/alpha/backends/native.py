from __future__ import annotations

import hashlib
import json
import uuid
import weakref
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ..modeling import build_model, feature_importance_frame, fit_model
from ..research_artifacts import strict_json_mapping
from ..research_dataset import (
    ResearchDataset,
    build_research_dataset_from_modeling_state,
)
from .base import (
    DatasetBuildRequest,
    ExperimentReceipt,
    FeatureImportanceResult,
    FittedModelHandle,
    TrainerFitRequest,
)


def _stable_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        strict_json_mapping(payload, field="model_identity"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class NativeDatasetBackend:
    backend_id = "native"

    def build(self, request: DatasetBuildRequest) -> ResearchDataset:
        return build_research_dataset_from_modeling_state(
            raw_panel=request.raw_panel,
            modeling_state=dict(request.modeling_state),
            backtest_pricing_frame=request.backtest_pricing_frame,
            features=list(request.features),
            target=request.target,
            train_target=request.train_target,
            missing_fill_features=list(request.missing_fill_features),
            feature_missing_method=request.feature_missing_method,
            feature_missing_add_indicators=request.feature_missing_add_indicators,
            winsorize_pct=request.winsorize_pct,
            cs_method=request.cs_method,
            cs_winsorize_pct=request.cs_winsorize_pct,
            train_target_transform=request.train_target_transform,
            train_target_group_cols=(
                list(request.train_target_group_cols)
                if request.train_target_group_cols is not None
                else None
            ),
            universe_by_date_applied=request.universe_by_date_applied,
            sample_on_rebalance_dates=request.sample_on_rebalance_dates,
            min_symbols_per_date=request.min_symbols_per_date,
        )


class NativeTrainerBackend:
    backend_id = "native"

    def __init__(self) -> None:
        self._models: dict[str, object] = {}

    def fit(self, request: TrainerFitRequest) -> FittedModelHandle:
        model = build_model(request.model_type, dict(request.model_params))
        fit_model(
            model,
            request.model_type,
            request.frame,
            features=list(request.features),
            target_col=request.target_col,
            sample_weight=request.sample_weight,
            date_col=request.date_col,
        )
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
        self._models[runtime_ref] = model
        handle = FittedModelHandle(
            backend_id=self.backend_id,
            model_id=model_id,
            model_type=request.model_type,
            metadata={
                "features": list(request.features),
                "target_col": request.target_col,
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
        model: Any = self.unwrap_legacy_model(handle)
        values = model.predict(frame[list(features)])
        return pd.Series(values, index=frame.index, dtype=float, name="pred")

    def feature_importance(
        self,
        handle: FittedModelHandle,
        *,
        features: Sequence[str],
    ) -> FeatureImportanceResult:
        frame, source = feature_importance_frame(
            self.unwrap_legacy_model(handle),
            list(features),
            model_type=handle.model_type,
        )
        return FeatureImportanceResult(frame=frame, source=source)

    def unwrap_legacy_model(self, handle: FittedModelHandle) -> object:
        if handle.backend_id != self.backend_id:
            raise ValueError(
                f"{self.backend_id} backend cannot unwrap {handle.backend_id!r} model handle"
            )
        if handle.runtime_ref is None or handle.runtime_ref not in self._models:
            raise ValueError("model handle is not active in this backend process")
        return self._models[handle.runtime_ref]


class NullExperimentRecorder:
    backend_id = "none"

    def start(
        self,
        *,
        experiment_name: str,
        run_name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentReceipt:
        return ExperimentReceipt(
            backend_id=self.backend_id,
            experiment_id=experiment_name,
            run_id=run_name,
            metadata=dict(metadata or {}),
        )

    def log_metrics(self, receipt: ExperimentReceipt, metrics: Mapping[str, float]) -> None:
        return None

    def close(self, receipt: ExperimentReceipt, *, status: str) -> None:
        return None
