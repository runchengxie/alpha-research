from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from ..research_dataset import ResearchDataset


@dataclass(frozen=True)
class DatasetBuildRequest:
    raw_panel: pd.DataFrame
    modeling_state: Mapping[str, Any]
    backtest_pricing_frame: pd.DataFrame
    features: tuple[str, ...]
    target: str
    train_target: str
    missing_fill_features: tuple[str, ...]
    feature_missing_method: str
    feature_missing_add_indicators: bool
    winsorize_pct: float | None
    cs_method: str
    cs_winsorize_pct: float | None
    train_target_transform: str
    train_target_group_cols: tuple[str, ...] | None
    universe_by_date_applied: bool
    sample_on_rebalance_dates: bool
    min_symbols_per_date: int


@runtime_checkable
class DatasetBackend(Protocol):
    backend_id: str

    def build(self, request: DatasetBuildRequest) -> ResearchDataset: ...


@dataclass(frozen=True)
class TrainerFitRequest:
    frame: pd.DataFrame
    model_type: str
    model_params: Mapping[str, Any]
    features: tuple[str, ...]
    target_col: str
    date_col: str = "trade_date"
    symbol_col: str = "symbol"
    sample_weight: Any | None = None


@dataclass(frozen=True)
class FittedModelHandle:
    backend_id: str
    model_id: str
    model_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _opaque_model: object = field(repr=False, compare=False, default=None)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "model_id": self.model_id,
            "model_type": self.model_type,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FeatureImportanceResult:
    frame: pd.DataFrame
    source: str


@runtime_checkable
class TrainerBackend(Protocol):
    backend_id: str

    def fit(self, request: TrainerFitRequest) -> FittedModelHandle: ...

    def predict(
        self,
        handle: FittedModelHandle,
        frame: pd.DataFrame,
        *,
        features: Sequence[str],
    ) -> pd.Series: ...

    def feature_importance(
        self,
        handle: FittedModelHandle,
        *,
        features: Sequence[str],
    ) -> FeatureImportanceResult: ...

    def unwrap_legacy_model(self, handle: FittedModelHandle) -> object: ...


@dataclass(frozen=True)
class ExperimentReceipt:
    backend_id: str
    experiment_id: str
    run_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class ExperimentRecorder(Protocol):
    backend_id: str

    def start(
        self,
        *,
        experiment_name: str,
        run_name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentReceipt: ...

    def log_metrics(self, receipt: ExperimentReceipt, metrics: Mapping[str, float]) -> None: ...

    def close(self, receipt: ExperimentReceipt, *, status: str) -> None: ...
