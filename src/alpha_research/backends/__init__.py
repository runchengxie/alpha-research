import contextlib

from .base import (
    DatasetBackend,
    DatasetBuildRequest,
    ExperimentReceipt,
    ExperimentRecorder,
    FeatureImportanceResult,
    FittedModelHandle,
    TrainerBackend,
    TrainerFitRequest,
)
from .native import NativeDatasetBackend, NativeTrainerBackend, NullExperimentRecorder

with contextlib.suppress(ImportError):
    # pyqlib not installed; keep module importable per ADR-0005.
    from .qlib import QlibDatasetBackend, QlibTrainerBackend

__all__ = [
    "DatasetBackend",
    "DatasetBuildRequest",
    "ExperimentReceipt",
    "ExperimentRecorder",
    "FeatureImportanceResult",
    "FittedModelHandle",
    "NativeDatasetBackend",
    "NativeTrainerBackend",
    "NullExperimentRecorder",
    "QlibDatasetBackend",
    "QlibTrainerBackend",
    "TrainerBackend",
    "TrainerFitRequest",
]
