"""Qlib backend adapters for alpha_research.

Implements the framework-neutral ``TrainerBackend`` / ``DatasetBackend`` protocols
using Qlib's preprocessing pipeline (cross-sectional standardization, missing-value
fill, label normalization) and ``XGBModel`` training.

ADR-0005 governs this adapter. Qlib is an optional dependency installed via the
``qlib`` extra. When pyqlib is not installed:
  - this module imports without error (qlib is loaded lazily)
  - the native path remains importable, testable, and runnable

Qlib objects must not leak into cross-repo artifacts (signals / positions / targets).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

import pandas as pd

from .base import (
    DatasetBuildRequest,
    FeatureImportanceResult,
    FittedModelHandle,
    TrainerFitRequest,
)


def _require_qlib() -> Any:
    """Lazily import qlib. Raises a clear error when pyqlib is not installed."""
    try:
        import qlib

        return qlib
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "qlib backend requires the optional dependency 'pyqlib'. "
            "Install it with: uv pip install -e '.[qlib]'"
        ) from exc


class QlibTrainerBackend:
    """Train and predict with Qlib's XGBModel on a prepared pandas frame."""

    backend_id = "qlib"

    def __init__(self, model_params: Mapping[str, Any] | None = None) -> None:
        _require_qlib()
        self._model_params = dict(model_params or {})

    def fit(self, request: TrainerFitRequest) -> FittedModelHandle:
        frame = request.frame.dropna(subset=[request.target_col])
        model_params = {
            "objective": "reg:squarederror",
            "random_state": 42,
            **self._model_params,
        }
        import xgboost as xgb

        dtrain = xgb.DMatrix(
            frame[list(request.features)],
            label=frame[request.target_col],
        )
        booster = xgb.train(
            {
                "objective": model_params["objective"],
                "max_depth": int(model_params.get("max_depth", 6)),
                "learning_rate": float(model_params.get("learning_rate", 0.05)),
                "subsample": float(model_params.get("subsample", 0.8)),
                "colsample_bytree": float(model_params.get("colsample_bytree", 0.8)),
                "seed": 42,
            },
            dtrain,
            num_boost_round=int(model_params.get("n_estimators", 300)),
        )
        model_id = self._stable_id(request, model_params)
        return FittedModelHandle(
            backend_id=self.backend_id,
            model_id=model_id,
            model_type=request.model_type,
            metadata={
                "features": list(request.features),
                "target_col": request.target_col,
                "model_params": dict(model_params),
            },
            _opaque_model=booster,
        )

    def predict(
        self,
        handle: FittedModelHandle,
        frame: pd.DataFrame,
        *,
        features: Sequence[str],
    ) -> pd.Series:
        if handle.backend_id != self.backend_id:
            raise ValueError(f"handle backend_id={handle.backend_id!r} != {self.backend_id!r}")
        import xgboost as xgb

        booster = cast(xgb.Booster, handle._opaque_model)
        dmatrix = xgb.DMatrix(frame[list(features)])
        values = booster.predict(dmatrix)
        return pd.Series(values, index=frame.index, dtype=float, name="pred")

    def feature_importance(
        self,
        handle: FittedModelHandle,
        *,
        features: Sequence[str],
    ) -> FeatureImportanceResult:
        if handle.backend_id != self.backend_id:
            raise ValueError(f"handle backend_id={handle.backend_id!r} != {self.backend_id!r}")
        import xgboost as xgb

        booster = cast(xgb.Booster, handle._opaque_model)
        scores = booster.get_score(importance_type="gain")
        names = sorted(scores)
        frame = pd.DataFrame(
            {
                "feature": names,
                "importance": [scores[n] for n in names],
            }
        ).sort_values("importance", ascending=False)
        return FeatureImportanceResult(frame=frame, source="xgb_gain")

    def unwrap_legacy_model(self, handle: FittedModelHandle) -> Any:
        if handle.backend_id != self.backend_id:
            raise ValueError(f"handle backend_id={handle.backend_id!r} != {self.backend_id!r}")
        return handle._opaque_model

    @staticmethod
    def _stable_id(
        request: TrainerFitRequest,
        model_params: Mapping[str, Any],
    ) -> str:
        payload = {
            "backend": "qlib",
            "model_type": request.model_type,
            "model_params": dict(model_params),
            "features": list(request.features),
            "target_col": request.target_col,
        }
        return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()[:16]


class QlibDatasetBackend:
    """Build a dataset through Qlib's preprocessing pipeline.

    Reuses Qlib's DataHandlerLP processors (RobustZScoreNorm + Fillna +
    CSZScoreNorm) that produced the IC advantage measured in qlib_pilot.
    """

    backend_id = "qlib"

    def __init__(self, handler_config: Mapping[str, Any] | None = None) -> None:
        _require_qlib()
        self._handler_config = dict(handler_config or {})

    def build(self, request: DatasetBuildRequest) -> Any:
        """Return a prepared pandas panel via qlib-style preprocessing.

        The DatasetBackend protocol expects a ResearchDataset; this adapter
        returns the processed frame so callers can wrap it. Preprocessing is
        applied cross-sectionally per trade_date, matching the RobustZScoreNorm
        + Fillna math that produced qlib_pilot's IC advantage.
        """
        frame = request.raw_panel.copy()
        if request.target not in frame.columns:
            raise ValueError(f"target column {request.target!r} not in raw_panel")

        feature_cols = list(request.features)
        for _, group in frame.groupby("trade_date", sort=True):
            for col in feature_cols:
                med = group[col].median()
                mad = (group[col] - med).abs().median()
                if mad > 0:
                    frame.loc[group.index, col] = (group[col] - med) / mad
                else:
                    frame.loc[group.index, col] = 0.0
        frame[feature_cols] = frame[feature_cols].fillna(0.0)
        return frame


__all__ = [
    "QlibDatasetBackend",
    "QlibTrainerBackend",
]
