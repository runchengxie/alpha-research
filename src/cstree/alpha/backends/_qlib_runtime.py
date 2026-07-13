"""Qlib runtime implementation kept behind the optional adapter boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import qlib
from qlib.contrib.model.linear import LinearModel
from qlib.contrib.model.xgboost import XGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader
from qlib.data.dataset.weight import Reweighter
from qlib.workflow.expm import MLflowExpManager
from qlib.workflow.recorder import Recorder

from .base import TrainerFitRequest
from .qlib import QlibFrameSet, _to_qlib_frame


def runtime_version() -> str:
    return str(getattr(qlib, "__version__", "unknown"))


class _CanonicalFrameHandler(DataHandlerLP):
    def __init__(self, frames: QlibFrameSet) -> None:
        # ``init_data=False`` avoids loading or re-processing the canonical
        # frames.  DK_R/DK_I/DK_L are populated from the alpha-owned lifecycle.
        super().__init__(data_loader=StaticDataLoader(frames.raw), init_data=False)
        self._data = frames.raw.copy()
        self._infer = frames.infer.copy()
        self._learn = frames.learn.copy()


def create_canonical_handler(frames: QlibFrameSet) -> DataHandlerLP:
    return _CanonicalFrameHandler(frames)


def fetch_handler_frame(handler: DataHandlerLP, data_key: str) -> pd.DataFrame:
    key_by_name: dict[str, Literal["raw", "infer", "learn"]] = {
        "raw": DataHandlerLP.DK_R,
        "infer": DataHandlerLP.DK_I,
        "learn": DataHandlerLP.DK_L,
    }
    key = key_by_name[data_key]
    stored = handler._get_df_by_key(key)
    groups = list(dict.fromkeys(stored.columns.get_level_values(0)))
    return handler.fetch(col_set=groups, data_key=key)


def create_dataset(
    frames: QlibFrameSet,
    segments: Mapping[str, tuple[object | None, object | None]],
) -> DatasetH:
    return DatasetH(handler=create_canonical_handler(frames), segments=dict(segments))


class _StaticReweighter(Reweighter):
    def __init__(self, weights: pd.Series) -> None:
        self._weights = weights

    def reweight(self, data: object) -> object:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Qlib sample reweighting requires a pandas DataFrame.")
        values = self._weights.reindex(data.index)
        if bool(values.isna().any()):
            raise ValueError("Qlib sample weights do not align with the training frame.")
        return values


@dataclass(frozen=True)
class _QlibFittedModel:
    model: Any
    model_type: str
    features: tuple[str, ...]
    target_col: str
    date_col: str
    symbol_col: str


def _segments(frame: pd.DataFrame, name: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    dates = pd.DatetimeIndex(frame.index.get_level_values("datetime"))
    return {
        name: (
            cast(pd.Timestamp, dates.min()),
            cast(pd.Timestamp, dates.max()),
        )
    }


def _model_frame(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    target_col: str,
    date_col: str,
    symbol_col: str,
) -> pd.DataFrame:
    return _to_qlib_frame(
        frame,
        date_col=date_col,
        symbol_col=symbol_col,
        feature_cols=features,
        label_cols=(target_col,),
    )


def _ridge_params(params: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"alpha", "fit_intercept", "random_state"}
    unsupported = sorted(set(params) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported Qlib ridge parameters: {unsupported}")
    return {
        "estimator": "ridge",
        "alpha": float(params.get("alpha", 1.0)),
        "fit_intercept": bool(params.get("fit_intercept", True)),
    }


def _xgb_params(params: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    values = dict(params)
    rounds = int(values.pop("n_estimators", 300))
    if rounds <= 0:
        raise ValueError("Qlib xgb_regressor n_estimators must be positive.")
    if "early_stopping_rounds" in values:
        raise ValueError(
            "Qlib xgb_regressor does not accept early_stopping_rounds through model_params; "
            "validation ownership remains with alpha-research CV."
        )
    if "random_state" in values:
        values["seed"] = values.pop("random_state")
    return values, rounds


def fit_model(request: TrainerFitRequest) -> tuple[_QlibFittedModel, dict[str, Any]]:
    symbol_col = request.symbol_col
    qlib_frame = _model_frame(
        request.frame,
        features=request.features,
        target_col=request.target_col,
        date_col=request.date_col,
        symbol_col=symbol_col,
    )
    train_segment = _segments(qlib_frame, "train")["train"]
    segments = {"train": train_segment}
    if request.model_type == "ridge":
        model: Any = LinearModel(**_ridge_params(request.model_params))
        model_mapping = "ridge->LinearModel(estimator=ridge)"
        fit_kwargs: dict[str, Any] = {}
    elif request.model_type == "xgb_regressor":
        model_params, rounds = _xgb_params(request.model_params)
        model = XGBModel(**model_params)
        model_mapping = "xgb_regressor->XGBModel"
        # Qlib's XGBModel requires a valid segment. It mirrors train only for
        # evaluation logging; early stopping is disabled and all train rows are
        # still fitted. Alpha's outer CV remains the validation authority.
        segments["valid"] = train_segment
        fit_kwargs = {
            "num_boost_round": rounds,
            "early_stopping_rounds": None,
            "verbose_eval": False,
            "evals_result": {},
        }
    else:
        raise ValueError(f"Unsupported Qlib model type: {request.model_type!r}")
    dataset = DatasetH(handler=DataHandlerLP.from_df(qlib_frame), segments=segments)
    reweighter = None
    if request.sample_weight is not None:
        raw_weights = np.asarray(request.sample_weight, dtype=float).reshape(-1)
        if raw_weights.size != len(request.frame):
            raise ValueError(
                "Qlib sample_weight must match the number of training rows: "
                f"{raw_weights.size} != {len(request.frame)}."
            )
        row_index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(request.frame[request.date_col], errors="raise").to_numpy(),
                request.frame[symbol_col].astype(str).to_numpy(),
            ],
            names=["datetime", "instrument"],
        )
        weights = pd.Series(raw_weights, index=row_index).sort_index(kind="mergesort")
        reweighter = _StaticReweighter(weights)
    if reweighter is None:
        model.fit(dataset, **fit_kwargs)
    else:
        model.fit(dataset, reweighter=reweighter, **fit_kwargs)
    fitted = _QlibFittedModel(
        model=model,
        model_type=request.model_type,
        features=tuple(request.features),
        target_col=request.target_col,
        date_col=request.date_col,
        symbol_col=symbol_col,
    )
    return fitted, {
        "name": "qlib",
        "package": "pyqlib",
        "version": runtime_version(),
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "model_mapping": model_mapping,
    }


def predict_model(
    fitted: _QlibFittedModel,
    frame: pd.DataFrame,
    features: Sequence[str],
) -> pd.Series:
    if tuple(features) != fitted.features:
        raise ValueError(
            f"Prediction features {tuple(features)!r} do not match fitted features "
            f"{fitted.features!r}."
        )
    prediction_input = frame.copy()
    if fitted.target_col not in prediction_input.columns:
        prediction_input[fitted.target_col] = np.nan
    qlib_frame = _model_frame(
        prediction_input,
        features=features,
        target_col=fitted.target_col,
        date_col=fitted.date_col,
        symbol_col=fitted.symbol_col,
    )
    dataset = DatasetH(
        handler=DataHandlerLP.from_df(qlib_frame),
        segments=_segments(qlib_frame, "test"),
    )
    prediction = fitted.model.predict(dataset, segment="test")
    row_dates = pd.to_datetime(frame[fitted.date_col], errors="raise")
    row_symbols = frame[fitted.symbol_col].astype(str)
    original_order = pd.MultiIndex.from_arrays(
        [row_dates.to_numpy(), row_symbols.to_numpy()],
        names=["datetime", "instrument"],
    )
    aligned = prediction.reindex(original_order)
    if bool(aligned.isna().any()) and not bool(prediction.isna().any()):
        raise ValueError("Qlib predictions could not be aligned to the canonical frame.")
    return pd.Series(aligned.to_numpy(dtype=float), index=frame.index, name="pred")


def feature_importance(
    fitted: _QlibFittedModel,
    features: Sequence[str],
) -> tuple[np.ndarray, str]:
    if tuple(features) != fitted.features:
        raise ValueError("Feature-importance columns do not match the fitted Qlib model.")
    if fitted.model_type == "ridge":
        coefficients = np.asarray(fitted.model.coef_, dtype=float).reshape(-1)
        if coefficients.size != len(features):
            raise ValueError("Qlib model returned an unexpected number of coefficients.")
        return np.abs(coefficients), "qlib_linear_coef_abs"
    if fitted.model_type == "xgb_regressor":
        score = fitted.model.get_feature_importance()
        values = np.array([float(score.get(f"f{idx}", 0.0)) for idx in range(len(features))])
        return values, "qlib_xgboost_weight"
    raise ValueError(f"Unsupported Qlib model type: {fitted.model_type!r}")


def unwrap_model(fitted: _QlibFittedModel) -> object:
    return fitted.model


def _param_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class QlibRecorderRuntime:
    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self._manager = MLflowExpManager(tracking_uri, "alpha-research")
        self._active_run_id: str | None = None

    def start(
        self,
        *,
        experiment_name: str,
        run_name: str,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self._active_run_id is not None:
            raise RuntimeError("QlibExperimentRecorder supports one active run per instance.")
        experiment = self._manager.start_exp(
            experiment_name=experiment_name,
            recorder_name=run_name,
        )
        recorder = experiment.active_recorder
        if recorder is None or recorder.id is None:
            self._manager.end_exp(Recorder.STATUS_FA)
            raise RuntimeError("Qlib did not create an active experiment recorder.")
        if metadata:
            params = {str(key): _param_value(value) for key, value in metadata.items()}
            recorder.log_params(**params)
        self._active_run_id = str(recorder.id)
        return {
            "experiment_id": str(experiment.id),
            "run_id": self._active_run_id,
            "metadata": {
                "tracking_uri": self.tracking_uri,
                "run_name": run_name,
                "runtime": {
                    "name": "qlib",
                    "package": "pyqlib",
                    "version": runtime_version(),
                },
            },
        }

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        recorder = self._active_recorder(run_id)
        recorder.log_metrics(**{str(key): float(value) for key, value in metrics.items()})

    def close(self, run_id: str, *, status: str) -> None:
        self._active_recorder(run_id)
        normalized = status.strip().lower()
        if normalized in {"completed", "complete", "finished", "succeeded", "success"}:
            recorder_status = Recorder.STATUS_FI
        elif normalized in {"failed", "failure", "error"}:
            recorder_status = Recorder.STATUS_FA
        else:
            raise ValueError(f"Unsupported Qlib recorder close status: {status!r}")
        self._manager.end_exp(recorder_status)
        self._active_run_id = None

    def _active_recorder(self, run_id: str) -> Recorder:
        if self._active_run_id != run_id:
            raise ValueError(f"Qlib recorder run {run_id!r} is not active.")
        experiment = self._manager.active_experiment
        if experiment is None or experiment.active_recorder is None:
            raise RuntimeError("Qlib experiment manager has no active recorder.")
        return experiment.active_recorder
