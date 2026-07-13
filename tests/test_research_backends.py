from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from cstree.alpha.backends import (
    ExperimentReceipt,
    FeatureImportanceResult,
    FittedModelHandle,
    NativeTrainerBackend,
    NullExperimentRecorder,
    TrainerFitRequest,
)
from cstree.alpha.train_eval_contracts import (
    TrainEvalFeatureTarget,
    TrainEvalModelSettings,
    TrainEvalSignalSettings,
)
from cstree.alpha.train_eval_fit import fit_model_and_score_train


class RecordingTrainerBackend:
    backend_id = "recording"

    def __init__(self) -> None:
        self.fit_requests: list[TrainerFitRequest] = []
        self.predict_calls = 0

    def fit(self, request: TrainerFitRequest) -> FittedModelHandle:
        self.fit_requests.append(request)
        return FittedModelHandle(
            backend_id=self.backend_id,
            model_id="recording-model",
            model_type=request.model_type,
            metadata={"feature_count": len(request.features)},
            _opaque_model={"framework": "private"},
        )

    def predict(
        self,
        handle: FittedModelHandle,
        frame: pd.DataFrame,
        *,
        features: Sequence[str],
    ) -> pd.Series:
        assert handle.backend_id == self.backend_id
        self.predict_calls += 1
        return pd.Series(frame[list(features)].sum(axis=1), index=frame.index, dtype=float)

    def feature_importance(
        self,
        handle: FittedModelHandle,
        *,
        features: Sequence[str],
    ) -> FeatureImportanceResult:
        assert handle.backend_id == self.backend_id
        return FeatureImportanceResult(
            frame=pd.DataFrame({"feature": list(features), "importance": 1.0}),
            source="recording",
        )

    def unwrap_legacy_model(self, handle: FittedModelHandle) -> object:
        assert handle.backend_id == self.backend_id
        return {"legacy_compatibility": True}


def _model_settings() -> TrainEvalModelSettings:
    return TrainEvalModelSettings(
        model_type="ridge",
        model_params={"alpha": 1.0},
        model_cfg={"type": "ridge", "params": {"alpha": 1.0}},
        sample_weight_mode="none",
        sample_weight_params={},
        n_splits=2,
        embargo_steps=0,
        purge_steps=0,
        cv_purge_mode="gap",
        train_window_mode="full",
        train_window_size=None,
        train_window_unit="dates",
    )


def _signal_settings() -> TrainEvalSignalSettings:
    return TrainEvalSignalSettings(
        signal_direction_mode="fixed",
        signal_direction=1.0,
        min_abs_ic_to_flip=0.0,
        score_postprocess_method="none",
        score_postprocess_columns=[],
        score_postprocess_strength=1.0,
        score_postprocess_min_obs=2,
        report_train_ic=True,
    )


def test_trainer_port_keeps_opaque_model_out_of_backend_metadata() -> None:
    backend = RecordingTrainerBackend()
    train = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "f1": [0.0, 1.0, 2.0, 3.0],
            "target": [0.0, 1.0, 2.0, 3.0],
        }
    )

    result = fit_model_and_score_train(
        train,
        feature_target=TrainEvalFeatureTarget(
            features=["f1"],
            target="target",
            train_target="target",
            price_col="close",
            fundamentals_mcap_col="market_cap",
        ),
        model_settings=_model_settings(),
        signal_settings=_signal_settings(),
        cv_scores_raw=[],
        trainer_backend=backend,
    )

    assert len(backend.fit_requests) == 1
    assert backend.predict_calls == 1
    assert result.model == {"legacy_compatibility": True}
    assert result.model_handle is not None
    metadata = result.model_handle.to_metadata()
    assert metadata == {
        "backend_id": "recording",
        "model_id": "recording-model",
        "model_type": "ridge",
        "metadata": {"feature_count": 1},
    }
    assert "private" not in json.dumps(metadata, sort_keys=True)


def test_native_trainer_backend_preserves_fit_predict_behavior() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "f1": [0.0, 1.0, 2.0],
            "target": [1.0, 3.0, 5.0],
        }
    )
    backend = NativeTrainerBackend()
    handle = backend.fit(
        TrainerFitRequest(
            frame=frame,
            model_type="ridge",
            model_params={"alpha": 0.0},
            features=("f1",),
            target_col="target",
        )
    )

    predictions = backend.predict(handle, frame, features=("f1",))

    np.testing.assert_allclose(predictions.to_numpy(), frame["target"].to_numpy(), atol=1e-8)
    assert handle.to_metadata()["backend_id"] == "native"


def test_experiment_receipts_are_framework_neutral() -> None:
    recorder = NullExperimentRecorder()
    receipt = recorder.start(
        experiment_name="daily-watch",
        run_name="run-001",
        metadata={"seed": 7},
    )

    assert isinstance(receipt, ExperimentReceipt)
    assert json.loads(json.dumps(receipt.to_metadata())) == {
        "backend_id": "none",
        "experiment_id": "daily-watch",
        "run_id": "run-001",
        "metadata": {"seed": 7},
    }
    recorder.log_metrics(receipt, {"rank_ic": 0.1})
    recorder.close(receipt, status="completed")
