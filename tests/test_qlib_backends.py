from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cstree.alpha.backends import (
    DatasetBuildRequest,
    NativeDatasetBackend,
    NativeTrainerBackend,
    QlibDatasetBackend,
    QlibExperimentRecorder,
    QlibModelUnsupportedError,
    QlibTrainerBackend,
    TrainerFitRequest,
)
from cstree.alpha.signal_artifact import build_signal_artifact_frame

pytest.importorskip("qlib")


def _canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"]),
            "symbol": ["000001.SZ", "600000.SH", "000001.SZ", "600000.SH"],
            "f1": [0.0, 1.0, 2.0, 3.0],
            "future_return": [1.0, 3.0, 5.0, 7.0],
            "close": [10.0, 20.0, 11.0, 21.0],
        }
    )


def _dataset_request() -> DatasetBuildRequest:
    frame = _canonical_frame()
    return DatasetBuildRequest(
        raw_panel=frame.copy(),
        modeling_state={
            "df_features": frame.copy(),
            "df_model_all": frame.copy(),
            "valid_dates": sorted(frame["trade_date"].unique()),
            "dropped_date_counts": {},
        },
        backtest_pricing_frame=frame[["trade_date", "symbol", "close"]].copy(),
        features=("f1",),
        target="future_return",
        train_target="future_return",
        missing_fill_features=(),
        feature_missing_method="none",
        feature_missing_add_indicators=False,
        winsorize_pct=None,
        cs_method="none",
        cs_winsorize_pct=None,
        train_target_transform="none",
        train_target_group_cols=None,
        universe_by_date_applied=True,
        sample_on_rebalance_dates=False,
        min_symbols_per_date=2,
    )


def test_qlib_dataset_backend_preserves_canonical_lifecycle() -> None:
    source = {
        "contract": {"sha256": "abc123"},
        "sources": [{"content_fingerprint": "def456"}],
    }
    backend = QlibDatasetBackend(source_metadata=source)

    dataset = backend.build(_dataset_request())
    native_dataset = NativeDatasetBackend().build(_dataset_request())

    assert_frame_equal(dataset.raw_feature_label, _canonical_frame())
    assert_frame_equal(dataset.infer_frame, _canonical_frame())
    assert_frame_equal(dataset.learn_frame, _canonical_frame())
    metadata = dataset.summary()["metadata"]["research_backend"]
    assert metadata["data_keys"] == {
        "raw_feature_label": "DK_R",
        "infer_frame": "DK_I",
        "learn_frame": "DK_L",
    }
    assert metadata["source"] == source
    assert set(metadata["frame_sha256"]) == {"raw", "infer", "learn"}
    assert dataset.processors == native_dataset.processors
    assert dataset.processors[0].name == "universe_by_date_filter"
    json.dumps(dataset.summary())


def test_qlib_dataset_exposes_real_dataset_h_views() -> None:
    backend = QlibDatasetBackend()
    dataset = backend.build(_dataset_request())

    qlib_dataset = backend.as_qlib_dataset(
        dataset,
        segments={"train": ("2026-01-05", "2026-01-06")},
    )
    learn = qlib_dataset.prepare("train", col_set="feature", data_key="learn")

    assert learn.columns.tolist() == ["f1"]
    assert learn.index.names == ["datetime", "instrument"]


def test_qlib_dataset_maps_derived_label_only_at_learn_boundary() -> None:
    request = _dataset_request()
    model_frame = _canonical_frame()
    model_frame["rank_target"] = [0.0, 1.0, 0.0, 1.0]
    request = replace(
        request,
        modeling_state={
            **request.modeling_state,
            "df_model_all": model_frame,
        },
        train_target="rank_target",
        train_target_transform="rank_pct",
    )
    backend = QlibDatasetBackend()

    dataset = backend.build(request)
    qlib_dataset = backend.as_qlib_dataset(
        dataset,
        segments={"train": ("2026-01-05", "2026-01-06")},
    )

    raw_labels = qlib_dataset.prepare("train", col_set="label", data_key="raw")
    learn_labels = qlib_dataset.prepare("train", col_set="label", data_key="learn")
    assert raw_labels.columns.tolist() == ["future_return"]
    assert learn_labels.columns.tolist() == ["future_return", "rank_target"]


def test_qlib_ridge_prediction_matches_native_and_builds_canonical_signal() -> None:
    frame = _canonical_frame().sample(frac=1.0, random_state=7)
    request = TrainerFitRequest(
        frame=frame,
        model_type="ridge",
        model_params={"alpha": 0.0, "fit_intercept": True, "random_state": 42},
        features=("f1",),
        target_col="future_return",
        sample_weight=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    native = NativeTrainerBackend()
    qlib_backend = QlibTrainerBackend()

    native_handle = native.fit(request)
    qlib_handle = qlib_backend.fit(request)
    native_pred = native.predict(native_handle, frame, features=("f1",))
    qlib_pred = qlib_backend.predict(qlib_handle, frame, features=("f1",))

    np.testing.assert_allclose(qlib_pred.to_numpy(), native_pred.to_numpy(), atol=1e-10)
    scored = frame[["trade_date", "symbol"]].copy()
    scored["pred"] = qlib_pred
    signals = build_signal_artifact_frame(
        scored,
        model_version=qlib_handle.model_id,
        feature_set_id="f1",
        signal_direction=1.0,
        eligible_for_backtest=True,
        eligible_for_live=False,
    )
    assert signals["symbol"].tolist() == frame["symbol"].tolist()
    assert qlib_handle.to_metadata()["backend_id"] == "qlib"
    assert "_opaque_model" not in json.dumps(qlib_handle.to_metadata())


def test_qlib_trainer_rejects_unmapped_models() -> None:
    backend = QlibTrainerBackend()
    with pytest.raises(QlibModelUnsupportedError, match="xgb_regressor"):
        backend.fit(
            TrainerFitRequest(
                frame=_canonical_frame(),
                model_type="xgb_ranker",
                model_params={},
                features=("f1",),
                target_col="future_return",
            )
        )


def test_qlib_xgb_regressor_is_deterministic_and_aligned() -> None:
    frame = _canonical_frame().sample(frac=1.0, random_state=11)
    request = TrainerFitRequest(
        frame=frame,
        model_type="xgb_regressor",
        model_params={
            "n_estimators": 5,
            "learning_rate": 0.1,
            "max_depth": 2,
            "objective": "reg:squarederror",
            "random_state": 7,
            "nthread": 1,
        },
        features=("f1",),
        target_col="future_return",
    )
    first = QlibTrainerBackend()
    second = QlibTrainerBackend()

    first_handle = first.fit(request)
    second_handle = second.fit(request)
    first_prediction = first.predict(first_handle, frame, features=("f1",))
    second_prediction = second.predict(second_handle, frame, features=("f1",))
    importance = first.feature_importance(first_handle, features=("f1",))

    np.testing.assert_allclose(first_prediction, second_prediction, atol=0.0)
    assert first_prediction.index.equals(frame.index)
    assert np.isfinite(first_prediction).all()
    assert importance.source == "qlib_xgboost_weight"
    assert importance.frame["feature"].tolist() == ["f1"]


def test_qlib_recorder_returns_neutral_receipt_and_persists_metrics(tmp_path: Path) -> None:
    from mlflow.tracking import MlflowClient

    recorder = QlibExperimentRecorder(tmp_path / "mlruns")
    receipt = recorder.start(
        experiment_name="alpha-backend-test",
        run_name="ridge-parity",
        metadata={"seed": 7, "features": ["f1"]},
    )
    recorder.log_metrics(receipt, {"rank_ic": 0.125})
    recorder.close(receipt, status="completed")

    assert json.loads(json.dumps(receipt.to_metadata()))["backend_id"] == "qlib"
    tracking_uri = receipt.metadata["tracking_uri"]
    run = MlflowClient(tracking_uri=str(tracking_uri)).get_run(receipt.run_id)
    assert run.data.metrics["rank_ic"] == pytest.approx(0.125)
    assert run.info.status == "FINISHED"
