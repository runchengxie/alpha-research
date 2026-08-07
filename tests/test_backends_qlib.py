from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_research.backends import (
    FeatureImportanceResult,
    FittedModelHandle,
    QlibDatasetBackend,
    QlibTrainerBackend,
    TrainerFitRequest,
)

HAS_QLIB = pytest.importorskip if False else True
try:
    import qlib  # noqa: F401

    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False


def _sample_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=20)
    rows = []
    for inst in range(5):
        f1 = rng.standard_normal(20)
        f2 = rng.standard_normal(20)
        label = 0.5 * f1 + 0.2 * f2 + rng.standard_normal(20) * 0.1
        rows.append(
            pd.DataFrame(
                {
                    "trade_date": dates,
                    "symbol": [f"S{inst}"] * 20,
                    "f1": f1,
                    "f2": f2,
                    "label": label,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _fit_request(frame: pd.DataFrame | None = None) -> TrainerFitRequest:
    frame = frame if frame is not None else _sample_frame()
    return TrainerFitRequest(
        frame=frame,
        model_type="xgb_regressor",
        model_params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
        features=("f1", "f2"),
        target_col="label",
        date_col="trade_date",
    )


def test_module_imports_without_qlib() -> None:
    """backends importable even when pyqlib is absent (ADR-0005)."""
    from alpha_research import backends

    assert hasattr(backends, "QlibTrainerBackend")


def test_trainer_backend_requires_qlib_when_absent() -> None:
    """Constructing the qlib backend without pyqlib raises a clear error."""
    if QLIB_AVAILABLE:
        pytest.skip("pyqlib installed; skipping absence path")
    with pytest.raises(ImportError, match="pyqlib"):
        QlibTrainerBackend()
    with pytest.raises(ImportError, match="pyqlib"):
        QlibDatasetBackend()


@pytest.mark.skipif(not QLIB_AVAILABLE, reason="requires pyqlib")
def test_trainer_fit_and_predict() -> None:
    backend = QlibTrainerBackend()
    handle = backend.fit(_fit_request())
    assert handle.backend_id == "qlib"
    assert isinstance(handle, FittedModelHandle)
    assert handle.model_type == "xgb_regressor"

    frame = _sample_frame()
    pred = backend.predict(handle, frame, features=("f1", "f2"))
    assert isinstance(pred, pd.Series)
    assert len(pred) == len(frame)
    assert np.isfinite(pred).all()


@pytest.mark.skipif(not QLIB_AVAILABLE, reason="requires pyqlib")
def test_trainer_feature_importance() -> None:
    backend = QlibTrainerBackend()
    handle = backend.fit(_fit_request())
    result = backend.feature_importance(handle, features=("f1", "f2"))
    assert isinstance(result, FeatureImportanceResult)
    assert set(result.frame["feature"]) == {"f1", "f2"}
    assert result.frame["importance"].dtype == float


@pytest.mark.skipif(not QLIB_AVAILABLE, reason="requires pyqlib")
def test_dataset_backend_standardizes() -> None:
    from alpha_research.backends import DatasetBuildRequest

    backend = QlibDatasetBackend()
    frame = _sample_frame()
    request = DatasetBuildRequest(
        raw_panel=frame,
        modeling_state={},
        backtest_pricing_frame=frame,
        features=("f1", "f2"),
        target="label",
        train_target="label",
        missing_fill_features=(),
        feature_missing_method="none",
        feature_missing_add_indicators=False,
        winsorize_pct=None,
        cs_method="none",
        cs_winsorize_pct=None,
        train_target_transform="none",
        train_target_group_cols=None,
        universe_by_date_applied=False,
        sample_on_rebalance_dates=False,
        min_symbols_per_date=1,
    )
    # build() is adapter-specific; validate it normalizes the panel.
    out = backend.build(request)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(frame.columns)
    assert out["f1"].notna().all()
