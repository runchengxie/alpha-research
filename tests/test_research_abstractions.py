from __future__ import annotations

import pandas as pd

from alpha_research.research_dataset import ResearchDataset
from alpha_research.research_model import CSTreeModel


def _dataset() -> ResearchDataset:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"]),
            "symbol": ["600519.SH", "000858.SZ", "600519.SH", "000858.SZ"],
            "f1": [1.0, 2.0, 2.0, 1.0],
            "f2": [0.5, 0.3, 0.7, 0.2],
            "target": [0.01, -0.01, 0.02, -0.02],
            "_segment": ["train", "train", "test", "test"],
            "close": [10.0, 11.0, 10.5, 10.8],
        }
    )
    return ResearchDataset(
        raw_panel=frame,
        raw_feature_label=frame,
        infer_frame=frame,
        learn_frame=frame,
        backtest_pricing_frame=frame[["trade_date", "symbol", "close"]],
        feature_cols=("f1", "f2"),
        target_col="target",
        train_target_col="target",
    )


def test_research_dataset_fetch_infer_hides_targets() -> None:
    dataset = _dataset()

    infer = dataset.fetch_infer("test")
    audit = dataset.fetch_infer("test", audit=True)

    assert "target" not in infer.columns
    assert "target" in audit.columns
    assert len(dataset.fetch_learn("train")) == 2
    assert dataset.summary()["learn_rows"] == 4


def test_research_model_facade_fit_predict_detail() -> None:
    dataset = _dataset()
    model = CSTreeModel.from_config(
        {"type": "ridge", "params": {"alpha": 0.1}},
        features=["f1", "f2"],
        target_col="target",
    )

    before = model.detail()
    assert before["fitted"] is False

    model.fit(dataset, "train")
    pred = model.predict(dataset, "test")
    detail = model.detail()

    assert {"signal_date", "symbol", "raw_pred", "model_version", "feature_set_id"}.issubset(
        pred.columns
    )
    assert detail["fitted"] is True
    assert detail["feature_importance_source"] in {"coef_abs", "feature_importances"}
