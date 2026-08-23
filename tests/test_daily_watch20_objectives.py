from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import alpha_research.daily_watch20 as daily_watch20
from alpha_research.daily_watch20 import DailyWatch20Config, DailyWatch20Ranker


def _panel(n_dates: int = 9) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=n_dates)
    rows: list[dict[str, object]] = []
    for date_number, date in enumerate(dates):
        rows.extend(
            [
                {
                    "trade_date": date,
                    "symbol": "A",
                    "adj_close": 10.0 + date_number,
                    "f1": float(date_number),
                    "f2": 0.0,
                },
                {
                    "trade_date": date,
                    "symbol": "B",
                    "adj_close": 20.0 - date_number,
                    "f1": -float(date_number),
                    "f2": 1.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def _config(objective: str) -> DailyWatch20Config:
    return DailyWatch20Config(
        features=("f1", "f2"),
        train_window_dates=None,
        model_params={
            "n_estimators": 3,
            "max_depth": 1,
            "learning_rate": 0.1,
            "objective": objective,
            "n_jobs": 1,
            "random_state": 7,
        },
    )


@pytest.mark.parametrize(
    ("objective", "expected_model_type"),
    [
        ("reg:squarederror", "xgb_regressor"),
        ("rank:pairwise", "xgb_ranker"),
        ("rank:ndcg", "xgb_ranker"),
    ],
)
def test_objective_selects_pointwise_pairwise_or_listwise_training(
    objective: str,
    expected_model_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(daily_watch20, "build_model", lambda *_args: object())

    def _capture_fit(
        model: object,
        model_type: str,
        train_data: pd.DataFrame,
        **kwargs: object,
    ) -> object:
        captured["model_type"] = model_type
        captured["data"] = train_data.copy()
        captured["target_col"] = kwargs["target_col"]
        return model

    monkeypatch.setattr(daily_watch20, "fit_model", _capture_fit)
    ranker = DailyWatch20Ranker(_config(objective)).fit(_panel())

    assert ranker.model_type == expected_model_type
    assert captured["model_type"] == expected_model_type
    assert ranker.model_params["objective"] == objective

    fit_data = cast(pd.DataFrame, captured["data"])
    target_col = cast(str, captured["target_col"])
    if objective == "rank:ndcg":
        assert pd.api.types.is_integer_dtype(fit_data[target_col])
        assert fit_data[target_col].between(0, 31).all()
    else:
        assert not pd.api.types.is_integer_dtype(fit_data[target_col])


def test_listwise_relevance_grades_preserve_label_order(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(daily_watch20, "build_model", lambda *_args: object())

    def _capture_fit(model: object, *_args: object, **kwargs: object) -> object:
        captured["data"] = cast(pd.DataFrame, _args[1]).copy()
        captured["target_col"] = kwargs["target_col"]
        return model

    monkeypatch.setattr(daily_watch20, "fit_model", _capture_fit)
    DailyWatch20Ranker(_config("rank:ndcg")).fit(_panel())

    fit_data = cast(pd.DataFrame, captured["data"])
    target_col = cast(str, captured["target_col"])
    by_date = fit_data.groupby("trade_date", sort=False)[target_col]
    assert all(group.max() > group.min() for _, group in by_date)


def test_model_identity_separates_training_objectives() -> None:
    pointwise = DailyWatch20Ranker(_config("reg:squarederror"))
    pairwise = DailyWatch20Ranker(_config("rank:pairwise"))
    listwise = DailyWatch20Ranker(_config("rank:ndcg"))

    assert len({pointwise.model_version, pairwise.model_version, listwise.model_version}) == 3
    assert pointwise.feature_set_id == pairwise.feature_set_id == listwise.feature_set_id


def test_unsupported_objective_is_rejected() -> None:
    with pytest.raises(ValueError, match="supports pointwise reg:\\* objectives"):
        DailyWatch20Ranker(_config("binary:logistic"))


def test_actual_pointwise_model_keeps_relative_output_contract() -> None:
    panel = _panel()
    ranker = DailyWatch20Ranker(_config("reg:squarederror")).fit(panel)
    prediction = panel.loc[panel["trade_date"].eq(panel["trade_date"].max())]

    relative = ranker.predict_relative(prediction)

    assert list(relative.columns) == ["trade_date", "symbol", "relative_percentile", "rank"]
    assert np.isfinite(relative["relative_percentile"]).all()
    assert relative["relative_percentile"].between(0.0, 1.0).all()
