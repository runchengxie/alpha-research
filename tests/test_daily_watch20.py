from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cstree.alpha.daily_watch20 as daily_watch20
from cstree.alpha.daily_watch20 import (
    DailyWatch20Config,
    DailyWatch20Ranker,
    build_forward_rank_label,
)
from cstree.alpha.signal_artifact import CANONICAL_SIGNAL_COLUMNS


def _panel(n_dates: int = 9) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=n_dates)
    rows = []
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


def _tiny_config(**overrides: object) -> DailyWatch20Config:
    values = {
        "features": ("f1", "f2"),
        "train_window_dates": None,
        "model_params": {
            "n_estimators": 3,
            "max_depth": 1,
            "learning_rate": 0.1,
            "objective": "rank:pairwise",
            "n_jobs": 1,
            "random_state": 7,
        },
    }
    values.update(overrides)
    return DailyWatch20Config(**values)


def test_build_forward_rank_label_uses_five_observed_days_and_date_rank() -> None:
    panel = _panel()

    labeled = build_forward_rank_label(panel)

    first_date = panel["trade_date"].min()
    first = labeled[labeled["trade_date"].eq(first_date)].set_index("symbol")
    assert first.loc["A", "forward_return_5d"] == pytest.approx(0.5)
    assert first.loc["B", "forward_return_5d"] == pytest.approx(-0.25)
    assert first.loc["A", "forward_rank_5d"] == 1.0
    assert first.loc["B", "forward_rank_5d"] == 0.5
    assert labeled.groupby("symbol", sort=False)["forward_rank_5d"].tail(5).isna().all()


def test_fit_uses_rolling_queries_date_equal_weights_and_as_of_purge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel(10)
    captured: dict[str, object] = {}

    class _Model:
        pass

    monkeypatch.setattr(daily_watch20, "build_model", lambda *_args: _Model())

    def _capture_fit(
        model: object,
        model_type: str,
        train_data: pd.DataFrame,
        **kwargs: object,
    ) -> object:
        captured["model_type"] = model_type
        captured["data"] = train_data.copy()
        captured["weight"] = np.asarray(kwargs["sample_weight"])
        return model

    monkeypatch.setattr(daily_watch20, "fit_model", _capture_fit)
    ranker = DailyWatch20Ranker(_tiny_config(train_window_dates=3))

    ranker.fit(panel, as_of_date=panel["trade_date"].max())

    train_data = captured["data"]
    assert isinstance(train_data, pd.DataFrame)
    assert captured["model_type"] == "xgb_ranker"
    assert train_data["trade_date"].nunique() == 3
    assert train_data["trade_date"].max() == panel["trade_date"].sort_values().unique()[-6]
    weights = pd.Series(captured["weight"], index=train_data.index)
    group_weights = weights.groupby(train_data["trade_date"]).sum()
    assert np.allclose(group_weights.to_numpy(), 1.0)
    assert ranker.training_summary is not None
    assert ranker.training_summary.query_groups == 3


def test_time_decay_increases_recent_query_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(daily_watch20, "build_model", lambda *_args: object())

    def _capture_fit(model: object, *_args: object, **kwargs: object) -> object:
        captured["data"] = _args[1].copy()
        captured["weight"] = np.asarray(kwargs["sample_weight"])
        return model

    monkeypatch.setattr(daily_watch20, "fit_model", _capture_fit)
    ranker = DailyWatch20Ranker(
        _tiny_config(sample_weight_mode="exp_decay", sample_weight_params={"halflife": 1})
    )

    ranker.fit(_panel(9))

    train_data = captured["data"]
    assert isinstance(train_data, pd.DataFrame)
    weights = pd.Series(captured["weight"], index=train_data.index)
    group_weights = weights.groupby(train_data["trade_date"]).sum().sort_index()
    assert group_weights.iloc[-1] > group_weights.iloc[0]


def test_actual_ranker_outputs_only_relative_scores_and_live_defaults_false() -> None:
    panel = _panel(9)
    ranker = DailyWatch20Ranker(_tiny_config()).fit(panel)
    prediction = panel[panel["trade_date"].eq(panel["trade_date"].max())]

    relative = ranker.predict_relative(prediction)
    signals = ranker.predict_signals(prediction)

    assert list(relative.columns) == ["trade_date", "symbol", "relative_percentile", "rank"]
    assert relative["relative_percentile"].between(0.0, 1.0).all()
    assert list(signals.columns) == list(CANONICAL_SIGNAL_COLUMNS)
    assert signals["raw_pred"].equals(signals["signal_eval"])
    assert signals["raw_pred"].equals(signals["signal_backtest"])
    assert set(signals["raw_pred"]) == set(relative["relative_percentile"])
    assert not signals["eligible_for_live"].any()


def test_explain_uses_native_xgboost_contributions() -> None:
    panel = _panel(9)
    ranker = DailyWatch20Ranker(_tiny_config()).fit(panel)
    prediction = panel[panel["trade_date"].eq(panel["trade_date"].max())]

    explanation = ranker.explain(prediction)

    assert explanation.source == "xgb_pred_contribs"
    assert set(explanation.local_contributions["feature"]) == {"f1", "f2", "__bias__"}
    assert len(explanation.local_contributions) == len(prediction) * 3
    assert set(explanation.feature_importance["feature"]) == {"f1", "f2"}


def test_explain_falls_back_to_global_feature_importance() -> None:
    class _FallbackModel:
        feature_importances_ = np.array([0.75, 0.25])

    panel = _panel(1)
    ranker = DailyWatch20Ranker(_tiny_config())
    ranker.model = _FallbackModel()

    explanation = ranker.explain(panel)

    assert explanation.source == "feature_importances"
    assert explanation.local_contributions.empty
    assert explanation.feature_importance.set_index("feature").loc["f1", "importance"] == 0.75


def test_duplicate_stock_date_rows_are_rejected() -> None:
    panel = _panel()
    duplicate = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="requires unique"):
        build_forward_rank_label(duplicate)
