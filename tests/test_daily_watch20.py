from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cstree.alpha.daily_watch20 as daily_watch20
from cstree.alpha.daily_watch20 import (
    DailyWatch20Config,
    DailyWatch20Ranker,
    DailyWatch20TrainingSummary,
    build_forward_rank_label,
    build_multi_horizon_forward_rank_label,
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


def test_multi_horizon_label_requires_all_components_and_uses_weighted_ranks() -> None:
    panel = _panel()

    labeled = build_multi_horizon_forward_rank_label(panel)

    first = labeled.loc[labeled["trade_date"].eq(panel["trade_date"].min())].set_index("symbol")
    expected_a = 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0
    expected_b = 0.5 * 0.5 + 0.3 * 0.5 + 0.2 * 0.5
    assert first.loc["A", "forward_rank_blended"] == pytest.approx(expected_a)
    assert first.loc["B", "forward_rank_blended"] == pytest.approx(expected_b)
    assert labeled.groupby("symbol", sort=False)["forward_rank_1d"].tail(1).isna().all()
    assert labeled.groupby("symbol", sort=False)["forward_rank_3d"].tail(3).isna().all()
    assert labeled.groupby("symbol", sort=False)["forward_rank_blended"].tail(5).isna().all()


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


def test_fit_purges_blended_next_open_label_until_longest_horizon_is_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel(12)
    dates = pd.Index(panel["trade_date"].unique()).sort_values()
    date_to_number = {date: number for number, date in enumerate(dates)}
    panel["forward_rank_blended"] = 0.75
    panel["forward_label_end_date"] = panel["trade_date"].map(
        lambda date: (
            dates[date_to_number[date] + 6] if date_to_number[date] + 6 < len(dates) else pd.NaT
        )
    )
    captured: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr(daily_watch20, "build_model", lambda *_args: object())

    def _capture_fit(model: object, *_args: object, **_kwargs: object) -> object:
        captured["data"] = _args[1].copy()
        return model

    monkeypatch.setattr(daily_watch20, "fit_model", _capture_fit)

    DailyWatch20Ranker(_tiny_config()).fit(panel, as_of_date=dates[-1])

    assert captured["data"]["trade_date"].max() == dates[-7]


def test_ranker_config_can_select_legacy_single_five_day_target() -> None:
    config = _tiny_config(label_horizon_weights=((5, 1.0),))

    assert config.label_col == "forward_rank_5d"
    assert config.forward_return_col == "forward_return_5d"


def test_label_policy_changes_feature_set_identity() -> None:
    price_only = DailyWatch20Ranker(_tiny_config(label_policy_id="price_only.v1"))
    limit_aware = DailyWatch20Ranker(_tiny_config(label_policy_id="limit_aware.v2"))

    assert price_only.feature_set_id != limit_aware.feature_set_id


def test_feature_timing_policy_changes_feature_set_identity() -> None:
    lag0 = DailyWatch20Ranker(_tiny_config(feature_policy_id="minute.close.v2:lag=0"))
    lag1 = DailyWatch20Ranker(_tiny_config(feature_policy_id="minute.close.v2:lag=1"))

    assert lag0.config.label_policy_id == lag1.config.label_policy_id
    assert lag0.feature_set_id != lag1.feature_set_id
    assert lag0.model_version == lag1.model_version
    assert lag0.training_policy_id == lag1.training_policy_id


@pytest.mark.parametrize("field", ["feature_policy_id", "label_policy_id"])
def test_ranker_config_rejects_blank_policy_identity(field: str) -> None:
    with pytest.raises(
        ValueError,
        match=rf"DailyWatch20Config\.{field} must be non-empty\.",
    ):
        _tiny_config(**{field: "  "})


def test_ranker_config_normalizes_policy_identity_whitespace() -> None:
    config = _tiny_config(feature_policy_id=" feature.v2 ", label_policy_id=" label.v2 ")

    assert config.feature_policy_id == "feature.v2"
    assert config.label_policy_id == "label.v2"


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


def test_restore_requires_matching_model_and_feature_metadata() -> None:
    class _PersistedModel:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(frame))

    ranker = DailyWatch20Ranker(_tiny_config())
    dates = pd.bdate_range("2026-01-05", periods=3)
    summary = DailyWatch20TrainingSummary(
        as_of_date=dates[2],
        train_start_date=dates[0],
        train_end_date=dates[1],
        rows=4,
        query_groups=2,
        sample_weight_mode="date_equal",
    )

    restored = ranker.restore(
        _PersistedModel(),
        summary,
        metadata=ranker.persistence_metadata,
    )

    assert restored is ranker
    assert ranker.training_summary == summary
    with pytest.raises(ValueError, match="model_version"):
        DailyWatch20Ranker(_tiny_config()).restore(
            _PersistedModel(),
            summary,
            metadata={**ranker.persistence_metadata, "model_version": "other"},
        )
    changed_features = DailyWatch20Ranker(_tiny_config(features=("f1",)))
    with pytest.raises(ValueError, match="feature_set_id"):
        changed_features.restore(
            _PersistedModel(),
            summary,
            metadata=ranker.persistence_metadata,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_window_dates": 126},
        {"sample_weight_mode": "date_equal"},
        {
            "sample_weight_mode": "exp_decay",
            "sample_weight_params": {"halflife": 63.0, "min_weight": 0.05},
        },
        {
            "sample_weight_mode": "exp_decay",
            "sample_weight_params": {"halflife": 126.0, "min_weight": 0.10},
        },
    ],
)
def test_training_policy_identity_rejects_changed_fit_policy(
    overrides: dict[str, object],
) -> None:
    class _PersistedModel:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(frame))

    base = DailyWatch20Ranker(
        _tiny_config(
            train_window_dates=252,
            sample_weight_mode="exp_decay",
            sample_weight_params={"min_weight": 0.05, "halflife": 126.0},
        )
    )
    changed_policy: dict[str, object] = {
        "train_window_dates": 252,
        "sample_weight_mode": "exp_decay",
        "sample_weight_params": {"min_weight": 0.05, "halflife": 126.0},
    }
    changed_policy.update(overrides)
    changed = DailyWatch20Ranker(_tiny_config(**changed_policy))
    summary = DailyWatch20TrainingSummary(
        as_of_date=pd.Timestamp("2026-07-10"),
        train_start_date=pd.Timestamp("2025-01-02"),
        train_end_date=pd.Timestamp("2026-07-03"),
        rows=100,
        query_groups=50,
        sample_weight_mode="exp_decay",
    )

    assert changed.training_policy_id != base.training_policy_id
    with pytest.raises(ValueError, match="training_policy_id"):
        changed.restore(
            _PersistedModel(),
            summary,
            metadata=base.persistence_metadata,
        )


def test_training_policy_identity_is_stable_across_parameter_order() -> None:
    left = DailyWatch20Ranker(
        _tiny_config(
            sample_weight_mode="exp_decay",
            sample_weight_params={"halflife": 126.0, "min_weight": 0.05},
        )
    )
    right = DailyWatch20Ranker(
        _tiny_config(
            sample_weight_mode="exp_decay",
            sample_weight_params={"min_weight": 0.05, "halflife": 126.0},
        )
    )

    assert left.training_policy_id == right.training_policy_id


def test_restore_rejects_incomplete_metadata_and_invalid_training_summary() -> None:
    class _PersistedModel:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(frame))

    ranker = DailyWatch20Ranker(_tiny_config())
    with pytest.raises(ValueError, match="feature_set_id"):
        ranker.restore(_PersistedModel(), {}, metadata={"model_version": ranker.model_version})
    with pytest.raises(ValueError, match="training_policy_id"):
        ranker.restore(
            _PersistedModel(),
            {},
            metadata={
                "model_version": ranker.model_version,
                "feature_set_id": ranker.feature_set_id,
            },
        )
    with pytest.raises(ValueError, match="ends after"):
        ranker.restore(
            _PersistedModel(),
            {
                "as_of_date": "2026-01-05",
                "train_start_date": "2026-01-05",
                "train_end_date": "2026-01-06",
                "rows": 2,
                "query_groups": 1,
                "sample_weight_mode": None,
            },
            metadata=ranker.persistence_metadata,
        )


def test_restore_from_path_round_trips_native_model(tmp_path: Path) -> None:
    panel = _panel(9)
    trained = DailyWatch20Ranker(_tiny_config()).fit(panel)
    assert trained.model is not None
    assert trained.training_summary is not None
    model_path = tmp_path / "daily_watch20.ubj"
    trained.model.save_model(model_path)
    prediction = panel.loc[panel["trade_date"].eq(panel["trade_date"].max())]

    restored = DailyWatch20Ranker(_tiny_config()).restore_from_path(
        model_path,
        trained.training_summary,
        metadata=trained.persistence_metadata,
    )

    pd.testing.assert_frame_equal(
        restored.predict_relative(prediction),
        trained.predict_relative(prediction),
    )


def test_restore_from_path_rejects_incompatible_metadata(tmp_path: Path) -> None:
    trained = DailyWatch20Ranker(_tiny_config()).fit(_panel(9))
    assert trained.model is not None
    assert trained.training_summary is not None
    model_path = tmp_path / "daily_watch20.ubj"
    trained.model.save_model(model_path)

    with pytest.raises(ValueError, match="feature_set_id"):
        DailyWatch20Ranker(_tiny_config(features=("f1",))).restore_from_path(
            model_path,
            trained.training_summary,
            metadata=trained.persistence_metadata,
        )
