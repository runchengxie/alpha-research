import numpy as np
import pandas as pd
import pytest

from alpha_research.modeling import (
    SUPPORTED_MODEL_TYPES,
    build_model,
    feature_importance_frame,
    fit_model,
    normalize_model_type,
    resolve_model_spec,
)


def test_resolve_model_spec_defaults_to_xgb():
    model_type, params = resolve_model_spec(None)
    assert model_type == "xgb_regressor"
    assert params["objective"] == "reg:squarederror"


def test_resolve_model_spec_accepts_alias():
    model_type, params = resolve_model_spec({"type": "ridge_regressor", "params": {"alpha": 2.0}})
    assert model_type == "ridge"
    assert params == {"alpha": 2.0}


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("xgb", "xgb_regressor"),
        ("xgboost-ranker", "xgb_ranker"),
        ("ridge_regression", "ridge"),
        ("elastic_net", "elasticnet"),
        ("fixed_scored_artifact", "fixed_score_artifact"),
    ],
)
def test_normalize_model_type_uses_registry_aliases(alias, expected):
    assert normalize_model_type(alias) == expected


def test_supported_model_types_come_from_registry():
    assert SUPPORTED_MODEL_TYPES == (
        "xgb_regressor",
        "xgb_ranker",
        "ridge",
        "ridge_scaled",
        "random_forest_regressor",
        "elasticnet",
        "fixed_score_artifact",
    )


@pytest.mark.parametrize(
    ("model_type", "params", "expected_class"),
    [
        (
            "xgb_regressor",
            {"n_estimators": 1, "max_depth": 1, "learning_rate": 0.1},
            "XGBRegressor",
        ),
        (
            "xgb_ranker",
            {"n_estimators": 1, "max_depth": 1, "learning_rate": 0.1, "objective": "rank:pairwise"},
            "XGBRanker",
        ),
        ("ridge", {"alpha": 1.0}, "Ridge"),
        ("elasticnet", {"alpha": 0.1, "l1_ratio": 0.5}, "ElasticNet"),
        ("fixed_score_artifact", {"score_col": "score"}, "FixedScoreArtifactModel"),
    ],
)
def test_build_model_supported_types(model_type, params, expected_class):
    model = build_model(model_type, params)
    assert model.__class__.__name__ == expected_class


def test_feature_importance_frame_for_linear_model_uses_abs_coef():
    X = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ]
    )
    y = np.array([1.0, 2.0, -1.0, -2.0])
    model = build_model("ridge", {"alpha": 0.1})
    model.fit(X, y)

    importance_df, source = feature_importance_frame(model, ["f1", "f2"])
    assert source == "coef_abs"
    assert set(importance_df["feature"]) == {"f1", "f2"}
    assert importance_df["importance"].ge(0).all()


def test_feature_importance_frame_accepts_model_type_hint():
    model = build_model("ridge", {"alpha": 0.1})
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))

    importance_df, source = feature_importance_frame(model, ["f1"], model_type="ridge")

    assert source == "coef_abs"
    assert importance_df["feature"].tolist() == ["f1"]
    assert float(importance_df["importance"].iloc[0]) > 0


def test_scaled_ridge_model_scales_features_before_regression():
    model = build_model("ridge_scaled", {"alpha": 1.0})
    model.fit(
        np.array([[1.0e-3, 1.0e3], [2.0e-3, 2.0e3], [3.0e-3, 3.0e3]]),
        np.array([1.0, 2.0, 3.0]),
    )

    assert model.named_steps["scaler"].mean_.tolist() == pytest.approx([0.002, 2000.0])
    assert model.predict(np.array([[4.0e-3, 4.0e3]])).shape == (1,)


def test_resolve_model_spec_rejects_unknown_type():
    with pytest.raises(ValueError, match=r"Unsupported model\.type"):
        resolve_model_spec({"type": "unknown_estimator", "params": {}})


def test_random_forest_uses_shared_fit_prediction_and_importance_interfaces():
    model_type, params = resolve_model_spec(
        {
            "type": "random_forest",
            "params": {
                "n_estimators": 20,
                "max_depth": 2,
                "random_state": 7,
            },
        }
    )
    frame = pd.DataFrame({"feature": [0.0] * 20 + [1.0] * 20, "target": [0.0] * 20 + [1.0] * 20})
    model = build_model(model_type, params)
    fit_model(model, model_type, frame, features=["feature"], target_col="target")
    predictions = model.predict(pd.DataFrame({"feature": [0.0, 1.0]}))
    assert predictions.tolist() == pytest.approx([0.0, 1.0])
    importance, source = feature_importance_frame(model, ["feature"], model_type=model_type)
    assert importance.importance.tolist() == pytest.approx([1.0])
    assert source == "feature_importances"


def test_fixed_score_artifact_model_replays_configured_score_column():
    frame = pd.DataFrame({"score": [0.2, -0.1], "target": [1.0, 0.0]})
    model = build_model("fixed_score_artifact", {"score_col": "score"})

    fit_model(model, "fixed_score_artifact", frame, features=["score"], target_col="target")
    preds = model.predict(frame[["score"]])
    importance, source = feature_importance_frame(
        model,
        ["score"],
        model_type="fixed_score_artifact",
    )

    assert preds.tolist() == [0.2, -0.1]
    assert source == "fixed_score_artifact"
    assert importance["importance"].tolist() == [0.0]


def test_fit_model_supports_ranker_groups():
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "f1": [0.1, 0.9, 0.2, 0.8],
            "target": [0.0, 1.0, 0.0, 1.0],
        }
    )

    model = build_model(
        "xgb_ranker",
        {"n_estimators": 2, "max_depth": 1, "learning_rate": 0.1, "objective": "rank:pairwise"},
    )
    fit_model(model, "xgb_ranker", frame, features=["f1"], target_col="target")
    preds = model.predict(frame[["f1"]])
    assert preds.shape[0] == len(frame)


def test_fit_model_supports_listwise_ranker_with_relevance_labels():
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3),
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "f1": [0.1, 0.9, 0.4, 0.2, 0.8, 0.5],
            "target": [0.0, 2.0, 1.0, 0.0, 2.0, 1.0],
        }
    )

    model = build_model(
        "xgb_ranker",
        {"n_estimators": 1, "max_depth": 1, "learning_rate": 0.1, "objective": "rank:ndcg"},
    )
    fit_model(model, "xgb_ranker", frame, features=["f1"], target_col="target")
    preds = model.predict(frame[["f1"]])
    assert preds.shape[0] == len(frame)


def test_fit_model_supports_ranker_row_weights():
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                ]
            ),
            "symbol": ["A", "B", "A", "B", "A", "B"],
            "f1": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            "target": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )

    model = build_model(
        "xgb_ranker",
        {"n_estimators": 3, "max_depth": 1, "learning_rate": 0.1, "objective": "rank:pairwise"},
    )
    fit_model(
        model,
        "xgb_ranker",
        frame,
        features=["f1"],
        target_col="target",
        sample_weight=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    )
    preds = model.predict(frame[["f1"]])
    assert preds.shape[0] == len(frame)
