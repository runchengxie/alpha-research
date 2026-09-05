import numpy as np
import pandas as pd
import pytest


def frames():
    train = pd.DataFrame(
        {
            "formation_date": pd.to_datetime(["2020-01-01"] * 4),
            "label_end_date": pd.to_datetime(["2020-02-01"] * 4),
            "symbol": ["A", "B", "C", "D"],
            "target": [4.0, 3.0, 2.0, 1.0],
            "f": [1.0, 2.0, np.nan, 4.0],
        }
    )
    test = pd.DataFrame(
        {
            "formation_date": pd.to_datetime(["2020-03-01"] * 2),
            "symbol": ["X", "Y"],
            "f": [np.nan, 10000.0],
        }
    )
    return train, test


def test_matched_models_use_train_only_imputation_and_identical_labels():
    from alpha_research.matched_rank_models import fit_matched_rank_models

    train, test = frames()
    configs = {
        "linear": {"type": "ridge_scaled", "params": {"alpha": 10.0}},
        "forest": {
            "type": "random_forest",
            "params": {"n_estimators": 5, "max_depth": 2, "random_state": 7},
        },
    }
    predictions, receipt = fit_matched_rank_models(train, test, ["f"], "2020-03-01", configs)
    assert predictions.symbol.tolist() == ["X", "Y"]
    assert np.isfinite(predictions[["linear", "forest"]]).all().all()
    assert receipt["imputation_medians"] == [2.0]
    assert receipt["training_rows"] == 4
    assert receipt["rank_labels"] == [1.0, 0.75, 0.5, 0.25]
    changed = test.copy()
    changed.loc[1, "f"] = -100000.0
    pred2, receipt2 = fit_matched_rank_models(train, changed, ["f"], "2020-03-01", configs)
    assert receipt2 == receipt
    assert pred2.loc[0, "linear"] == pytest.approx(predictions.loc[0, "linear"])
    assert train.f.isna().sum() == 1


def test_fit_rejects_label_maturing_at_or_after_decision():
    from alpha_research.matched_rank_models import fit_matched_rank_models

    train, test = frames()
    train.loc[0, "label_end_date"] = pd.Timestamp("2020-03-01")
    with pytest.raises(ValueError, match="mature"):
        fit_matched_rank_models(train, test, ["f"], "2020-03-01", {"ridge": {"type": "ridge"}})


def test_numeric_risk_target_is_not_converted_to_cross_sectional_rank():
    from alpha_research.matched_rank_models import fit_matched_rank_models

    train, test = frames()
    train["target"] = [0.01, 0.02, 0.03, 0.04]
    _, receipt = fit_matched_rank_models(
        train,
        test,
        ["f"],
        "2020-03-01",
        {"ridge": {"type": "ridge"}},
        target_transform="identity",
    )
    assert receipt["rank_labels"] == pytest.approx([0.01, 0.02, 0.03, 0.04])
    assert receipt["target_semantics"] == "identity numeric target"


def test_pairwise_uses_same_imputed_rows_and_rank_labels_as_pointwise():
    from alpha_research.matched_rank_models import fit_matched_rank_models

    train, test = frames()
    config = {
        "pair": {
            "type": "xgb_ranker",
            "params": {
                "objective": "rank:pairwise",
                "n_estimators": 5,
                "max_depth": 2,
                "random_state": 7,
                "n_jobs": 1,
            },
        }
    }
    pred, receipt = fit_matched_rank_models(train, test, ["f"], "2020-03-01", config)
    _, point = fit_matched_rank_models(
        train, test, ["f"], "2020-03-01", {"ridge": {"type": "ridge"}}
    )
    assert receipt["training_sha256"] == point["training_sha256"]
    assert pred.symbol.tolist() == ["X", "Y"]
    assert np.isfinite(pred.pair).all()
    with pytest.raises(ValueError, match="pairwise"):
        fit_matched_rank_models(
            train, test, ["f"], "2020-03-01", config, target_transform="identity"
        )
