"""Matched chronological cross-sectional rank regression via the shared registry."""

import hashlib

import numpy as np
import pandas as pd

from .modeling import build_model, fit_model, resolve_model_spec


def fit_matched_rank_models(
    train, test, features, decision_date, configs, *, target_transform="rank"
):
    """Fit estimators on identical rows with training-only imputation.

    Callers supply already-selected training rows. This boundary rejects immature
    labels rather than silently dropping them. Missing features retain their rows;
    all-missing training columns use zero. Targets must be finite. Return scores
    are not calibrated expected returns or confidence estimates.
    """
    date = pd.Timestamp(decision_date)
    train = train.copy().sort_values(["formation_date", "symbol"])
    test = test.copy()
    formation = pd.to_datetime(train.formation_date)
    end = pd.to_datetime(train.label_end_date)
    if train.empty or not (formation.lt(date) & end.lt(date)).all():
        raise ValueError("training observations must be mature strictly before decision")
    if test.empty or not pd.to_datetime(test.formation_date).eq(date).all():
        raise ValueError("inference observations must match decision date")
    if train.duplicated(["formation_date", "symbol"]).any() or test.symbol.duplicated().any():
        raise ValueError("duplicate observation keys")
    if not features or len(features) != len(set(features)) or not configs:
        raise ValueError("unique features and model configurations required")
    target = pd.to_numeric(train.target, errors="coerce")
    if not np.isfinite(target).all():
        raise ValueError("training targets must be finite")
    if target_transform not in {"rank", "identity"}:
        raise ValueError("target_transform must be rank or identity")
    train["rank_target"] = (
        target.groupby(formation).rank(pct=True) if target_transform == "rank" else target
    )
    x = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x.median().fillna(0.0)
    train[features] = x.fillna(medians)
    test[features] = (
        test[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
    )
    predictions = test[["formation_date", "symbol"]].copy()
    model_records = {}
    for name, config in configs.items():
        model_type, params = resolve_model_spec(config)
        if model_type not in {
            "ridge",
            "ridge_scaled",
            "random_forest_regressor",
            "xgb_regressor",
            "xgb_ranker",
        }:
            raise ValueError("matched comparison requires a supported estimator")
        if model_type == "xgb_ranker" and (
            target_transform != "rank"
            or params.get("objective", "rank:pairwise") != "rank:pairwise"
        ):
            raise ValueError(
                "pairwise comparator requires rank targets and rank:pairwise objective"
            )
        model = build_model(model_type, params)
        fit_model(
            model,
            model_type,
            train,
            features=features,
            target_col="rank_target",
            date_col="formation_date",
        )
        predictions[name] = model.predict(test[features])
        if not np.isfinite(predictions[name]).all():
            raise ValueError("nonfinite model predictions")
        model_records[name] = {"type": model_type, "params": params}
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(
            train[["formation_date", "symbol", "label_end_date", "rank_target", *features]],
            index=False,
        )
        .to_numpy()
        .tobytes()
    ).hexdigest()
    return predictions, {
        "training_rows": len(train),
        "training_sha256": digest,
        "max_label_end": str(end.max()),
        "decision_date": str(date),
        "features": list(features),
        "imputation_medians": medians.tolist(),
        "rank_labels": train.rank_target.tolist(),
        "models": model_records,
        "target_semantics": "within-formation percentile rank; average ties"
        if target_transform == "rank"
        else "identity numeric target",
    }
