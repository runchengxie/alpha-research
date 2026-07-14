from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from xgboost import XGBRanker, XGBRegressor

FitFn = Callable[
    [Any, pd.DataFrame],
    Any,
]
ImportanceFn = Callable[[Any], tuple[np.ndarray, str]]
FactoryFn = Callable[[Mapping[str, Any]], Any]


class FixedScoreArtifactModel:
    def __init__(self, score_col: str = "pred") -> None:
        self.score_col = str(score_col)
        self.is_fitted_ = False

    def fit(self, frame: pd.DataFrame) -> FixedScoreArtifactModel:
        if self.score_col not in frame.columns:
            raise ValueError(f"fixed_score_artifact score column not found: {self.score_col}")
        self.is_fitted_ = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.score_col not in frame.columns:
            raise ValueError(f"fixed_score_artifact score column not found: {self.score_col}")
        return pd.to_numeric(frame[self.score_col], errors="coerce").to_numpy(dtype=float)


def _fit_fixed_score_artifact_model(
    model: FixedScoreArtifactModel,
    train_data: pd.DataFrame,
    *,
    features: Sequence[str],
    target_col: str,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    date_col: str = "trade_date",
) -> FixedScoreArtifactModel:
    del features, target_col, sample_weight, date_col
    return model.fit(train_data)


def _fixed_score_feature_importance(model: Any) -> tuple[np.ndarray, str]:
    del model
    return np.array([], dtype=float), "fixed_score_artifact"


def _ranker_group_weights(
    train_data: pd.DataFrame,
    train_sorted: pd.DataFrame,
    sample_weight: Sequence[float] | np.ndarray,
    *,
    date_col: str,
    n_groups: int,
) -> np.ndarray:
    weight_values = np.asarray(sample_weight, dtype=float).reshape(-1)
    if weight_values.size == len(train_data):
        weight_series = pd.Series(weight_values, index=train_data.index, dtype=float)
        # XGBoost rankers expect one weight per query group, not one per row.
        weight_frame = pd.DataFrame(
            {
                date_col: train_sorted[date_col].to_numpy(),
                "_weight": weight_series.loc[train_sorted.index].to_numpy(),
            }
        )
        return weight_frame.groupby(date_col, sort=False)["_weight"].sum().to_numpy()
    if weight_values.size == n_groups:
        return weight_values
    raise ValueError(
        "xgb_ranker sample_weight must match either the number of training rows "
        f"({len(train_data)}) or the number of date groups ({n_groups})."
    )


def _fit_regressor_model(
    model: Any,
    train_data: pd.DataFrame,
    *,
    features: Sequence[str],
    target_col: str,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    date_col: str = "trade_date",
) -> Any:
    del date_col
    x_train = train_data[list(features)]
    y_train = train_data[target_col]
    if sample_weight is not None:
        model.fit(x_train, y_train, sample_weight=np.asarray(sample_weight, dtype=float))
    else:
        model.fit(x_train, y_train)
    return model


def _fit_ranker_model(
    model: Any,
    train_data: pd.DataFrame,
    *,
    features: Sequence[str],
    target_col: str,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    date_col: str = "trade_date",
) -> Any:
    train_sorted = train_data.sort_values(date_col, kind="mergesort")
    groups = train_sorted.groupby(date_col, sort=False)[date_col].size().tolist()
    if not groups:
        raise ValueError("xgb_ranker requires non-empty grouped training data.")
    x_train = train_sorted[list(features)]
    y_train = train_sorted[target_col]
    if sample_weight is not None:
        group_weight = _ranker_group_weights(
            train_data,
            train_sorted,
            sample_weight,
            date_col=date_col,
            n_groups=len(groups),
        )
        model.fit(x_train, y_train, group=groups, sample_weight=group_weight)
    else:
        model.fit(x_train, y_train, group=groups)
    return model


def _generic_feature_importance(model: Any) -> tuple[np.ndarray, str]:
    if hasattr(model, "feature_importances_"):
        return (
            np.asarray(model.feature_importances_, dtype=float).reshape(-1),
            "feature_importances",
        )
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        coef = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        return np.asarray(coef, dtype=float).reshape(-1), "coef_abs"
    return np.array([], dtype=float), "none"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    aliases: tuple[str, ...]
    default_params: Mapping[str, Any]
    factory: FactoryFn
    fit: Callable[..., Any]
    importance: ImportanceFn = _generic_feature_importance

    def build(self, params: Mapping[str, Any]) -> Any:
        return self.factory(dict(params))

    def defaults(self) -> dict[str, Any]:
        return dict(self.default_params)


_MODEL_REGISTRY: dict[str, ModelSpec] = {
    "xgb_regressor": ModelSpec(
        name="xgb_regressor",
        aliases=("xgb", "xgboost", "xgb_regressor", "xgbregressor"),
        default_params={
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
        },
        factory=lambda params: XGBRegressor(**dict(params)),
        fit=_fit_regressor_model,
    ),
    "xgb_ranker": ModelSpec(
        name="xgb_ranker",
        aliases=("xgb_ranker", "xgb_rank", "xgboost_ranker", "ranker"),
        default_params={
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "rank:pairwise",
            "random_state": 42,
        },
        factory=lambda params: XGBRanker(**dict(params)),
        fit=_fit_ranker_model,
    ),
    "ridge": ModelSpec(
        name="ridge",
        aliases=("ridge", "ridge_regressor", "ridge_regression"),
        default_params={
            "alpha": 1.0,
            "fit_intercept": True,
            "random_state": 42,
        },
        factory=lambda params: Ridge(**dict(params)),
        fit=_fit_regressor_model,
    ),
    "elasticnet": ModelSpec(
        name="elasticnet",
        aliases=("elasticnet", "elastic_net", "elasticnet_regressor", "elasticnet_regression"),
        default_params={
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "fit_intercept": True,
            "max_iter": 5000,
            "random_state": 42,
        },
        factory=lambda params: ElasticNet(**dict(params)),
        fit=_fit_regressor_model,
    ),
    "fixed_score_artifact": ModelSpec(
        name="fixed_score_artifact",
        aliases=("fixed_score_artifact", "fixed_scored_artifact", "fixed_score", "artifact_score"),
        default_params={"score_col": "pred"},
        factory=lambda params: FixedScoreArtifactModel(
            score_col=str(dict(params).get("score_col", "pred"))
        ),
        fit=_fit_fixed_score_artifact_model,
        importance=_fixed_score_feature_importance,
    ),
}

SUPPORTED_MODEL_TYPES = tuple(_MODEL_REGISTRY)
_MODEL_ALIASES: dict[str, str] = {
    alias: name for name, spec in _MODEL_REGISTRY.items() for alias in spec.aliases
}
_DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    name: spec.defaults() for name, spec in _MODEL_REGISTRY.items()
}


def _get_model_spec(model_type: object | None) -> ModelSpec:
    return _MODEL_REGISTRY[normalize_model_type(model_type)]


def normalize_model_type(value: object | None) -> str:
    text = str(value or "xgb_regressor").strip().lower().replace("-", "_")
    model_type = _MODEL_ALIASES.get(text)
    if model_type:
        return model_type
    supported = ", ".join(SUPPORTED_MODEL_TYPES)
    raise ValueError(f"Unsupported model.type: {value}. Supported values: {supported}.")


def resolve_model_spec(model_cfg: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if model_cfg is None:
        model_cfg = {}
    if not isinstance(model_cfg, Mapping):
        raise ValueError("model must be a mapping with keys: type, params.")
    spec = _get_model_spec(model_cfg.get("type"))
    params_raw = model_cfg.get("params") or {}
    if not isinstance(params_raw, Mapping):
        raise ValueError("model.params must be a mapping.")
    model_params = dict(params_raw)
    if not model_params:
        model_params = spec.defaults()
    return spec.name, model_params


def build_model(model_type: str, model_params: Mapping[str, Any]) -> Any:
    return _get_model_spec(model_type).build(model_params)


def build_model_from_config(model_cfg: Mapping[str, Any] | None) -> tuple[Any, str, dict[str, Any]]:
    model_type, model_params = resolve_model_spec(model_cfg)
    return build_model(model_type, model_params), model_type, model_params


def fit_model(
    model: Any,
    model_type: str,
    train_data: pd.DataFrame,
    *,
    features: Sequence[str],
    target_col: str,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    date_col: str = "trade_date",
) -> Any:
    spec = _get_model_spec(model_type)
    return spec.fit(
        model,
        train_data,
        features=features,
        target_col=target_col,
        sample_weight=sample_weight,
        date_col=date_col,
    )


def feature_importance_frame(
    model: Any,
    features: Sequence[str],
    *,
    model_type: str | None = None,
) -> tuple[pd.DataFrame, str]:
    n_features = len(features)
    if model_type is None:
        raw, source = _generic_feature_importance(model)
    else:
        raw, source = _get_model_spec(model_type).importance(model)
    if raw.size == 0:
        raw = np.zeros(n_features, dtype=float)

    if raw.size < n_features:
        raw = np.pad(raw, (0, n_features - raw.size), constant_values=np.nan)
    elif raw.size > n_features:
        raw = raw[:n_features]

    frame = pd.DataFrame(
        {
            "feature": list(features),
            "importance": raw,
        }
    ).sort_values("importance", ascending=False)
    return frame, source
