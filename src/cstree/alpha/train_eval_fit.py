from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import daily_ic_series, summarize_ic
from .modeling import build_model, fit_model
from .split import build_sample_weight
from .train_eval_contracts import (
    TrainEvalFeatureTarget,
    TrainEvalModelSettings,
    TrainEvalSignalSettings,
)
from .transform import apply_score_postprocess

logger = logging.getLogger("cstree")


@dataclass(frozen=True)
class _TrainFitResult:
    model: Any
    train_eval_df: pd.DataFrame
    updated_signal_direction: float
    train_signal_col: str
    train_ic_raw_stats: dict[str, Any]
    train_ic_series: pd.Series
    train_ic_stats: dict[str, Any]
    train_pearson_ic_series: pd.Series
    train_pearson_ic_stats: dict[str, Any]
    cv_scores_raw: list[float] | None = None
    cv_scores_adj: list[float] | None = None


def _fit_stage_model(
    train_df: pd.DataFrame,
    *,
    feature_target: TrainEvalFeatureTarget,
    model_settings: TrainEvalModelSettings,
) -> Any:
    logger.info("Fitting model (%s) ...", model_settings.model_type)
    model = build_model(model_settings.model_type, model_settings.model_params)
    train_weights = build_sample_weight(
        train_df,
        model_settings.sample_weight_mode,
        params=model_settings.sample_weight_params,
    )
    fit_model(
        model,
        model_settings.model_type,
        train_df,
        features=feature_target.features,
        target_col=feature_target.train_target,
        sample_weight=train_weights,
    )
    return model


def _score_train_frame(
    train_df: pd.DataFrame,
    model: Any,
    *,
    feature_target: TrainEvalFeatureTarget,
    signal_settings: TrainEvalSignalSettings,
) -> pd.DataFrame:
    train_eval_df = train_df.copy()
    train_eval_df["pred"] = model.predict(train_eval_df[feature_target.features])
    train_eval_df["pred"] = apply_score_postprocess(
        train_eval_df,
        "pred",
        method=signal_settings.score_postprocess_method,
        columns=signal_settings.score_postprocess_columns,
        strength=signal_settings.score_postprocess_strength,
        min_obs=signal_settings.score_postprocess_min_obs,
    )
    return train_eval_df


def _resolve_train_signal_direction(
    train_eval_df: pd.DataFrame,
    *,
    target: str,
    signal_settings: TrainEvalSignalSettings,
) -> tuple[float, dict[str, Any]]:
    train_ic_raw_stats = {}
    updated_signal_direction = signal_settings.signal_direction
    if signal_settings.signal_direction_mode != "train_ic":
        return updated_signal_direction, train_ic_raw_stats

    train_ic_raw_series = daily_ic_series(train_eval_df, target, "pred")
    train_ic_raw_stats = summarize_ic(train_ic_raw_series)
    raw_mean = train_ic_raw_stats.get("mean", np.nan)
    if np.isfinite(raw_mean) and raw_mean != 0:
        updated_signal_direction = float(np.sign(raw_mean))
    else:
        updated_signal_direction = 1.0
    logger.info("Signal direction set from Train IC: %s", updated_signal_direction)
    return updated_signal_direction, train_ic_raw_stats


def _adjust_cv_scores(
    cv_scores_raw: list[float],
    updated_signal_direction: float,
) -> list[float] | None:
    if not cv_scores_raw:
        return None
    cv_scores_adj = [float(score) * updated_signal_direction for score in cv_scores_raw]
    if updated_signal_direction != 1.0:
        logger.info(
            "CV IC (adj): mean=%.4f, std=%.4f",
            np.nanmean(cv_scores_adj),
            np.nanstd(cv_scores_adj),
        )
        logger.info("CV fold ICs (adj): %s", [f"{s:.4f}" for s in cv_scores_adj])
    return cv_scores_adj


def _train_signal_column(
    train_eval_df: pd.DataFrame,
    updated_signal_direction: float,
) -> str:
    if updated_signal_direction == 1.0:
        return "pred"
    train_eval_df["signal"] = train_eval_df["pred"] * updated_signal_direction
    return "signal"


def _empty_train_ic_report() -> tuple[pd.Series, dict[str, Any], pd.Series, dict[str, Any]]:
    return (
        pd.Series(dtype=float, name="ic"),
        {},
        pd.Series(dtype=float, name="ic_pearson"),
        {},
    )


def _train_ic_report(
    train_eval_df: pd.DataFrame,
    *,
    target: str,
    train_signal_col: str,
    signal_settings: TrainEvalSignalSettings,
) -> tuple[pd.Series, dict[str, Any], pd.Series, dict[str, Any]]:
    if not signal_settings.report_train_ic:
        return _empty_train_ic_report()

    train_ic_series = daily_ic_series(train_eval_df, target, train_signal_col)
    train_ic_stats = summarize_ic(train_ic_series)
    logger.info(
        "Train Daily IC: mean=%.4f, std=%.4f, IR=%.2f, t=%.2f, p=%.4f (n=%s)",
        train_ic_stats["mean"],
        train_ic_stats["std"],
        train_ic_stats["ir"],
        train_ic_stats["t_stat"],
        train_ic_stats["p_value"],
        train_ic_stats["n"],
    )
    train_pearson_ic_series = daily_ic_series(
        train_eval_df, target, train_signal_col, method="pearson"
    )
    train_pearson_ic_stats = summarize_ic(train_pearson_ic_series)
    logger.info(
        "Train Daily Pearson IC: mean=%.4f, std=%.4f, IR=%.2f, t=%.2f, p=%.4f (n=%s)",
        train_pearson_ic_stats["mean"],
        train_pearson_ic_stats["std"],
        train_pearson_ic_stats["ir"],
        train_pearson_ic_stats["t_stat"],
        train_pearson_ic_stats["p_value"],
        train_pearson_ic_stats["n"],
    )
    return (
        train_ic_series,
        train_ic_stats,
        train_pearson_ic_series,
        train_pearson_ic_stats,
    )


def fit_model_and_score_train(
    train_df: pd.DataFrame,
    *,
    feature_target: TrainEvalFeatureTarget,
    model_settings: TrainEvalModelSettings,
    signal_settings: TrainEvalSignalSettings,
    cv_scores_raw: list[float],
) -> _TrainFitResult:
    target = feature_target.target
    model = _fit_stage_model(
        train_df,
        feature_target=feature_target,
        model_settings=model_settings,
    )
    train_eval_df = _score_train_frame(
        train_df,
        model,
        feature_target=feature_target,
        signal_settings=signal_settings,
    )
    updated_signal_direction, train_ic_raw_stats = _resolve_train_signal_direction(
        train_eval_df,
        target=target,
        signal_settings=signal_settings,
    )
    train_signal_col = _train_signal_column(train_eval_df, updated_signal_direction)
    cv_scores_adj = _adjust_cv_scores(cv_scores_raw, updated_signal_direction)
    train_ic_series, train_ic_stats, train_pearson_ic_series, train_pearson_ic_stats = (
        _train_ic_report(
            train_eval_df,
            target=target,
            train_signal_col=train_signal_col,
            signal_settings=signal_settings,
        )
    )

    return _TrainFitResult(
        model=model,
        train_eval_df=train_eval_df,
        updated_signal_direction=updated_signal_direction,
        train_signal_col=train_signal_col,
        train_ic_raw_stats=train_ic_raw_stats,
        train_ic_series=train_ic_series,
        train_ic_stats=train_ic_stats,
        train_pearson_ic_series=train_pearson_ic_series,
        train_pearson_ic_stats=train_pearson_ic_stats,
        cv_scores_adj=cv_scores_adj,
    )
