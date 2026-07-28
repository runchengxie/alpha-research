"""Train/eval stage orchestration.

This module is a thin public surface for the train/eval stage implementation.
The historical single-file implementation has been split into private
submodules (``_train_eval_stage_context`` / ``_train_eval_stage_eval``) to keep
individual files smaller while preserving the exact public and private symbol
surface.

The monkeypatched helpers ``time_series_cv_ic`` / ``_fit_model_and_score_train``
/ ``feature_importance_frame`` are imported first (before the submodules) so
that ``monkeypatch.setattr(train_eval_stage, ...)`` keeps replacing the same
module attributes the submodules resolve through ``alpha_research.train_eval_stage``.
Everything below is re-exported so existing ``alpha_research.train_eval_stage``
imports keep working unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from ._train_eval_stage_context import (
    _build_period_eval_context,
    _build_walk_forward_context,
    _industry_exposure_columns,
    _test_window_full_data,
)
from ._train_eval_stage_eval import (
    _annotated_stage_positions,
    _cv_stats,
    _evaluate_main_period,
    _feature_importance_diagnostics,
    _importance_diagnostics,
    _prediction_diagnostics,
    _prepare_stage_live_state,
    _recency_eval_diagnostics,
    _require_live_positions_if_needed,
    _rolling_eval_summaries,
    _run_cv_and_fit_model,
    _run_walk_forward_evaluation,
)
from .backends import FittedModelHandle, TrainerBackend
from .modeling import feature_importance_frame
from .split import time_series_cv_ic
from .train_eval_contracts import TrainEvalData, TrainEvalRequest
from .train_eval_diagnostics import (
    _annotate_positions_window,
    _compute_rolling_ic,
    _compute_rolling_sharpe,
    _latest_rolling_stats,
    _summarize_walk_forward_feature_stability,
    build_recency_diagnostics,
)
from .train_eval_fit import (
    _TrainFitResult,
    fit_model_and_score_train as _fit_model_and_score_train,
)
from .train_eval_request_builder import (
    train_eval_request_from_kwargs as _build_train_eval_request_from_kwargs,
)
from .train_eval_result import build_train_eval_stage_result as _build_train_eval_stage_result
from .walk_forward import _evaluate_walk_forward_window
from .walk_forward_windows import build_walk_forward_windows

__all__ = [
    "FittedModelHandle",
    "TrainEvalData",
    "TrainEvalRequest",
    "TrainerBackend",
    "_TrainFitResult",
    "_annotate_positions_window",
    "_build_period_eval_context",
    "_build_train_eval_stage_result",
    "_build_walk_forward_context",
    "_compute_rolling_ic",
    "_compute_rolling_sharpe",
    "_cv_stats",
    "_evaluate_main_period",
    "_evaluate_walk_forward_window",
    "_feature_importance_diagnostics",
    "_fit_model_and_score_train",
    "_importance_diagnostics",
    "_industry_exposure_columns",
    "_latest_rolling_stats",
    "_prediction_diagnostics",
    "_prepare_stage_live_state",
    "_recency_eval_diagnostics",
    "_require_live_positions_if_needed",
    "_rolling_eval_summaries",
    "_run_cv_and_fit_model",
    "_run_train_eval_stage_impl",
    "_run_walk_forward_evaluation",
    "_summarize_walk_forward_feature_stability",
    "_test_window_full_data",
    "_train_eval_request_from_kwargs",
    "build_recency_diagnostics",
    "build_walk_forward_windows",
    "feature_importance_frame",
    "run_train_eval_stage",
    "time_series_cv_ic",
]


def run_train_eval_stage(
    *,
    request: TrainEvalRequest | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if request is not None:
        if kwargs:
            raise TypeError("Pass either request or keyword stage fields, not both.")
        return _run_train_eval_stage_impl(request)
    return _run_train_eval_stage_impl(_train_eval_request_from_kwargs(kwargs))


def _train_eval_request_from_kwargs(kwargs: dict[str, Any]) -> TrainEvalRequest:
    return _build_train_eval_request_from_kwargs(kwargs)


def _run_train_eval_stage_impl(request: TrainEvalRequest) -> dict[str, Any]:
    data = request.data
    feature_target = request.feature_target
    live_settings = request.live
    backtest_settings = request.backtest
    period_settings = request.period

    features = feature_target.features
    live_enabled = live_settings.live_enabled
    backtest_enabled = backtest_settings.backtest_enabled
    rolling_windows_months = period_settings.rolling_windows_months
    recency_windows = period_settings.recency_windows

    fit_state = _run_cv_and_fit_model(request)
    model = fit_state.model
    updated_signal_direction = fit_state.updated_signal_direction
    train_ic_raw_stats = fit_state.train_ic_raw_stats
    train_ic_series = fit_state.train_ic_series
    train_ic_stats = fit_state.train_ic_stats
    train_pearson_ic_series = fit_state.train_pearson_ic_series
    train_pearson_ic_stats = fit_state.train_pearson_ic_stats
    cv_scores_raw = fit_state.cv_scores_raw or []
    cv_scores_adj = fit_state.cv_scores_adj

    logger = _logger
    logger.info("Evaluating model on train/test sets ...")
    live_state = _prepare_stage_live_state(
        request,
        model,
        updated_signal_direction=updated_signal_direction,
    )
    live_positions_ready = bool(live_state["live_positions_ready"])
    _require_live_positions_if_needed(
        live_enabled=live_enabled,
        backtest_enabled=backtest_enabled,
        live_positions_ready=live_positions_ready,
    )

    backtest_signal_direction, period_eval_context, eval_main = _evaluate_main_period(
        request,
        model,
        live_state=live_state,
        updated_signal_direction=updated_signal_direction,
    )

    (
        rolling_ic_results,
        rolling_ic_obs_per_year,
        rolling_ic_latest,
        rolling_sharpe_results,
        rolling_sharpe_latest,
    ) = _rolling_eval_summaries(eval_main, rolling_windows_months)
    recency_diagnostics = _recency_eval_diagnostics(eval_main, recency_windows)
    positions_by_rebalance, positions_by_rebalance_live = _annotated_stage_positions(
        eval_main,
        live_state,
    )

    cv_stats_raw, cv_stats, cv_scores_adj = _cv_stats(
        cv_scores_raw,
        cv_scores_adj,
        updated_signal_direction,
    )

    (
        walk_forward_results,
        walk_forward_importance_df,
        walk_forward_feature_stability_df,
    ) = _run_walk_forward_evaluation(
        request,
        updated_signal_direction=updated_signal_direction,
    )

    eval_scored_data = eval_main["scored_data"]
    (
        importance_df,
        importance_source,
        pred_nunique,
        constant_prediction,
        feature_importance_nonzero,
        zero_feature_importance,
    ) = _feature_importance_diagnostics(
        model,
        features,
        eval_scored_data,
        trainer_backend=request.services.trainer_backend,
        model_handle=fit_state.model_handle,
    )

    return _build_train_eval_stage_result(locals())


_logger = logging.getLogger("alpha_research")
