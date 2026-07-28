"""Train/eval stage: evaluation orchestration and diagnostics.

Private helpers that run CV + model fit, walk-forward evaluation, the main
period evaluation, rolling/recency diagnostics, and feature-importance
diagnostics, plus the ``run_train_eval_stage`` entry points. Split out of the
historical single-file :mod:`alpha_research.train_eval_stage` implementation to
keep individual files smaller while preserving the exact public/private symbol
surface.

The monkeypatched helpers ``time_series_cv_ic`` / ``_fit_model_and_score_train``
/ ``feature_importance_frame`` are referenced through the
:mod:`alpha_research.train_eval_stage` module object (they are re-exported
there) so that ``monkeypatch.setattr(train_eval_stage, ...)`` keeps working
after the split.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from . import train_eval_stage as _stage
from ._train_eval_stage_context import (
    _build_period_eval_context,
    _build_walk_forward_context,
    _industry_exposure_columns,
    _test_window_full_data,
)
from .backends import FittedModelHandle, TrainerBackend
from .train_eval_contracts import TrainEvalRequest
from .train_eval_diagnostics import (
    _annotate_positions_window,
    _compute_rolling_ic,
    _compute_rolling_sharpe,
    _latest_rolling_stats,
    _summarize_walk_forward_feature_stability,
    build_recency_diagnostics,
)
from .train_eval_fit import _TrainFitResult
from .walk_forward import _evaluate_walk_forward_window
from .walk_forward_windows import build_walk_forward_windows

logger = logging.getLogger("alpha_research")


def _run_walk_forward_evaluation(
    request: TrainEvalRequest,
    *,
    updated_signal_direction: float,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    walk_forward_settings = request.walk_forward
    walk_forward_results: list[dict] = []
    walk_forward_importance_rows: list[dict[str, Any]] = []
    if walk_forward_settings.wf_enabled:
        walk_forward_context = _build_walk_forward_context(
            request,
            updated_signal_direction=updated_signal_direction,
        )
        try:
            walk_forward_test_size = float(walk_forward_settings.wf_test_size)
        except (TypeError, ValueError):
            walk_forward_test_size = None
        windows = build_walk_forward_windows(
            request.data.all_dates,
            walk_forward_test_size,
            walk_forward_settings.wf_n_windows,
            walk_forward_settings.wf_step_size,
            walk_forward_settings.effective_gap_steps,
            walk_forward_settings.wf_anchor_end,
        )
        if not windows:
            logger.info("Walk-forward evaluation skipped: insufficient windows.")
        else:
            if len(windows) < walk_forward_settings.wf_n_windows:
                logger.warning(
                    "Walk-forward requested %s windows but only %s fit "
                    "(test_size=%s, step_size=%s, anchor_end=%s). "
                    "Reduce eval.test_size / eval.walk_forward.test_size, "
                    "set a smaller eval.walk_forward.step_size, or lower n_windows.",
                    walk_forward_settings.wf_n_windows,
                    len(windows),
                    walk_forward_test_size,
                    walk_forward_settings.wf_step_size,
                    walk_forward_settings.wf_anchor_end,
                )
            logger.info("Walk-forward evaluation: %s windows.", len(windows))
            for window_meta in windows:
                window_result, window_importance_rows = _evaluate_walk_forward_window(
                    window_meta,
                    context=walk_forward_context,
                )
                walk_forward_results.append(window_result)
                walk_forward_importance_rows.extend(window_importance_rows)

    walk_forward_importance_df = pd.DataFrame(walk_forward_importance_rows)
    walk_forward_feature_stability_df = _summarize_walk_forward_feature_stability(
        walk_forward_importance_df,
        walk_forward_settings.wf_feature_top_k,
    )
    return walk_forward_results, walk_forward_importance_df, walk_forward_feature_stability_df


def _run_cv_and_fit_model(request: TrainEvalRequest) -> _TrainFitResult:
    data = request.data
    feature_target = request.feature_target
    model_settings = request.model
    signal_settings = request.signal

    logger.info("Time-series cross-validation (IC) ...")
    cv_scores_raw = _stage.time_series_cv_ic(
        data.train_df,
        feature_target.features,
        feature_target.target,
        model_settings.n_splits,
        model_settings.embargo_steps,
        model_settings.purge_steps,
        model_settings.model_cfg,
        1.0,
        sample_weight_mode=model_settings.sample_weight_mode,
        sample_weight_params=model_settings.sample_weight_params,
        train_window_mode=model_settings.train_window_mode,
        train_window_size=model_settings.train_window_size,
        train_window_unit=model_settings.train_window_unit,
        fit_target_col=feature_target.train_target,
        score_postprocess_method=signal_settings.score_postprocess_method,
        score_postprocess_columns=signal_settings.score_postprocess_columns,
        score_postprocess_strength=signal_settings.score_postprocess_strength,
        score_postprocess_min_obs=signal_settings.score_postprocess_min_obs,
        cv_purge_mode=model_settings.cv_purge_mode,
        label_horizon_mode=request.period.label_horizon_mode,
        label_horizon_days=int(request.period.label_horizon_effective),
        label_shift_days=request.backtest.label_shift_days,
        all_trade_dates=data.all_dates,
        trainer_backend=request.services.trainer_backend,
    )
    if cv_scores_raw:
        logger.info(
            "CV IC (raw): mean=%.4f, std=%.4f",
            np.nanmean(cv_scores_raw),
            np.nanstd(cv_scores_raw),
        )
        logger.info("CV fold ICs (raw): %s", [f"{s:.4f}" for s in cv_scores_raw])
    else:
        logger.info("CV IC not available - insufficient data after embargo/purge.")

    updated_signal_direction = signal_settings.signal_direction
    if signal_settings.signal_direction_mode == "cv_ic" and cv_scores_raw:
        cv_mean = float(np.nanmean(cv_scores_raw))
        if (
            np.isfinite(cv_mean)
            and cv_mean != 0
            and abs(cv_mean) >= signal_settings.min_abs_ic_to_flip
        ):
            updated_signal_direction = float(np.sign(cv_mean))
            logger.info("Signal direction set from CV IC: %s", updated_signal_direction)
        else:
            logger.info(
                "CV IC mean below threshold (|mean| < %.4f); keeping signal direction: %s",
                signal_settings.min_abs_ic_to_flip,
                updated_signal_direction,
            )

    fit_state = _stage._fit_model_and_score_train(
        data.train_df,
        feature_target=feature_target,
        model_settings=model_settings,
        signal_settings=signal_settings,
        cv_scores_raw=cv_scores_raw,
        trainer_backend=request.services.trainer_backend,
    )
    return _TrainFitResult(
        model=fit_state.model,
        train_eval_df=fit_state.train_eval_df,
        updated_signal_direction=fit_state.updated_signal_direction,
        train_signal_col=fit_state.train_signal_col,
        train_ic_raw_stats=fit_state.train_ic_raw_stats,
        train_ic_series=fit_state.train_ic_series,
        train_ic_stats=fit_state.train_ic_stats,
        train_pearson_ic_series=fit_state.train_pearson_ic_series,
        train_pearson_ic_stats=fit_state.train_pearson_ic_stats,
        cv_scores_raw=cv_scores_raw,
        cv_scores_adj=fit_state.cv_scores_adj,
        model_handle=fit_state.model_handle,
    )


def _prepare_stage_live_state(
    request: TrainEvalRequest,
    model: Any,
    *,
    updated_signal_direction: float,
) -> dict[str, Any]:
    feature_target = request.feature_target
    model_settings = request.model
    signal_settings = request.signal
    live_settings = request.live
    backtest_settings = request.backtest
    return request.services.live_snapshot_fn(
        request.data.df_features,
        model,
        context={
            "live_enabled": live_settings.live_enabled,
            "live_as_of_token": live_settings.live_as_of,
            "live_signal_asof_token": live_settings.live_signal_asof,
            "live_entry_date_token": live_settings.live_entry_date,
            "market": live_settings.market,
            "provider": live_settings.provider,
            "target": feature_target.target,
            "live_train_mode": live_settings.live_train_mode,
            "model_type": model_settings.model_type,
            "model_params": model_settings.model_params,
            "train_window_mode": model_settings.train_window_mode,
            "train_window_size": model_settings.train_window_size,
            "train_window_unit": model_settings.train_window_unit,
            "sample_weight_mode": model_settings.sample_weight_mode,
            "sample_weight_params": model_settings.sample_weight_params,
            "train_target": feature_target.train_target,
            "features": feature_target.features,
            "signal_direction": updated_signal_direction,
            "score_postprocess_method": signal_settings.score_postprocess_method,
            "score_postprocess_columns": signal_settings.score_postprocess_columns,
            "score_postprocess_strength": signal_settings.score_postprocess_strength,
            "score_postprocess_min_obs": signal_settings.score_postprocess_min_obs,
            "backtest_rebalance_frequency": backtest_settings.backtest_rebalance_frequency,
            "min_symbols_per_date": live_settings.min_symbols_per_date,
            "price_col": feature_target.price_col,
            "backtest_top_k": backtest_settings.backtest_top_k,
            "label_shift_days": backtest_settings.label_shift_days,
            "backtest_weighting": backtest_settings.backtest_weighting,
            "backtest_buffer_exit": backtest_settings.backtest_buffer_exit,
            "backtest_buffer_entry": backtest_settings.backtest_buffer_entry,
            "backtest_long_only": backtest_settings.backtest_long_only,
            "backtest_short_k": backtest_settings.backtest_short_k,
            "backtest_tradable_col": backtest_settings.backtest_tradable_col,
            "backtest_group_col": backtest_settings.backtest_group_col,
            "backtest_max_names_per_group": backtest_settings.backtest_max_names_per_group,
            "backtest_liquidity_floor_col": backtest_settings.backtest_liquidity_floor_col,
            "backtest_liquidity_floor_quantile": (
                backtest_settings.backtest_liquidity_floor_quantile
            ),
            "backtest_weighting_liquidity_col": backtest_settings.backtest_weighting_liquidity_col,
            "backtest_max_turnover_per_rebalance": (
                backtest_settings.backtest_max_turnover_per_rebalance
            ),
            "backtest_selection_tiebreak_col": (backtest_settings.backtest_selection_tiebreak_col),
            "backtest_selection_score_bucket_size": (
                backtest_settings.backtest_selection_score_bucket_size
            ),
            "backtest_selection_score_margin": (backtest_settings.backtest_selection_score_margin),
            "backtest_selection_score_margin_rank_limit": (
                backtest_settings.backtest_selection_score_margin_rank_limit
            ),
            "post_buffer_exposure_repair": backtest_settings.backtest_post_buffer_exposure_repair,
            "cash_gross_overlay": backtest_settings.backtest_cash_gross_overlay,
            "freshness_overlay": backtest_settings.backtest_freshness_overlay,
            "backtest_pricing_df": request.data.backtest_pricing_df,
            "benchmark_df": request.data.benchmark_df,
            "benchmark_return_series": request.data.benchmark_return_series,
            "industry_source_df": request.data.industry_source_df,
            "industry_columns": _industry_exposure_columns(request.data),
            "fundamentals_mcap_col": feature_target.fundamentals_mcap_col,
            "execution_model": backtest_settings.execution_model,
        },
    )


def _cv_stats(
    cv_scores_raw: list[float],
    cv_scores_adj: list[float] | None,
    updated_signal_direction: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[float] | None]:
    if not cv_scores_raw:
        return None, None, cv_scores_adj
    cv_stats_raw = {
        "mean": float(np.nanmean(cv_scores_raw)),
        "std": float(np.nanstd(cv_scores_raw)),
        "scores": [float(score) for score in cv_scores_raw],
    }
    if cv_scores_adj is None:
        cv_scores_adj = [float(score) * updated_signal_direction for score in cv_scores_raw]
    cv_stats = {
        "mean": float(np.nanmean(cv_scores_adj)),
        "std": float(np.nanstd(cv_scores_adj)),
        "scores": [float(score) for score in cv_scores_adj],
    }
    return cv_stats_raw, cv_stats, cv_scores_adj


def _require_live_positions_if_needed(
    *,
    live_enabled: bool,
    backtest_enabled: bool,
    live_positions_ready: bool,
) -> None:
    if live_enabled and not backtest_enabled and not live_positions_ready:
        raise SystemExit(
            "live.enabled=true but no live positions were generated; "
            "refusing to fall back to backtest holdings."
        )


def _evaluate_main_period(
    request: TrainEvalRequest,
    model: Any,
    *,
    live_state: dict[str, Any],
    updated_signal_direction: float,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    data = request.data
    backtest_settings = request.backtest
    period_settings = request.period
    backtest_signal_direction = (
        updated_signal_direction
        if backtest_settings.backtest_signal_direction_raw is None
        else backtest_settings.backtest_signal_direction_raw
    )
    period_eval_context = _build_period_eval_context(
        request,
        live_state=live_state,
        updated_signal_direction=updated_signal_direction,
        backtest_signal_direction=backtest_signal_direction,
    )
    test_df_full = _test_window_full_data(data)
    eval_main = request.services.period_eval_fn(
        "Test",
        model,
        test_df_full,
        data.test_dates,
        context=period_eval_context,
        run_perm_test=period_settings.perm_test_enabled,
        perm_train_df=data.train_df,
        perm_test_df=data.test_df,
        allow_live_fallback=True,
    )
    return backtest_signal_direction, period_eval_context, eval_main


def _rolling_eval_summaries(
    eval_main: dict[str, Any],
    rolling_windows_months: list[int],
) -> tuple[dict[str, Any], float, dict[str, Any], dict[str, Any], dict[str, Any]]:
    rolling_ic_results, rolling_ic_obs_per_year = _compute_rolling_ic(
        eval_main["ic_series"], rolling_windows_months
    )
    rolling_ic_latest = {
        label: _latest_rolling_stats(frame, ["ic_mean", "ic_ir"])
        for label, frame in rolling_ic_results.items()
    }
    rolling_sharpe_results = {}
    rolling_sharpe_latest = {}
    if eval_main["bt_stats"] is not None and not eval_main["bt_net_series"].empty:
        periods_per_year = eval_main["bt_stats"].get("periods_per_year", np.nan)
        rolling_sharpe_results = _compute_rolling_sharpe(
            eval_main["bt_net_series"], rolling_windows_months, periods_per_year
        )
        rolling_sharpe_latest = {
            label: _latest_rolling_stats(frame, ["mean", "std", "sharpe"])
            for label, frame in rolling_sharpe_results.items()
        }
    return (
        rolling_ic_results,
        rolling_ic_obs_per_year,
        rolling_ic_latest,
        rolling_sharpe_results,
        rolling_sharpe_latest,
    )


def _recency_eval_diagnostics(
    eval_main: dict[str, Any],
    recency_windows: list[str],
) -> pd.DataFrame:
    periods_per_year = np.nan
    if eval_main["bt_stats"] is not None:
        periods_per_year = eval_main["bt_stats"].get("periods_per_year", np.nan)
    return build_recency_diagnostics(
        window_labels=recency_windows,
        ic_series=eval_main["ic_series"],
        returns=eval_main["bt_net_series"],
        active_returns=eval_main["bt_active_series"],
        turnover=eval_main["bt_turnover_series"],
        periods_per_year=periods_per_year,
    )


def _annotated_stage_positions(
    eval_main: dict[str, Any],
    live_state: dict[str, Any],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    positions_by_rebalance = eval_main["positions_by_rebalance"]
    positions_by_rebalance_live = live_state["positions_by_rebalance_live"]
    if positions_by_rebalance is not None and not positions_by_rebalance.empty:
        positions_by_rebalance = _annotate_positions_window(positions_by_rebalance)
    if positions_by_rebalance_live is not None and not positions_by_rebalance_live.empty:
        positions_by_rebalance_live = _annotate_positions_window(positions_by_rebalance_live)
    return positions_by_rebalance, positions_by_rebalance_live


def _prediction_diagnostics(
    eval_scored_data: pd.DataFrame | None,
) -> tuple[int | None, bool | None]:
    if eval_scored_data is None or eval_scored_data.empty or "pred" not in eval_scored_data.columns:
        return None, None
    pred_nunique = int(eval_scored_data["pred"].nunique(dropna=True))
    return pred_nunique, pred_nunique <= 1


def _importance_diagnostics(importance_df: pd.DataFrame) -> tuple[int | None, bool | None]:
    if importance_df.empty or "importance" not in importance_df.columns:
        return None, None
    importance_values = pd.to_numeric(importance_df["importance"], errors="coerce").fillna(0.0)
    feature_importance_nonzero = int((importance_values.abs() > 0.0).sum())
    return feature_importance_nonzero, feature_importance_nonzero == 0


def _feature_importance_diagnostics(
    model: Any,
    features: list[str],
    eval_scored_data: pd.DataFrame | None,
    *,
    trainer_backend: TrainerBackend | None = None,
    model_handle: FittedModelHandle | None = None,
) -> tuple[pd.DataFrame, str, int | None, bool | None, int | None, bool | None]:
    logger.info("Feature importance:")
    if trainer_backend is not None and model_handle is not None:
        importance = trainer_backend.feature_importance(model_handle, features=features)
        importance_df, importance_source = importance.frame, importance.source
    else:
        importance_df, importance_source = _stage.feature_importance_frame(model, features)
    logger.info("Feature importance source: %s", importance_source)
    for _, row in importance_df.iterrows():
        logger.info("  %-20s: %.4f", row["feature"], float(row["importance"]))

    pred_nunique, constant_prediction = _prediction_diagnostics(eval_scored_data)
    feature_importance_nonzero, zero_feature_importance = _importance_diagnostics(importance_df)
    return (
        importance_df,
        importance_source,
        pred_nunique,
        constant_prediction,
        feature_importance_nonzero,
        zero_feature_importance,
    )
