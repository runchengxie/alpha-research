from __future__ import annotations

import logging
import sys
from typing import Any, cast

import numpy as np

from ._feature_dataset_config import (
    _build_bucket_eval_frame,
    _build_feature_availability_diagnostics,
    _build_feature_dataset_config,
    _build_passthrough_cols,
    _FeatureDatasetConfig,
    _FeatureDatasetPrepared,
    _log_feature_availability_warning,
    _log_modeling_dataset_summary,
    _resolve_engineered_features,
)
from .backends import DatasetBackend, DatasetBuildRequest, NativeDatasetBackend
from .compat import ensure_numpy_nan_alias
from .dataset_sampling import (
    _normalize_extra_sample_dates,
    apply_feature_missing_fill,
    build_modeling_dataset,
    prepare_backtest_pricing_frame,
)
from .feature_engineering import engineer_symbol_features

ensure_numpy_nan_alias()
import pandas as pd  # noqa: E402

logger = logging.getLogger("alpha_research")


def _engineer_features_by_symbol(
    *,
    df: pd.DataFrame,
    features: list[str],
    feature_params: dict,
    price_col: str,
    target: str,
    label_shift_days: int,
    label_horizon_days: int,
    label_horizon_mode: str,
    label_next_rebalance_map: dict[pd.Timestamp, pd.Timestamp] | None,
    modeling_date_candidates: set[pd.Timestamp] | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    symbol_count = int(df["symbol"].nunique())
    input_rows = 0
    output_rows = 0
    for idx, (_symbol, group) in enumerate(df.groupby("symbol", sort=False), start=1):
        engineered = engineer_symbol_features(
            group,
            features=features,
            feature_params=feature_params,
            price_col=price_col,
            target=target,
            label_shift_days=label_shift_days,
            label_horizon_days=label_horizon_days,
            label_horizon_mode=label_horizon_mode,
            label_next_rebalance_map=label_next_rebalance_map,
        )
        input_rows += len(engineered)
        if modeling_date_candidates is not None:
            engineered = cast(
                pd.DataFrame,
                engineered[
                    cast(pd.Series, engineered["trade_date"]).isin(list(modeling_date_candidates))
                ].copy(),
            )
        output_rows += len(engineered)
        frames.append(engineered)
        if idx % 500 == 0:
            logger.info("Engineered features for %s/%s symbols ...", idx, symbol_count)
    if modeling_date_candidates is not None:
        logger.info(
            "Applied per-symbol rebalance candidate prefilter after feature engineering: "
            "%s -> %s rows (candidate_dates=%s, tail_days_per_period=%s).",
            input_rows,
            output_rows,
            len(modeling_date_candidates),
            5,
        )
    if not frames:
        return df.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def _validate_feature_dataset_inputs(
    df: pd.DataFrame,
    *,
    price_col: str,
    features: list[str],
) -> None:
    if price_col not in df.columns:
        if price_col == "tr_close":
            sys.exit(
                "Price column 'tr_close' not found in data. "
                "Configure data.rqdata.ex_factors_dir for local RQData assets, "
                "or provide tr_close directly in the source daily data."
            )
        sys.exit(f"Price column '{price_col}' not found in data.")
    if not features:
        sys.exit("Feature list is empty.")


def _prepare_modeling_date_candidates(
    df: pd.DataFrame,
    *,
    sample_on_rebalance_dates: bool,
    rebalance_frequency: str,
    extra_no_label_dates: pd.DatetimeIndex,
) -> set[pd.Timestamp] | None:
    if not sample_on_rebalance_dates:
        return None

    from ._feature_dataset_config import _build_rebalance_tail_candidate_dates

    modeling_date_candidates = _build_rebalance_tail_candidate_dates(
        cast(pd.Series, df["trade_date"]),
        rebalance_frequency=rebalance_frequency,
    )
    if modeling_date_candidates is not None and not extra_no_label_dates.empty:
        modeling_date_candidates.update(
            cast(pd.Timestamp, pd.Timestamp(date)).normalize() for date in extra_no_label_dates
        )
    if modeling_date_candidates is not None:
        logger.info(
            "Prepared per-symbol rebalance candidate dates: %s dates "
            "(frequency=%s, tail_days_per_period=%s).",
            len(modeling_date_candidates),
            rebalance_frequency,
            5,
        )
    return modeling_date_candidates


def _build_research_dataset(
    *,
    raw_panel: pd.DataFrame,
    modeling_state: dict[str, Any],
    backtest_pricing_df: pd.DataFrame,
    features: list[str],
    target: str,
    train_target: str,
    missing_fill_features: list[str],
    feature_missing_method: str,
    feature_missing_add_indicators: bool,
    winsorize_pct: float | None,
    cs_method: str,
    cs_winsorize_pct: float | None,
    train_target_transform: str,
    train_target_group_cols: list[str] | None,
    universe_by_date_applied: bool,
    sample_on_rebalance_dates: bool,
    min_symbols_per_date: int,
    raw_daily_panel_rows: int,
    dataset_backend: DatasetBackend | None,
):
    backend = dataset_backend or NativeDatasetBackend()
    research_dataset = backend.build(
        DatasetBuildRequest(
            raw_panel=raw_panel,
            modeling_state=modeling_state,
            backtest_pricing_frame=backtest_pricing_df,
            features=tuple(features),
            target=target,
            train_target=train_target,
            missing_fill_features=tuple(missing_fill_features),
            feature_missing_method=feature_missing_method,
            feature_missing_add_indicators=feature_missing_add_indicators,
            winsorize_pct=winsorize_pct,
            cs_method=cs_method,
            cs_winsorize_pct=cs_winsorize_pct,
            train_target_transform=train_target_transform,
            train_target_group_cols=(
                tuple(train_target_group_cols) if train_target_group_cols is not None else None
            ),
            universe_by_date_applied=universe_by_date_applied,
            sample_on_rebalance_dates=sample_on_rebalance_dates,
            min_symbols_per_date=min_symbols_per_date,
        )
    )
    dataset_lifecycle = research_dataset.summary()
    metadata = dataset_lifecycle.setdefault("metadata", {})
    metadata["backend"] = backend.backend_id
    metadata["raw_daily_panel_rows"] = raw_daily_panel_rows
    metadata["engineered_feature_label_rows"] = len(raw_panel)
    return research_dataset, dataset_lifecycle


def _prepare_engineered_feature_dataset(
    df: pd.DataFrame,
    *,
    features: list[str],
    config: _FeatureDatasetConfig,
) -> _FeatureDatasetPrepared:
    _validate_feature_dataset_inputs(df, price_col=config.price_col, features=features)
    raw_daily_panel_rows = len(df)
    raw_reference_trade_dates = np.sort(pd.to_datetime(df["trade_date"].unique()).to_numpy())
    backtest_pricing_df, price_passthrough_cols = prepare_backtest_pricing_frame(
        df=df,
        price_col=config.price_col,
        execution_pricing_cols=config.execution_pricing_cols,
        backtest_tradable_col=config.backtest_tradable_col,
    )
    extra_no_label_dates = _normalize_extra_sample_dates(config.extra_sample_dates_without_target)
    modeling_date_candidates = _prepare_modeling_date_candidates(
        df,
        sample_on_rebalance_dates=config.sample_on_rebalance_dates,
        rebalance_frequency=config.rebalance_frequency,
        extra_no_label_dates=extra_no_label_dates,
    )
    df = _engineer_features_by_symbol(
        df=df,
        features=features,
        feature_params=config.feature_params,
        price_col=config.price_col,
        target=config.target,
        label_shift_days=config.label_shift_days,
        label_horizon_days=config.label_horizon_days,
        label_horizon_mode=config.label_horizon_mode,
        label_next_rebalance_map=config.label_next_rebalance_map,
        modeling_date_candidates=modeling_date_candidates,
    )
    features = _resolve_engineered_features(
        df,
        features=features,
        fundamentals_allow_missing=config.fundamentals_allow_missing,
    )
    eval_extra_df, bucket_cols = _build_bucket_eval_frame(
        df,
        bucket_ic_enabled=config.bucket_ic_enabled,
        bucket_ic_schemes=config.bucket_ic_schemes,
    )
    passthrough_cols = _build_passthrough_cols(
        fundamentals_cols=config.fundamentals_cols,
        industry_cols=config.industry_cols,
        extra_passthrough_cols=config.extra_passthrough_cols,
    )
    df, features, missing_fill_features = apply_feature_missing_fill(
        df=df,
        features=features,
        feature_missing_features=config.feature_missing_features,
        feature_missing_method=config.feature_missing_method,
        feature_missing_add_indicators=config.feature_missing_add_indicators,
        feature_missing_suffix=config.feature_missing_suffix,
    )
    feature_availability_diagnostics = _build_feature_availability_diagnostics(
        df,
        price_col=config.price_col,
        features=features,
    )
    _log_feature_availability_warning(feature_availability_diagnostics)
    return _FeatureDatasetPrepared(
        df=df,
        features=features,
        backtest_pricing_df=backtest_pricing_df,
        bucket_cols=bucket_cols,
        passthrough_cols=passthrough_cols,
        price_passthrough_cols=price_passthrough_cols,
        eval_extra_df=eval_extra_df,
        feature_availability_diagnostics=feature_availability_diagnostics,
        missing_fill_features=missing_fill_features,
        raw_daily_panel_rows=raw_daily_panel_rows,
        raw_reference_trade_dates=raw_reference_trade_dates,
        extra_no_label_dates=extra_no_label_dates,
    )


def _build_feature_modeling_state(
    prepared: _FeatureDatasetPrepared,
    *,
    config: _FeatureDatasetConfig,
) -> dict[str, Any]:
    return build_modeling_dataset(
        df=prepared.df,
        price_col=config.price_col,
        target=config.target,
        train_target=config.train_target,
        train_target_transform=config.train_target_transform,
        train_target_group_cols=config.train_target_group_cols,
        features=prepared.features,
        price_passthrough_cols=prepared.price_passthrough_cols,
        passthrough_cols=prepared.passthrough_cols,
        winsorize_pct=config.winsorize_pct,
        cs_method=config.cs_method,
        cs_winsorize_pct=config.cs_winsorize_pct,
        sample_on_rebalance_dates=config.sample_on_rebalance_dates,
        rebalance_frequency=config.rebalance_frequency,
        min_symbols_per_date=config.min_symbols_per_date,
        universe_by_date=config.universe_by_date,
        eval_extra_df=prepared.eval_extra_df,
        reference_trade_dates=prepared.raw_reference_trade_dates,
        extra_sample_dates_without_target=list(prepared.extra_no_label_dates),
    )


def _prepare_feature_dataset(
    *,
    df: pd.DataFrame,
    features: list[str],
    feature_params: dict,
    price_col: str,
    target: str,
    label_shift_days: int,
    label_horizon_days: int,
    label_horizon_mode: str,
    label_next_rebalance_map: dict[pd.Timestamp, pd.Timestamp] | None,
    fundamentals_allow_missing: bool,
    bucket_ic_enabled: bool,
    bucket_ic_schemes: list[dict[str, Any]],
    feature_missing_features: list[str],
    feature_missing_method: str,
    feature_missing_add_indicators: bool,
    feature_missing_suffix: str,
    fundamentals_cols: list[str],
    industry_cols: list[str],
    extra_passthrough_cols: list[str] | None = None,
    execution_pricing_cols: set[str],
    backtest_tradable_col: str | None,
    universe_by_date: pd.DataFrame | None,
    winsorize_pct: float | None,
    cs_method: str,
    cs_winsorize_pct: float | None,
    train_target: str,
    train_target_transform: str,
    train_target_group_cols: list[str] | None = None,
    sample_on_rebalance_dates: bool,
    rebalance_frequency: str,
    min_symbols_per_date: int,
    extra_sample_dates_without_target: list[object] | None = None,
    dataset_backend: DatasetBackend | None = None,
) -> dict[str, Any]:
    logger.info("Engineering features ...")
    config = _build_feature_dataset_config(locals())
    prepared = _prepare_engineered_feature_dataset(
        df,
        features=list(features),
        config=config,
    )
    modeling_state = _build_feature_modeling_state(prepared, config=config)
    _log_modeling_dataset_summary(
        modeling_state,
        min_symbols_per_date=config.min_symbols_per_date,
        sample_on_rebalance_dates=config.sample_on_rebalance_dates,
        feature_availability_diagnostics=prepared.feature_availability_diagnostics,
    )
    research_dataset, dataset_lifecycle = _build_research_dataset(
        raw_panel=prepared.df,
        modeling_state=modeling_state,
        backtest_pricing_df=prepared.backtest_pricing_df,
        features=prepared.features,
        target=config.target,
        train_target=config.train_target,
        missing_fill_features=prepared.missing_fill_features,
        feature_missing_method=config.feature_missing_method,
        feature_missing_add_indicators=config.feature_missing_add_indicators,
        winsorize_pct=config.winsorize_pct,
        cs_method=config.cs_method,
        cs_winsorize_pct=config.cs_winsorize_pct,
        train_target_transform=config.train_target_transform,
        train_target_group_cols=config.train_target_group_cols,
        universe_by_date_applied=config.universe_by_date is not None,
        sample_on_rebalance_dates=config.sample_on_rebalance_dates,
        min_symbols_per_date=config.min_symbols_per_date,
        raw_daily_panel_rows=prepared.raw_daily_panel_rows,
        dataset_backend=dataset_backend,
    )

    return {
        "features": prepared.features,
        "backtest_pricing_df": prepared.backtest_pricing_df,
        "bucket_cols": prepared.bucket_cols,
        "passthrough_cols": prepared.passthrough_cols,
        "price_passthrough_cols": prepared.price_passthrough_cols,
        "feature_availability_diagnostics": prepared.feature_availability_diagnostics,
        "dataset_lifecycle": dataset_lifecycle,
        "research_dataset": research_dataset,
        **modeling_state,
    }
