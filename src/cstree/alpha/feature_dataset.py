from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ..compat import ensure_numpy_nan_alias
from ..pipeline.dataset_sampling import (
    _normalize_extra_sample_dates,
    apply_feature_missing_fill,
    build_modeling_dataset,
    prepare_backtest_pricing_frame,
)
from .feature_engineering import engineer_symbol_features
from .research_dataset import build_research_dataset_from_modeling_state

ensure_numpy_nan_alias()
import pandas as pd  # noqa: E402

from ..rebalance import get_rebalance_dates  # noqa: E402

logger = logging.getLogger("cstree")

_REBALANCE_CANDIDATE_TAIL_DAYS = 5


@dataclass(frozen=True)
class _FeatureDatasetConfig:
    feature_params: dict
    price_col: str
    target: str
    label_shift_days: int
    label_horizon_days: int
    label_horizon_mode: str
    label_next_rebalance_map: dict[pd.Timestamp, pd.Timestamp] | None
    fundamentals_allow_missing: bool
    bucket_ic_enabled: bool
    bucket_ic_schemes: list[dict[str, Any]]
    feature_missing_features: list[str]
    feature_missing_method: str
    feature_missing_add_indicators: bool
    feature_missing_suffix: str
    fundamentals_cols: list[str]
    industry_cols: list[str]
    extra_passthrough_cols: list[str]
    execution_pricing_cols: set[str]
    backtest_tradable_col: str | None
    universe_by_date: pd.DataFrame | None
    winsorize_pct: float | None
    cs_method: str
    cs_winsorize_pct: float | None
    train_target: str
    train_target_transform: str
    train_target_group_cols: list[str] | None
    sample_on_rebalance_dates: bool
    rebalance_frequency: str
    min_symbols_per_date: int
    extra_sample_dates_without_target: list[object] | None


@dataclass(frozen=True)
class _FeatureDatasetPrepared:
    df: pd.DataFrame
    features: list[str]
    backtest_pricing_df: pd.DataFrame
    bucket_cols: list[str]
    passthrough_cols: list[str]
    price_passthrough_cols: list[str]
    eval_extra_df: pd.DataFrame | None
    feature_availability_diagnostics: dict[str, Any]
    missing_fill_features: list[str]
    raw_daily_panel_rows: int
    raw_reference_trade_dates: np.ndarray
    extra_no_label_dates: pd.DatetimeIndex


def _build_feature_dataset_config(values: Mapping[str, Any]) -> _FeatureDatasetConfig:
    return _FeatureDatasetConfig(
        feature_params=values["feature_params"],
        price_col=values["price_col"],
        target=values["target"],
        label_shift_days=values["label_shift_days"],
        label_horizon_days=values["label_horizon_days"],
        label_horizon_mode=values["label_horizon_mode"],
        label_next_rebalance_map=values["label_next_rebalance_map"],
        fundamentals_allow_missing=values["fundamentals_allow_missing"],
        bucket_ic_enabled=values["bucket_ic_enabled"],
        bucket_ic_schemes=values["bucket_ic_schemes"],
        feature_missing_features=values["feature_missing_features"],
        feature_missing_method=values["feature_missing_method"],
        feature_missing_add_indicators=values["feature_missing_add_indicators"],
        feature_missing_suffix=values["feature_missing_suffix"],
        fundamentals_cols=values["fundamentals_cols"],
        industry_cols=values["industry_cols"],
        extra_passthrough_cols=list(values.get("extra_passthrough_cols") or []),
        execution_pricing_cols=values["execution_pricing_cols"],
        backtest_tradable_col=values["backtest_tradable_col"],
        universe_by_date=values["universe_by_date"],
        winsorize_pct=values["winsorize_pct"],
        cs_method=values["cs_method"],
        cs_winsorize_pct=values["cs_winsorize_pct"],
        train_target=values["train_target"],
        train_target_transform=values["train_target_transform"],
        train_target_group_cols=values["train_target_group_cols"],
        sample_on_rebalance_dates=values["sample_on_rebalance_dates"],
        rebalance_frequency=values["rebalance_frequency"],
        min_symbols_per_date=values["min_symbols_per_date"],
        extra_sample_dates_without_target=values["extra_sample_dates_without_target"],
    )


def _build_feature_availability_diagnostics(
    df: pd.DataFrame,
    *,
    price_col: str,
    features: list[str],
    top_n: int = 5,
) -> dict[str, Any]:
    if df.empty or "trade_date" not in df.columns:
        return {
            "total_rows": len(df),
            "total_dates": 0,
            "complete_rows": 0,
            "complete_dates": 0,
            "complete_row_ratio": 0.0,
            "complete_date_ratio": 0.0,
            "worst_features": [],
        }

    total_rows = len(df)
    total_dates = int(df["trade_date"].nunique())
    required = [price_col] + [feature for feature in features if feature in df.columns]
    if not required:
        return {
            "total_rows": total_rows,
            "total_dates": total_dates,
            "complete_rows": total_rows,
            "complete_dates": total_dates,
            "complete_row_ratio": 1.0,
            "complete_date_ratio": 1.0,
            "worst_features": [],
        }

    complete_mask = cast(pd.Series, df[required].notna().all(axis=1))
    complete_rows = int(complete_mask.sum())
    complete_dates = int(df.loc[complete_mask, "trade_date"].nunique()) if complete_rows else 0

    feature_rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in df.columns:
            feature_rows.append(
                {
                    "feature": feature,
                    "missing_rows": total_rows,
                    "missing_pct": 100.0,
                    "dates_with_values": 0,
                }
            )
            continue
        missing_rows = int(df[feature].isna().sum())
        if missing_rows <= 0:
            continue
        dates_with_values = int(df.loc[df[feature].notna(), "trade_date"].nunique())
        feature_rows.append(
            {
                "feature": feature,
                "missing_rows": missing_rows,
                "missing_pct": round(missing_rows / total_rows * 100.0, 2) if total_rows else 0.0,
                "dates_with_values": dates_with_values,
            }
        )

    feature_rows.sort(
        key=lambda item: (
            item["missing_rows"],
            -item["dates_with_values"],
            item["feature"],
        ),
        reverse=True,
    )
    return {
        "total_rows": total_rows,
        "total_dates": total_dates,
        "complete_rows": complete_rows,
        "complete_dates": complete_dates,
        "complete_row_ratio": round(complete_rows / total_rows, 4) if total_rows else 0.0,
        "complete_date_ratio": round(complete_dates / total_dates, 4) if total_dates else 0.0,
        "worst_features": feature_rows[:top_n],
    }


def _format_feature_availability_rows(items: list[dict[str, Any]]) -> str:
    if not items:
        return "<none>"
    return ", ".join(
        (
            f"{item['feature']}"
            f"(missing={item['missing_pct']:.2f}%, dates={item['dates_with_values']})"
        )
        for item in items
    )


def _build_rebalance_tail_candidate_dates(
    trade_dates: pd.Series | list[pd.Timestamp],
    *,
    rebalance_frequency: str,
    tail_days_per_period: int = _REBALANCE_CANDIDATE_TAIL_DAYS,
) -> set[pd.Timestamp] | None:
    if not rebalance_frequency or str(rebalance_frequency).upper() == "D":
        return None
    dates = pd.to_datetime(pd.Series(trade_dates), errors="coerce").dropna().dt.normalize()
    if dates.empty:
        return set()

    tail_count = max(1, int(tail_days_per_period))
    unique_dates = np.asarray(sorted(dates.unique()), dtype="datetime64[ns]")
    rebalance_dates = np.asarray(
        pd.to_datetime(get_rebalance_dates(unique_dates, rebalance_frequency)),
        dtype="datetime64[ns]",
    )
    candidates: set[pd.Timestamp] = set()
    start_pos = 0
    for rebalance_date in rebalance_dates:
        end_pos = int(np.searchsorted(unique_dates, rebalance_date, side="right"))
        if end_pos <= start_pos:
            continue
        tail_start = max(start_pos, end_pos - tail_count)
        candidates.update(
            cast(pd.Timestamp, pd.Timestamp(date)).normalize()
            for date in unique_dates[tail_start:end_pos]
        )
        start_pos = end_pos
    return candidates


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
            _REBALANCE_CANDIDATE_TAIL_DAYS,
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
            _REBALANCE_CANDIDATE_TAIL_DAYS,
        )
    return modeling_date_candidates


def _resolve_engineered_features(
    df: pd.DataFrame,
    *,
    features: list[str],
    fundamentals_allow_missing: bool,
) -> list[str]:
    missing_features = [feat for feat in features if feat not in df.columns]
    if not missing_features:
        return features
    if fundamentals_allow_missing:
        logger.warning("Dropping missing features: %s", missing_features)
        return [feat for feat in features if feat in df.columns]
    sys.exit(f"Missing features after engineering: {missing_features}")


def _build_bucket_eval_frame(
    df: pd.DataFrame,
    *,
    bucket_ic_enabled: bool,
    bucket_ic_schemes: list[dict[str, Any]],
) -> tuple[pd.DataFrame | None, list[str]]:
    bucket_cols: list[str] = []
    if not (bucket_ic_enabled and bucket_ic_schemes):
        return None, bucket_cols

    bucket_cols = list(dict.fromkeys([scheme["column"] for scheme in bucket_ic_schemes]))
    missing_bucket_cols = [col for col in bucket_cols if col not in df.columns]
    if missing_bucket_cols:
        logger.warning("Bucket IC columns missing in data: %s", missing_bucket_cols)
    bucket_cols = [col for col in bucket_cols if col in df.columns]
    if not bucket_cols:
        return None, bucket_cols
    return cast(pd.DataFrame, df[["trade_date", "symbol", *bucket_cols]].copy()), bucket_cols


def _build_passthrough_cols(
    *,
    fundamentals_cols: list[str],
    industry_cols: list[str],
    extra_passthrough_cols: list[str],
) -> list[str]:
    pit_metadata_cols = [
        col
        for col in fundamentals_cols
        if col in {"report_period", "disclosure_date", "available_date"}
    ]
    return list(dict.fromkeys([*pit_metadata_cols, *industry_cols, *extra_passthrough_cols]))


def _log_feature_availability_warning(
    feature_availability_diagnostics: dict[str, Any],
) -> None:
    if feature_availability_diagnostics["total_dates"] >= 20 and (
        feature_availability_diagnostics["complete_dates"] < 20
        or feature_availability_diagnostics["complete_date_ratio"] < 0.25
    ):
        logger.warning(
            "Feature availability collapse before complete-case filter: "
            "complete_dates=%s/%s, complete_rows=%s/%s. "
            "Worst features after missing fill: %s. "
            "If this is a restored historical PIT config, verify the archived coverage "
            "report or trim the low-coverage feature block.",
            feature_availability_diagnostics["complete_dates"],
            feature_availability_diagnostics["total_dates"],
            feature_availability_diagnostics["complete_rows"],
            feature_availability_diagnostics["total_rows"],
            _format_feature_availability_rows(feature_availability_diagnostics["worst_features"]),
        )


def _log_modeling_dataset_summary(
    modeling_state: dict[str, Any],
    *,
    min_symbols_per_date: int,
    sample_on_rebalance_dates: bool,
    feature_availability_diagnostics: dict[str, Any],
) -> None:
    if not modeling_state["dropped_date_counts"].empty:
        logger.info(
            "Dropped %s dates with < %s symbols (min=%s, max=%s).",
            len(modeling_state["dropped_date_counts"]),
            min_symbols_per_date,
            int(modeling_state["dropped_date_counts"].min()),
            int(modeling_state["dropped_date_counts"].max()),
        )
    if len(modeling_state["all_dates_model_full"]) < 10:
        logger.warning(
            "Only %s model dates remain after feature filtering%s: %s. "
            "Worst features after missing fill: %s.",
            len(modeling_state["all_dates_model_full"]),
            " and sample-on-rebalance" if sample_on_rebalance_dates else "",
            [
                cast(pd.Timestamp, pd.Timestamp(date)).strftime("%Y-%m-%d")
                for date in modeling_state["all_dates_model_full"]
            ],
            _format_feature_availability_rows(feature_availability_diagnostics["worst_features"]),
        )


def _build_dataset_lifecycle_summary(
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
) -> dict[str, Any]:
    research_dataset = build_research_dataset_from_modeling_state(
        raw_panel=raw_panel,
        modeling_state=modeling_state,
        backtest_pricing_frame=backtest_pricing_df,
        features=features,
        target=target,
        train_target=train_target,
        missing_fill_features=missing_fill_features,
        feature_missing_method=feature_missing_method,
        feature_missing_add_indicators=feature_missing_add_indicators,
        winsorize_pct=winsorize_pct,
        cs_method=cs_method,
        cs_winsorize_pct=cs_winsorize_pct,
        train_target_transform=train_target_transform,
        train_target_group_cols=train_target_group_cols,
        universe_by_date_applied=universe_by_date_applied,
        sample_on_rebalance_dates=sample_on_rebalance_dates,
        min_symbols_per_date=min_symbols_per_date,
    )
    dataset_lifecycle = research_dataset.summary()
    metadata = dataset_lifecycle.setdefault("metadata", {})
    metadata["raw_daily_panel_rows"] = raw_daily_panel_rows
    metadata["engineered_feature_label_rows"] = len(raw_panel)
    del research_dataset
    return dataset_lifecycle


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
    dataset_lifecycle = _build_dataset_lifecycle_summary(
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
    )

    return {
        "features": prepared.features,
        "backtest_pricing_df": prepared.backtest_pricing_df,
        "bucket_cols": prepared.bucket_cols,
        "passthrough_cols": prepared.passthrough_cols,
        "price_passthrough_cols": prepared.price_passthrough_cols,
        "feature_availability_diagnostics": prepared.feature_availability_diagnostics,
        "dataset_lifecycle": dataset_lifecycle,
        **modeling_state,
    }
