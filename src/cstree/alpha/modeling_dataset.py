from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

from .dataset import DatasetSchema, build_dataset
from .dataset_sampling_core import (
    _apply_modeling_universe_filter,
    _build_modeling_column_plan,
    _ensure_symbol_alias,
    _resolve_reference_trade_dates,
    _sample_modeling_rows,
)
from .date_slices import _build_trade_date_slices, _slice_trade_dates
from .transform import apply_cross_sectional_series_transform, apply_cross_sectional_transform

logger = logging.getLogger("cstree")


def _apply_target_winsorization(
    df_features: pd.DataFrame,
    *,
    target: str,
    winsorize_pct: float | None,
) -> pd.DataFrame:
    if not winsorize_pct:
        return df_features

    def _winsorize(group: pd.DataFrame) -> pd.DataFrame:
        lower = group[target].quantile(winsorize_pct)
        upper = group[target].quantile(1 - winsorize_pct)
        if bool(cast(Any, pd.isna(lower))) or bool(cast(Any, pd.isna(upper))):
            return group
        group[target] = group[target].clip(lower, upper)
        return group

    return cast(pd.DataFrame, df_features.groupby("trade_date", group_keys=False).apply(_winsorize))


def _prepare_modeling_feature_frame(
    out: pd.DataFrame,
    *,
    price_col: str,
    target: str,
    train_target: str,
    train_target_transform: str,
    target_group_cols: list[str],
    features: list[str],
    winsorize_pct: float | None,
    cs_method: str,
    cs_winsorize_pct: float | None,
) -> pd.DataFrame:
    required_cols = [price_col, *features]
    df_features = out.dropna(subset=required_cols).reset_index(drop=True)
    df_features = _apply_target_winsorization(
        df_features,
        target=target,
        winsorize_pct=winsorize_pct,
    )
    if cs_method != "none":
        df_features = apply_cross_sectional_transform(
            df_features, features, cs_method, cs_winsorize_pct
        )
    if train_target != target:
        df_features[train_target] = apply_cross_sectional_series_transform(
            df_features,
            target,
            train_target_transform,
            group_cols=target_group_cols,
        )
        logger.info(
            "Applied training target transform: base=%s, method=%s, train_target=%s, group_cols=%s",
            target,
            train_target_transform,
            train_target,
            target_group_cols,
        )
    return df_features


def _build_modeling_dataset_frame(
    df_features: pd.DataFrame,
    *,
    price_col: str,
    target: str,
    train_target: str,
    features: list[str],
    price_passthrough_cols: list[str],
    passthrough_cols: list[str],
    target_group_extra_cols: list[str],
    eval_extra_df: pd.DataFrame | None,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    dataset_schema = DatasetSchema(
        date_col="trade_date",
        instrument_col="symbol",
        price_col=price_col,
        label_col=target,
        tradable_col="is_tradable" if "is_tradable" in df_features.columns else None,
        feature_cols=features,
        extra_cols=[
            *price_passthrough_cols,
            *passthrough_cols,
            *target_group_extra_cols,
            *([train_target] if train_target != target else []),
        ],
    )
    dataset = build_dataset(df_features, dataset_schema)
    df_features = dataset.frame
    complete_case_cols = [price_col, target, *features]
    if train_target != target:
        complete_case_cols.append(train_target)
    complete_case_cols = [
        column
        for column in list(dict.fromkeys(complete_case_cols))
        if column in df_features.columns
    ]
    df_full = df_features.dropna(subset=complete_case_cols).reset_index(drop=True)
    if eval_extra_df is not None and not eval_extra_df.empty:
        eval_extra_df = eval_extra_df.drop_duplicates(subset=("trade_date", "symbol"))
        extra_eval_cols = [
            col
            for col in eval_extra_df.columns
            if col not in {"trade_date", "symbol"} and col not in df_full.columns
        ]
        if extra_eval_cols:
            df_full = cast(
                pd.DataFrame,
                df_full.merge(
                    eval_extra_df[["trade_date", "symbol", *extra_eval_cols]],
                    on=["trade_date", "symbol"],
                    how="left",
                ),
            )
    return dataset, df_features, df_full


def _filter_model_dates_by_symbol_count(
    df_model_all: pd.DataFrame,
    *,
    min_symbols_per_date: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[pd.Timestamp, int],
    pd.Index,
    pd.Series,
]:
    date_counts = cast(pd.Series, df_model_all.groupby("trade_date")["symbol"].nunique())
    valid_date_mask = cast(pd.Series, date_counts >= min_symbols_per_date)
    valid_dates = cast(pd.Index, date_counts.loc[valid_date_mask].index)
    dropped_date_counts = cast(
        pd.Series,
        date_counts.loc[~valid_date_mask].sort_index(),
    )
    (
        df_model_all_sorted,
        all_dates_model_full,
        model_date_start_rows,
        model_date_end_rows,
        model_date_to_pos,
    ) = _build_trade_date_slices(df_model_all)
    if len(valid_dates) != len(date_counts):
        df_model_all = _slice_trade_dates(
            df_model_all_sorted,
            model_date_start_rows,
            model_date_end_rows,
            model_date_to_pos,
            valid_dates.to_numpy(),
        )
        (
            df_model_all_sorted,
            all_dates_model_full,
            model_date_start_rows,
            model_date_end_rows,
            model_date_to_pos,
        ) = _build_trade_date_slices(df_model_all)
    else:
        df_model_all = df_model_all_sorted
    return (
        df_model_all,
        df_model_all_sorted,
        all_dates_model_full,
        model_date_start_rows,
        model_date_end_rows,
        model_date_to_pos,
        valid_dates,
        dropped_date_counts,
    )


def _assemble_modeling_dataset_state(
    *,
    dataset: Any,
    df_features: pd.DataFrame,
    df_full: pd.DataFrame,
    df_full_sorted: pd.DataFrame,
    reference_trade_dates: np.ndarray,
    all_dates_full: np.ndarray,
    full_date_start_rows: np.ndarray,
    full_date_end_rows: np.ndarray,
    full_date_to_pos: dict[pd.Timestamp, int],
    df_model_all: pd.DataFrame,
    df_model_all_sorted: pd.DataFrame,
    all_dates_model_full: np.ndarray,
    model_date_start_rows: np.ndarray,
    model_date_end_rows: np.ndarray,
    model_date_to_pos: dict[pd.Timestamp, int],
    valid_dates: pd.Index,
    dropped_date_counts: pd.Series,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "df_features": df_features,
        "df_full": df_full,
        "df_full_sorted": df_full_sorted,
        "reference_trade_dates": reference_trade_dates,
        "all_dates_full": all_dates_full,
        "full_date_start_rows": full_date_start_rows,
        "full_date_end_rows": full_date_end_rows,
        "full_date_to_pos": full_date_to_pos,
        "df_model_all": df_model_all,
        "df_model_all_sorted": df_model_all_sorted,
        "all_dates_model_full": all_dates_model_full,
        "model_date_start_rows": model_date_start_rows,
        "model_date_end_rows": model_date_end_rows,
        "model_date_to_pos": model_date_to_pos,
        "valid_dates": valid_dates,
        "valid_dates_set": set(pd.to_datetime(valid_dates)),
        "dropped_date_counts": dropped_date_counts,
    }


def _build_modeling_output_state(
    *,
    dataset: Any,
    df_features: pd.DataFrame,
    df_full: pd.DataFrame,
    reference_trade_dates: np.ndarray,
    min_symbols_per_date: int,
) -> dict[str, Any]:
    (
        df_full_sorted,
        all_dates_full,
        full_date_start_rows,
        full_date_end_rows,
        full_date_to_pos,
    ) = _build_trade_date_slices(df_full)
    # When sample_on_rebalance_dates is enabled, the frame has already been sampled
    # before modeling transforms. Sampling it again is harmless for simple
    # period-end rules but incorrectly halves custom every-other-week schedules.
    df_model_all = df_full_sorted
    (
        df_model_all,
        df_model_all_sorted,
        all_dates_model_full,
        model_date_start_rows,
        model_date_end_rows,
        model_date_to_pos,
        valid_dates,
        dropped_date_counts,
    ) = _filter_model_dates_by_symbol_count(
        df_model_all,
        min_symbols_per_date=min_symbols_per_date,
    )
    return _assemble_modeling_dataset_state(
        dataset=dataset,
        df_features=df_features,
        df_full=df_full,
        df_full_sorted=df_full_sorted,
        reference_trade_dates=reference_trade_dates,
        all_dates_full=all_dates_full,
        full_date_start_rows=full_date_start_rows,
        full_date_end_rows=full_date_end_rows,
        full_date_to_pos=full_date_to_pos,
        df_model_all=df_model_all,
        df_model_all_sorted=df_model_all_sorted,
        all_dates_model_full=all_dates_model_full,
        model_date_start_rows=model_date_start_rows,
        model_date_end_rows=model_date_end_rows,
        model_date_to_pos=model_date_to_pos,
        valid_dates=valid_dates,
        dropped_date_counts=dropped_date_counts,
    )


def build_modeling_dataset(
    *,
    df: pd.DataFrame,
    price_col: str,
    target: str,
    train_target: str,
    train_target_transform: str,
    train_target_group_cols: list[str] | None = None,
    features: list[str],
    price_passthrough_cols: list[str],
    passthrough_cols: list[str],
    winsorize_pct: float | None,
    cs_method: str,
    cs_winsorize_pct: float | None,
    sample_on_rebalance_dates: bool,
    rebalance_frequency: str,
    min_symbols_per_date: int,
    universe_by_date: pd.DataFrame | None,
    eval_extra_df: pd.DataFrame | None,
    reference_trade_dates: np.ndarray | None = None,
    extra_sample_dates_without_target: list[object] | None = None,
) -> dict[str, Any]:
    out = _ensure_symbol_alias(df)
    column_plan = _build_modeling_column_plan(
        out,
        price_col=price_col,
        target=target,
        train_target=train_target,
        train_target_group_cols=train_target_group_cols,
        features=features,
        price_passthrough_cols=price_passthrough_cols,
        passthrough_cols=passthrough_cols,
    )
    out = cast(pd.DataFrame, out[column_plan.cols].copy())
    reference_trade_dates = _resolve_reference_trade_dates(out, reference_trade_dates)
    out = _apply_modeling_universe_filter(out, universe_by_date)
    out = _sample_modeling_rows(
        out,
        price_col=price_col,
        target=target,
        features=features,
        sample_on_rebalance_dates=sample_on_rebalance_dates,
        rebalance_frequency=rebalance_frequency,
        reference_trade_dates=reference_trade_dates,
        extra_sample_dates_without_target=extra_sample_dates_without_target,
    )
    df_features = _prepare_modeling_feature_frame(
        out,
        price_col=price_col,
        target=target,
        train_target=train_target,
        train_target_transform=train_target_transform,
        target_group_cols=column_plan.target_group_cols,
        features=features,
        winsorize_pct=winsorize_pct,
        cs_method=cs_method,
        cs_winsorize_pct=cs_winsorize_pct,
    )
    dataset, df_features, df_full = _build_modeling_dataset_frame(
        df_features,
        price_col=price_col,
        target=target,
        train_target=train_target,
        features=features,
        price_passthrough_cols=price_passthrough_cols,
        passthrough_cols=passthrough_cols,
        target_group_extra_cols=column_plan.target_group_extra_cols,
        eval_extra_df=eval_extra_df,
    )
    return _build_modeling_output_state(
        dataset=dataset,
        df_features=df_features,
        df_full=df_full,
        reference_trade_dates=reference_trade_dates,
        min_symbols_per_date=min_symbols_per_date,
    )
