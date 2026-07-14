from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from market_data_platform.symbols import canonicalize_symbol_columns

from .rebalance_calendar import get_rebalance_dates

logger = logging.getLogger("alpha_research")


@dataclass(frozen=True)
class _ModelingColumnPlan:
    target_group_cols: list[str]
    target_group_extra_cols: list[str]
    cols: list[str]


def _ensure_symbol_alias(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    if not any(
        col in frame.columns for col in ("symbol", "ts_code", "stock_ticker", "order_book_id")
    ):
        return frame
    return canonicalize_symbol_columns(frame, context="Alpha dataset")


def apply_universe_by_date(data: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return data
    rebalance_dates = np.array(sorted(universe["trade_date"].unique()))
    if rebalance_dates.size == 0:
        return data
    trade_dates = np.array(sorted(data["trade_date"].unique()))
    if trade_dates.size == 0:
        return data
    idx = np.searchsorted(rebalance_dates, trade_dates, side="right") - 1
    valid_mask = idx >= 0
    if not np.any(valid_mask):
        return data.iloc[0:0].copy()
    date_map = pd.DataFrame(
        {
            "trade_date": trade_dates[valid_mask],
            "rebalance_date": rebalance_dates[idx[valid_mask]],
        }
    )
    universe_map = universe.rename(columns={"trade_date": "rebalance_date"})
    data = data.merge(date_map, on="trade_date", how="inner")
    data = data.merge(
        universe_map[["rebalance_date", "symbol"]],
        on=["rebalance_date", "symbol"],
        how="inner",
    )
    return data.drop(columns=["rebalance_date"])


def prepare_backtest_pricing_frame(
    *,
    df: pd.DataFrame,
    price_col: str,
    execution_pricing_cols: set[str],
    backtest_tradable_col: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    execution_passthrough_cols = [
        col for col in execution_pricing_cols if col in df.columns and col != price_col
    ]
    backtest_pricing_cols = [
        "trade_date",
        "symbol",
        price_col,
        *execution_passthrough_cols,
    ]
    if backtest_tradable_col and backtest_tradable_col in df.columns:
        backtest_pricing_cols.append(backtest_tradable_col)
    backtest_pricing_cols.extend(
        col for col in ("is_buy_tradable", "is_sell_tradable") if col in df.columns
    )
    backtest_pricing_cols = list(dict.fromkeys(backtest_pricing_cols))
    pricing_cols_frame = cast(pd.DataFrame, df[backtest_pricing_cols])
    backtest_pricing_df = cast(
        pd.DataFrame,
        cast(Any, pricing_cols_frame).drop_duplicates(subset=["trade_date", "symbol"]).copy(),
    )
    price_passthrough_cols = list(
        dict.fromkeys(
            execution_passthrough_cols
            + [col for col in ("close", "tr_close") if col in df.columns and col != price_col]
        )
    )
    return backtest_pricing_df, price_passthrough_cols


def apply_feature_missing_fill(
    *,
    df: pd.DataFrame,
    features: list[str],
    feature_missing_features: list[str],
    feature_missing_method: str,
    feature_missing_add_indicators: bool,
    feature_missing_suffix: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df
    missing_fill_features = feature_missing_features or features
    missing_fill_features = [
        feature
        for feature in missing_fill_features
        if feature in features and feature in out.columns
    ]

    if feature_missing_add_indicators and missing_fill_features:
        indicator_features = []
        for feature in missing_fill_features:
            indicator_name = f"{feature}{feature_missing_suffix}"
            if indicator_name in out.columns and indicator_name not in features:
                sys.exit(
                    "features.missing.indicator_suffix collides with an existing column: "
                    f"{indicator_name}"
                )
            if indicator_name not in out.columns:
                out[indicator_name] = out[feature].isna().astype("int8")
            indicator_features.append(indicator_name)
        features = list(dict.fromkeys(features + indicator_features))

    if feature_missing_method != "none" and missing_fill_features:
        for feature in missing_fill_features:
            out[feature] = pd.to_numeric(out[feature], errors="coerce")
        if feature_missing_method == "zero":
            out[missing_fill_features] = out[missing_fill_features].fillna(0.0)
        elif feature_missing_method == "cross_sectional_median":
            by_date_median = out.groupby("trade_date")[missing_fill_features].transform("median")
            out[missing_fill_features] = out[missing_fill_features].fillna(by_date_median)
        remaining_missing = int(out[missing_fill_features].isna().sum().sum())
        logger.info(
            "Applied feature missing fill: method=%s, features=%s, add_indicators=%s, "
            "remaining_nans=%s.",
            feature_missing_method,
            len(missing_fill_features),
            feature_missing_add_indicators,
            remaining_missing,
        )

    return out, features, missing_fill_features


def _prefilter_to_rebalance_dates(
    *,
    out: pd.DataFrame,
    price_col: str,
    target: str,
    features: list[str],
    rebalance_frequency: str,
    reference_trade_dates: np.ndarray | None = None,
    require_target: bool = False,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    required_values = [price_col, *features, *([target] if require_target else [])]
    sample_source_cols = [col for col in list(dict.fromkeys(required_values)) if col in out.columns]
    sample_source = out.dropna(subset=sample_source_cols) if sample_source_cols else out
    sample_dates = np.sort(pd.to_datetime(sample_source["trade_date"].unique()).to_numpy())
    if reference_trade_dates is not None and len(reference_trade_dates):
        rebalance_calendar = np.sort(pd.to_datetime(reference_trade_dates).to_numpy())
    else:
        rebalance_calendar = sample_dates
    scheduled_rebalance_dates = get_rebalance_dates(rebalance_calendar, rebalance_frequency)
    rebalance_dates: list[pd.Timestamp] = []
    previous_scheduled_date: pd.Timestamp | None = None
    for scheduled_date in pd.to_datetime(scheduled_rebalance_dates):
        scheduled_value = np.datetime64(scheduled_date, "ns")
        start_pos = (
            int(
                np.searchsorted(
                    sample_dates,
                    np.datetime64(previous_scheduled_date, "ns"),
                    side="right",
                )
            )
            if previous_scheduled_date is not None
            else 0
        )
        end_pos = int(np.searchsorted(sample_dates, scheduled_value, side="right"))
        if end_pos > start_pos:
            rebalance_dates.append(
                cast(pd.Timestamp, pd.Timestamp(sample_dates[end_pos - 1])).normalize()
            )
        previous_scheduled_date = cast(pd.Timestamp, pd.Timestamp(scheduled_date))
    if not rebalance_dates:
        return cast(pd.DataFrame, out.iloc[0:0].copy()), []
    rebalance_date_index = pd.Index(pd.to_datetime(rebalance_dates))
    filtered = out[cast(pd.Series, out["trade_date"]).isin(list(rebalance_date_index))].copy()
    return cast(pd.DataFrame, filtered), rebalance_dates


def _normalize_extra_sample_dates(dates: list[object] | None) -> pd.DatetimeIndex:
    if dates is None:
        return pd.DatetimeIndex([])
    values = list(dates)
    if not values:
        return pd.DatetimeIndex([])
    normalized = pd.to_datetime(pd.Series(values), errors="coerce").dropna().dt.normalize()
    return pd.DatetimeIndex(normalized.unique())


def _append_extra_sample_dates_without_target(
    *,
    sampled: pd.DataFrame,
    source: pd.DataFrame,
    extra_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, int]:
    if extra_dates.empty:
        return sampled, 0
    extra_rows = source[cast(pd.Series, source["trade_date"]).isin(list(extra_dates))].copy()
    if extra_rows.empty:
        return sampled, 0
    if not sampled.empty:
        existing_index = pd.MultiIndex.from_frame(
            cast(pd.DataFrame, sampled[["trade_date", "symbol"]])
        )
        extra_index = pd.MultiIndex.from_frame(
            cast(pd.DataFrame, extra_rows[["trade_date", "symbol"]])
        )
        extra_rows = extra_rows.loc[~extra_index.isin(existing_index)].copy()
    if extra_rows.empty:
        return sampled, 0
    out = pd.concat([sampled, extra_rows], ignore_index=True)
    out = cast(
        pd.DataFrame,
        cast(Any, out).sort_values(["trade_date", "symbol"]).reset_index(drop=True),
    )
    return out, len(extra_rows)


def _build_modeling_column_plan(
    out: pd.DataFrame,
    *,
    price_col: str,
    target: str,
    train_target: str,
    train_target_group_cols: list[str] | None,
    features: list[str],
    price_passthrough_cols: list[str],
    passthrough_cols: list[str],
) -> _ModelingColumnPlan:
    target_group_cols = list(dict.fromkeys(train_target_group_cols or ["trade_date"]))
    if train_target != target:
        missing_group_cols = [col for col in target_group_cols if col not in out.columns]
        if missing_group_cols:
            missing_text = ", ".join(sorted(set(missing_group_cols)))
            sys.exit(f"label.train_target_group_cols columns not found: {missing_text}")
    target_group_extra_cols = [
        col
        for col in target_group_cols
        if col not in {"trade_date", "symbol", target} and col not in features
    ]
    cols = (
        ["trade_date", "symbol", price_col]
        + features
        + price_passthrough_cols
        + passthrough_cols
        + (target_group_extra_cols if train_target != target else [])
        + (["is_tradable"] if "is_tradable" in out.columns else [])
        + [target]
    )
    return _ModelingColumnPlan(
        target_group_cols=target_group_cols,
        target_group_extra_cols=target_group_extra_cols,
        cols=list(dict.fromkeys(cols)),
    )


def _resolve_reference_trade_dates(
    out: pd.DataFrame,
    reference_trade_dates: np.ndarray | None,
) -> np.ndarray:
    if reference_trade_dates is None:
        return np.sort(pd.to_datetime(out["trade_date"].unique()).to_numpy())
    return np.sort(pd.to_datetime(reference_trade_dates).to_numpy())


def _apply_modeling_universe_filter(
    out: pd.DataFrame,
    universe_by_date: pd.DataFrame | None,
) -> pd.DataFrame:
    if universe_by_date is None:
        return out
    before_rows = len(out)
    out = apply_universe_by_date(out, universe_by_date)
    after_rows = len(out)
    logger.info("Applied universe-by-date filter: %s -> %s rows", before_rows, after_rows)
    if out.empty:
        sys.exit("Universe-by-date filter removed all rows.")
    return out


def _sample_modeling_rows(
    out: pd.DataFrame,
    *,
    price_col: str,
    target: str,
    features: list[str],
    sample_on_rebalance_dates: bool,
    rebalance_frequency: str,
    reference_trade_dates: np.ndarray,
    extra_sample_dates_without_target: list[object] | None,
) -> pd.DataFrame:
    if not sample_on_rebalance_dates:
        return out

    before_rows = len(out)
    pre_sample_out = out
    out, early_rebalance_dates = _prefilter_to_rebalance_dates(
        out=out,
        price_col=price_col,
        target=target,
        features=features,
        rebalance_frequency=rebalance_frequency,
        reference_trade_dates=reference_trade_dates,
        require_target=True,
    )
    extra_dates = _normalize_extra_sample_dates(extra_sample_dates_without_target)
    out, extra_rows = _append_extra_sample_dates_without_target(
        sampled=out,
        source=pre_sample_out,
        extra_dates=extra_dates,
    )
    logger.info(
        "Applied sample-on-rebalance prefilter before modeling transforms: %s -> %s rows "
        "(dates=%s).",
        before_rows,
        len(out),
        len(early_rebalance_dates),
    )
    if extra_rows:
        logger.info(
            "Added %s no-label live scoring rows after rebalance sampling (dates=%s).",
            extra_rows,
            [date.strftime("%Y-%m-%d") for date in extra_dates],
        )
    if out.empty:
        sys.exit("sample_on_rebalance_dates removed all rows before modeling transforms.")
    return out
