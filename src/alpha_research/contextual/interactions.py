from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContextInteractionSpec:
    context_feature: str
    exposure_name: str
    output_name: str

    def __post_init__(self) -> None:
        for field_name in ("context_feature", "exposure_name", "output_name"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"ContextInteractionSpec.{field_name} must be non-empty")


def _utc(series: pd.Series, *, name: str) -> pd.Series:
    values = pd.to_datetime(series, utc=True, errors="coerce")
    if values.isna().any():
        raise ValueError(f"{name} contains invalid timestamps")
    return values


def _visibility_time(frame: pd.DataFrame) -> pd.Series:
    available = _utc(frame["available_at"], name="available_at")
    retrieved = _utc(frame["source_retrieved_at"], name="source_retrieved_at")
    return pd.concat([available, retrieved], axis=1).max(axis=1)


def attach_context_as_of(
    stock_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    trade_date_col: str = "trade_date",
    series_age_limits: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Attach the latest safely visible context state to each stock/date row.

    Visibility begins at the later of ``available_at`` and ``source_retrieved_at``.
    This prevents a late historical backfill from becoming visible merely because its
    source publication date predates the stock row.
    """

    if trade_date_col not in stock_frame:
        raise ValueError(f"stock_frame is missing {trade_date_col}")
    required_context = {"period_end", "available_at", "source_retrieved_at", *feature_names}
    missing = sorted(required_context.difference(context_frame.columns))
    if missing:
        raise ValueError(f"context_frame is missing columns: {', '.join(missing)}")
    if len(set(feature_names)) != len(tuple(feature_names)):
        raise ValueError("duplicate feature_names are not allowed")

    result = stock_frame.copy().reset_index(drop=True)
    result[trade_date_col] = _utc(result[trade_date_col], name=trade_date_col)
    result["__context_row_id"] = np.arange(len(result))
    limits = dict(series_age_limits or {})
    for feature_name in feature_names:
        if feature_name in limits and int(limits[feature_name]) < 0:
            raise ValueError(f"series age limit for {feature_name} must be non-negative")
        state = context_frame.loc[context_frame[feature_name].notna()].copy()
        if state.empty:
            result[feature_name] = np.nan
            result[f"{feature_name}__age_days"] = np.nan
            continue
        state["period_end"] = _utc(state["period_end"], name="period_end")
        state["available_at"] = _utc(state["available_at"], name="available_at")
        state["source_retrieved_at"] = _utc(
            state["source_retrieved_at"], name="source_retrieved_at"
        )
        state["__visibility_at"] = _visibility_time(state)
        state = state.sort_values(
            ["__visibility_at", "period_end", "available_at", "source_retrieved_at"],
            kind="stable",
        ).drop_duplicates("__visibility_at", keep="last")
        right = state.loc[
            :,
            ["__visibility_at", "period_end", "available_at", "source_retrieved_at", feature_name],
        ].rename(
            columns={
                "period_end": f"{feature_name}__period_end",
                "available_at": f"{feature_name}__available_at",
                "source_retrieved_at": f"{feature_name}__source_retrieved_at",
            }
        )
        left = result.sort_values(trade_date_col, kind="stable")
        merged = pd.merge_asof(
            left,
            right.sort_values("__visibility_at", kind="stable"),
            left_on=trade_date_col,
            right_on="__visibility_at",
            direction="backward",
            allow_exact_matches=True,
        )
        period_col = f"{feature_name}__period_end"
        age_col = f"{feature_name}__age_days"
        merged[age_col] = (merged[trade_date_col] - merged[period_col]).dt.total_seconds() / 86400.0
        if feature_name in limits:
            stale = merged[age_col] > float(limits[feature_name])
            merged.loc[stale, feature_name] = np.nan
        merged = merged.drop(columns=["__visibility_at"])
        result = merged.sort_values("__context_row_id", kind="stable").reset_index(drop=True)

    return result.drop(columns=["__context_row_id"])


def build_context_interactions(
    stock_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    exposure_frame: pd.DataFrame,
    specs: Sequence[ContextInteractionSpec],
    *,
    symbol_col: str = "symbol",
    trade_date_col: str = "trade_date",
    series_age_limits: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    names = [spec.output_name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate contextual interaction output_name values are not allowed")
    required_stock = {symbol_col, trade_date_col}
    missing_stock = sorted(required_stock.difference(stock_frame.columns))
    if missing_stock:
        raise ValueError(f"stock_frame is missing columns: {', '.join(missing_stock)}")
    required_exposure = {"trade_date", "symbol", "exposure_name", "exposure_value"}
    missing_exposure = sorted(required_exposure.difference(exposure_frame.columns))
    if missing_exposure:
        raise ValueError(f"exposure_frame is missing columns: {', '.join(missing_exposure)}")

    context_features = tuple(dict.fromkeys(spec.context_feature for spec in specs))
    result = attach_context_as_of(
        stock_frame,
        context_frame,
        feature_names=context_features,
        trade_date_col=trade_date_col,
        series_age_limits=series_age_limits,
    )
    result[trade_date_col] = _utc(result[trade_date_col], name=trade_date_col)
    result[symbol_col] = result[symbol_col].astype(str)

    exposures = exposure_frame.copy()
    exposures["trade_date"] = _utc(exposures["trade_date"], name="exposure trade_date")
    exposures["symbol"] = exposures["symbol"].astype(str)
    duplicate_keys = exposures.duplicated(["trade_date", "symbol", "exposure_name"], keep=False)
    if duplicate_keys.any():
        raise ValueError("exposure_frame contains duplicate date/symbol/exposure_name rows")

    for spec in specs:
        selected_columns = ["trade_date", "symbol", "exposure_value"]
        if "exposure_version" in exposures:
            selected_columns.append("exposure_version")
        selected = exposures.loc[
            exposures["exposure_name"].astype(str).eq(spec.exposure_name), selected_columns
        ].copy()
        value_col = f"__exposure_value__{spec.output_name}"
        version_col = f"{spec.output_name}__exposure_version"
        rename = {"exposure_value": value_col}
        if "exposure_version" in selected:
            rename["exposure_version"] = version_col
        selected = selected.rename(columns=rename)
        result = result.merge(
            selected,
            how="left",
            left_on=[trade_date_col, symbol_col],
            right_on=["trade_date", "symbol"],
            suffixes=("", "__exposure"),
            validate="many_to_one",
        )
        for duplicate_key in ("trade_date__exposure", "symbol__exposure"):
            if duplicate_key in result:
                result = result.drop(columns=duplicate_key)
        result[spec.output_name] = pd.to_numeric(
            result[spec.context_feature], errors="coerce"
        ) * pd.to_numeric(result[value_col], errors="coerce")
        result = result.drop(columns=[value_col])
    return result
