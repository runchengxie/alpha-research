"""Periodic point-in-time fundamental target construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from alpha_research.fundamental_state import (
        FundamentalTargetPanel,
        FundamentalTargetSpec,
        TargetTransform,
    )

FUNDAMENTAL_STATE_SCHEMA = "fundamental_state_forecasting.v1"


def _normalized_dates(series: pd.Series, *, column: str) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"fundamental state requires valid dates in {column}")
    if values.dt.tz is not None:
        values = values.dt.tz_localize(None)
    return values.dt.normalize()


def _nullable_normalized_dates(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    if values.dt.tz is not None:
        values = values.dt.tz_localize(None)
    return values.dt.normalize()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _validate_target_specs(specs: tuple[FundamentalTargetSpec, ...]) -> None:
    if not specs:
        raise ValueError("fundamental target specs must be non-empty")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("fundamental target names must be unique")


def _target_values(
    current: pd.Series,
    future: pd.Series,
    *,
    transform: TargetTransform,
) -> pd.Series:
    if transform == "level":
        return future
    if transform == "delta":
        return future - current
    valid_base = current.where(current.notna() & np.isfinite(current) & (current > 0))
    return ((future / valid_base) - 1.0).replace([np.inf, -np.inf], np.nan)


def build_periodic_fundamental_target_panel(
    frame: pd.DataFrame,
    target_specs: tuple[FundamentalTargetSpec, ...],
    *,
    horizon_periods: int,
    period_months: int = 3,
    symbol_col: str = "symbol",
    report_period_col: str = "report_period",
    available_date_col: str = "available_date",
) -> FundamentalTargetPanel:
    """Attach an exact quarterly (or other fixed-month) future state target.

    This is the periodic counterpart to the annual helper. The input remains
    one PIT-audited row per symbol/report period, and the future observation's
    availability date is retained as the label end date. It deliberately
    matches exact period boundaries instead of selecting the latest available
    observation, which would silently change the target definition.
    """

    from alpha_research.fundamental_state import FundamentalTargetPanel

    specs = tuple(target_specs)
    _validate_target_specs(specs)
    if int(horizon_periods) <= 0 or int(period_months) <= 0:
        raise ValueError("horizon_periods and period_months must be positive")
    required = {symbol_col, report_period_col, available_date_col}
    required.update(spec.source_col for spec in specs)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fundamental state frame missing columns: {missing}")

    out = frame.copy()
    out[symbol_col] = out[symbol_col].astype("string")
    if out[symbol_col].isna().any() or out[symbol_col].str.strip().eq("").any():
        raise ValueError("fundamental state requires non-empty symbols")
    out[symbol_col] = out[symbol_col].astype(str)
    out[report_period_col] = _normalized_dates(out[report_period_col], column=report_period_col)
    out[available_date_col] = _normalized_dates(out[available_date_col], column=available_date_col)
    if out.duplicated([symbol_col, report_period_col]).any():
        raise ValueError("fundamental state input contains duplicate symbol/report_period rows")
    if (out[available_date_col] <= out[report_period_col]).any():
        raise ValueError("observations must become available after their report period")

    out["feature_as_of_date"] = out[available_date_col]
    offset = pd.DateOffset(months=int(horizon_periods) * int(period_months))
    out["target_report_period"] = out[report_period_col] + offset
    future_columns = [symbol_col, report_period_col, available_date_col]
    future_columns.extend(sorted({spec.source_col for spec in specs}))
    future = out[future_columns].copy()
    rename = {
        report_period_col: "target_report_period",
        available_date_col: "target_available_date",
    }
    rename.update({spec.source_col: f"__future__{spec.source_col}" for spec in specs})
    future.rename(columns=rename, inplace=True)
    merged = out.merge(
        future,
        how="left",
        on=[symbol_col, "target_report_period"],
        validate="many_to_one",
        sort=False,
    )
    merged["target_available_date"] = _nullable_normalized_dates(merged["target_available_date"])
    invalid = merged["target_available_date"].notna() & (
        merged["target_available_date"] <= merged["feature_as_of_date"]
    )
    if invalid.any():
        raise ValueError("future fundamental labels must become available after feature_as_of_date")
    merged["fundamental_label_end_date"] = merged["target_available_date"]
    for spec in specs:
        current = _numeric(merged[spec.source_col])
        future_values = _numeric(merged[f"__future__{spec.source_col}"])
        merged[spec.name] = _target_values(current, future_values, transform=spec.transform)
    merged.drop(
        columns=[f"__future__{spec.source_col}" for spec in specs],
        inplace=True,
    )
    complete = merged["target_available_date"].notna()
    complete &= merged[[spec.name for spec in specs]].notna().all(axis=1)
    audit = {
        "schema_version": FUNDAMENTAL_STATE_SCHEMA,
        "input_contract": "one canonical PIT-audited row per symbol/report_period",
        "horizon_periods": int(horizon_periods),
        "period_months": int(period_months),
        "rows": len(merged),
        "complete_label_rows": int(complete.sum()),
        "target_names": [spec.name for spec in specs],
        "label_end_semantics": "target_available_date",
    }
    return FundamentalTargetPanel(merged, audit)


__all__ = ["build_periodic_fundamental_target_panel"]
