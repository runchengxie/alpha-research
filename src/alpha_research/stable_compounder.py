"""PIT-safe quarterly operating panels and stable-compounder labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _standalone_flow(values: pd.Series, periods: pd.Series) -> pd.Series:
    """Convert a cumulative YTD flow into a standalone quarterly flow."""

    result = values.copy()
    years = periods.dt.year
    quarters = periods.dt.quarter
    for year in years.dropna().unique():
        year_mask = years.eq(year)
        previous = values.where(year_mask).shift(1)
        previous_quarter = quarters.where(year_mask).shift(1)
        sequential = previous_quarter.eq(quarters - 1)
        result = result.where(~(year_mask & sequential), values - previous)
        q4 = year_mask & quarters.eq(4)
        q3_cumulative = values.where(year_mask & quarters.eq(3)).shift(1)
        result = result.where(~q4, values - q3_cumulative)
    return result.replace([np.inf, -np.inf], np.nan)


def build_quarterly_operating_panel(
    pit_panel: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    report_period_col: str = "report_period",
    available_date_col: str = "available_date",
    trade_date_col: str = "trade_date",
    flow_cols: tuple[str, ...] = ("revenue", "n_income_attr_p", "n_cashflow_act"),
) -> pd.DataFrame:
    """Merge PIT source rows and convert cumulative flows to standalone quarters.

    The input must already be selected as-of a formation date or otherwise
    contain one PIT observation per symbol/report period.  The function does
    not choose revisions and never uses a future ``available_date``.
    """

    required = {symbol_col, report_period_col, available_date_col, trade_date_col, *flow_cols}
    missing = sorted(required - set(pit_panel.columns))
    if missing:
        raise ValueError(f"quarterly operating panel missing columns: {missing}")
    out = pit_panel.copy()
    for column in (report_period_col, available_date_col, trade_date_col):
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    if out[[report_period_col, available_date_col, trade_date_col]].isna().any().any():
        raise ValueError("quarterly operating panel requires valid PIT dates")
    for column in flow_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    keys = [symbol_col, report_period_col, available_date_col, trade_date_col]
    aggregations = {column: "first" for column in out.columns if column not in keys}
    out = out.groupby(keys, as_index=False, sort=True).agg(aggregations)
    out = out.sort_values([symbol_col, report_period_col, available_date_col]).reset_index(drop=True)
    out["_report_year"] = out[report_period_col].dt.year
    out["_report_quarter"] = out[report_period_col].dt.quarter
    flow_groups = [out[symbol_col], out["_report_year"]]
    for column in flow_cols:
        previous = out.groupby(flow_groups, sort=False)[column].shift(1)
        previous_quarter = out.groupby(flow_groups, sort=False)["_report_quarter"].shift(1)
        sequential = previous_quarter.eq(out["_report_quarter"] - 1)
        standalone = out[column].where(~sequential, out[column] - previous)
        q3_cumulative = out[column].where(out["_report_quarter"].eq(3)).groupby(
            flow_groups, sort=False
        ).shift(1)
        standalone = standalone.where(~out["_report_quarter"].eq(4), out[column] - q3_cumulative)
        out[f"standalone_{column}"] = standalone.replace([np.inf, -np.inf], np.nan)
    out["quarter_index"] = out[report_period_col].dt.year * 4 + out[report_period_col].dt.quarter
    out["standalone_cfo_margin"] = (
        out["standalone_n_cashflow_act"] / out["standalone_revenue"].where(out["standalone_revenue"].ne(0))
    )
    out["standalone_cfo_to_profit"] = (
        out["standalone_n_cashflow_act"]
        / out["standalone_n_income_attr_p"].where(out["standalone_n_income_attr_p"].ne(0))
    )
    out = out.drop(columns=["_report_year", "_report_quarter"])
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def select_latest_pit_report_events(
    pit_panel: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp,
    symbol_col: str = "symbol",
    report_period_col: str = "report_period",
    available_date_col: str = "available_date",
) -> pd.DataFrame:
    """Select the latest available event per report period at a formation date."""

    required = {symbol_col, report_period_col, available_date_col}
    missing = sorted(required - set(pit_panel.columns))
    if missing:
        raise ValueError(f"PIT event panel missing columns: {missing}")
    as_of = pd.Timestamp(as_of_date).normalize()
    out = pit_panel.copy()
    out[available_date_col] = pd.to_datetime(out[available_date_col], errors="coerce").dt.normalize()
    out[report_period_col] = pd.to_datetime(out[report_period_col], errors="coerce").dt.normalize()
    out = out.loc[out[available_date_col].le(as_of)].dropna(
        subset=[symbol_col, report_period_col, available_date_col]
    )
    if out.empty:
        return out
    return (
        out.sort_values([symbol_col, report_period_col, available_date_col])
        .drop_duplicates([symbol_col, report_period_col], keep="last")
        .reset_index(drop=True)
    )


def build_rolling_stability_labels(
    quarterly_panel: pd.DataFrame,
    *,
    window_quarters: int = 12,
    minimum_observed: int = 8,
    minimum_positive_quarters: int = 10,
    minimum_cfo_to_profit: float = 0.8,
) -> pd.DataFrame:
    """Add auditable current and strict rolling stable-compounder labels."""

    required = {
        "symbol",
        "report_period",
        "quarter_index",
        "standalone_n_income_attr_p",
        "standalone_n_cashflow_act",
        "standalone_cfo_margin",
        "standalone_cfo_to_profit",
    }
    missing = sorted(required - set(quarterly_panel.columns))
    if missing:
        raise ValueError(f"stability label panel missing columns: {missing}")
    if not 2 <= minimum_observed <= window_quarters:
        raise ValueError("minimum_observed must be between 2 and window_quarters")
    out = quarterly_panel.sort_values(["symbol", "report_period"]).reset_index(drop=True).copy()
    grouped = out.groupby("symbol", sort=False)["quarter_index"]
    observed = grouped.rolling(window_quarters, min_periods=minimum_observed).count()
    observed.index = observed.index.droplevel(0)
    minimum = grouped.rolling(window_quarters, min_periods=window_quarters).min()
    minimum.index = minimum.index.droplevel(0)
    maximum = grouped.rolling(window_quarters, min_periods=window_quarters).max()
    maximum.index = maximum.index.droplevel(0)
    out["quarters_observed"] = observed.reindex(out.index)
    out["quarters_contiguous"] = (
        minimum.reindex(out.index).sub(maximum.reindex(out.index)).abs().eq(window_quarters - 1)
    )

    def rolling_stat(column: str, statistic: str, min_periods: int) -> pd.Series:
        values = out.groupby("symbol", sort=False)[column].rolling(
            window_quarters, min_periods=min_periods
        )
        result = getattr(values, statistic)()
        result.index = result.index.droplevel(0)
        return result.reindex(out.index)

    out["cfo_to_profit_median"] = rolling_stat(
        "standalone_cfo_to_profit", "median", minimum_observed
    )
    out["cfo_margin_std"] = rolling_stat("standalone_cfo_margin", "std", minimum_observed)
    # Convert positive/negative observations to ratios after the vectorized window.
    for source, output in (
        ("standalone_n_income_attr_p", "positive_profit_ratio"),
        ("standalone_n_cashflow_act", "positive_cfo_ratio"),
    ):
        positive = out.assign(_positive=out[source].gt(0)).groupby("symbol", sort=False)["_positive"].rolling(
            window_quarters, min_periods=minimum_observed
        ).mean()
        positive.index = positive.index.droplevel(0)
        out[output] = positive.reindex(out.index)
    out["stable_compounder_eligible"] = (
        out["quarters_observed"].ge(minimum_observed) & out["quarters_contiguous"]
    )
    out["stable_compounder_strict"] = (
        out["stable_compounder_eligible"]
        & out["quarters_observed"].ge(window_quarters)
        & out["positive_profit_ratio"].ge(minimum_positive_quarters / window_quarters)
        & out["positive_cfo_ratio"].ge(minimum_positive_quarters / window_quarters)
        & out["cfo_to_profit_median"].ge(minimum_cfo_to_profit)
    )
    return out
