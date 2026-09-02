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
    for column in flow_cols:
        standalone = pd.Series(np.nan, index=out.index, dtype=float)
        for _, indices in out.groupby(symbol_col, sort=False).groups.items():
            group = out.loc[indices]
            standalone.loc[indices] = _standalone_flow(
                group[column], group[report_period_col]
            ).to_numpy()
        out[f"standalone_{column}"] = standalone
    out["quarter_index"] = out[report_period_col].dt.year * 4 + out[report_period_col].dt.quarter
    out["standalone_cfo_margin"] = (
        out["standalone_n_cashflow_act"] / out["standalone_revenue"].where(out["standalone_revenue"].ne(0))
    )
    out["standalone_cfo_to_profit"] = (
        out["standalone_n_cashflow_act"]
        / out["standalone_n_income_attr_p"].where(out["standalone_n_income_attr_p"].ne(0))
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


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
    out_parts: list[pd.DataFrame] = []
    for _, group in quarterly_panel.groupby("symbol", sort=False):
        out_group = group.sort_values("report_period").copy()
        rolling = out_group.rolling(window_quarters, min_periods=minimum_observed)
        out_group["quarters_observed"] = rolling["quarter_index"].count()
        out_group["quarters_contiguous"] = rolling["quarter_index"].apply(
            lambda values: float(values.max() - values.min() == window_quarters - 1)
            if values.notna().sum() >= window_quarters
            else 0.0,
            raw=False,
        ).astype(bool)
        out_group["positive_profit_ratio"] = rolling["standalone_n_income_attr_p"].apply(
            lambda values: values.gt(0).mean(), raw=False
        )
        out_group["positive_cfo_ratio"] = rolling["standalone_n_cashflow_act"].apply(
            lambda values: values.gt(0).mean(), raw=False
        )
        out_group["cfo_to_profit_median"] = rolling["standalone_cfo_to_profit"].median()
        out_group["cfo_margin_std"] = rolling["standalone_cfo_margin"].std(ddof=0)
        out_group["stable_compounder_eligible"] = (
            out_group["quarters_observed"].ge(minimum_observed)
            & out_group["quarters_contiguous"]
        )
        out_group["stable_compounder_strict"] = (
            out_group["stable_compounder_eligible"]
            & out_group["quarters_observed"].ge(window_quarters)
            & out_group["positive_profit_ratio"].ge(minimum_positive_quarters / window_quarters)
            & out_group["positive_cfo_ratio"].ge(minimum_positive_quarters / window_quarters)
            & out_group["cfo_to_profit_median"].ge(minimum_cfo_to_profit)
        )
        out_parts.append(out_group)
    if not out_parts:
        return quarterly_panel.copy()
    return pd.concat(out_parts, ignore_index=True).sort_values(["symbol", "report_period"])
