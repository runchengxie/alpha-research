"""New-factor helpers sourced from locally-landed tushare datasets.

Each helper merges its auxiliary stock-day table into ``df`` (via ``_merge_aux``)
and assigns factor columns. Splitting the original monolithic ``_add_new_factors``
keeps every helper's McCabe complexity well under the complexity ceiling. Missing
source data leaves the factor columns all-NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._aux import _merge_aux


def _signed_log1p(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    return np.sign(values) * np.log1p(values.abs())


def _add_liquidity_flow_factor(df: pd.DataFrame, *, moneyflow: pd.DataFrame | None) -> pd.DataFrame:
    """Large-order net inflow (流动性资金流) from moneyflow_ths."""
    if moneyflow is not None and not moneyflow.empty:
        df = _merge_aux(df, moneyflow, ["net_amount", "buy_lg_amount_rate"])
        flow_raw = (
            df["buy_lg_amount_rate"].astype(float)
            if "buy_lg_amount_rate" in df
            else df["net_amount"].astype(float)
        )
        df["factor_liquidity_flow"] = flow_raw
    else:
        df["factor_liquidity_flow"] = np.nan
    return df


def _add_chip_concentration_factor(
    df: pd.DataFrame, *, holder: pd.DataFrame | None
) -> pd.DataFrame:
    """Chip concentration (top10 float concentration) from holder_structure."""
    if holder is not None and not holder.empty:
        df = _merge_aux(df, holder, ["top10_float_concentration", "top10_inst_float_hold_ratio"])
        df["factor_chip_concentration"] = (
            df["top10_float_concentration"].astype(float)
            if "top10_float_concentration" in df
            else np.nan
        )
        df["factor_institution_holding"] = (
            df["top10_inst_float_hold_ratio"].astype(float)
            if "top10_inst_float_hold_ratio" in df
            else np.nan
        )
    else:
        df["factor_chip_concentration"] = np.nan
        df["factor_institution_holding"] = np.nan
    return df


def _add_public_fund_ownership_factors(
    df: pd.DataFrame,
    *,
    fund_portfolio: pd.DataFrame | None,
) -> pd.DataFrame:
    """Public-fund ownership breadth, level and QoQ-change signals.

    The caller must provide a PIT materialized stock-date panel. The market-data
    asset already maps disclosures to an available date; this helper deliberately
    performs an exact-date merge so it cannot create a new look-ahead path.
    """
    factor_columns = (
        "factor_fund_breadth",
        "factor_fund_breadth_change",
        "factor_fund_ownership",
        "factor_fund_ownership_change",
    )
    if fund_portfolio is None or fund_portfolio.empty:
        for column in factor_columns:
            df[column] = np.nan
        return df

    source_columns = [
        "fund_count_holding_stock",
        "fund_count_holding_stock_qoq_change",
        "fund_hold_mv_to_float_mv",
        "fund_hold_mv_to_float_mv_qoq_change",
    ]
    df = _merge_aux(df, fund_portfolio, source_columns)

    count = pd.to_numeric(df["fund_count_holding_stock"], errors="coerce").clip(lower=0)
    count_change = pd.to_numeric(
        df["fund_count_holding_stock_qoq_change"], errors="coerce"
    )
    ownership = pd.to_numeric(df["fund_hold_mv_to_float_mv"], errors="coerce")
    ownership_change = pd.to_numeric(
        df["fund_hold_mv_to_float_mv_qoq_change"], errors="coerce"
    )

    # log1p reduces mega-cap / large-fund-count tail dominance while preserving
    # the ordering used by the subsequent cross-sectional standardization.
    df["factor_fund_breadth"] = np.log1p(count)
    df["factor_fund_breadth_change"] = _signed_log1p(count_change)
    df["factor_fund_ownership"] = ownership
    df["factor_fund_ownership_change"] = ownership_change
    return df


def _add_dividend_ps_value_factor(
    df: pd.DataFrame, *, basics_extra: pd.DataFrame | None
) -> pd.DataFrame:
    """Dividend yield & PS value (value group) from daily_basic extras."""
    if basics_extra is not None and not basics_extra.empty:
        df = _merge_aux(df, basics_extra, ["dv_ttm", "ps_ttm"])
        df["factor_dividend_yield"] = df["dv_ttm"].astype(float) if "dv_ttm" in df else np.nan
        df["factor_ps_value"] = (
            (1.0 / df["ps_ttm"].astype(float).where(df["ps_ttm"] > 0)) if "ps_ttm" in df else np.nan
        )
    else:
        df["factor_dividend_yield"] = np.nan
        df["factor_ps_value"] = np.nan
    return df


def add_new_factors(df: pd.DataFrame, *, aux: dict | None) -> pd.DataFrame:
    """Compute auxiliary daily and ownership factors from local datasets.

    Each sub-indicator is subsequently winsorized (1%/99%) and standardized by
    ``factor_calc._standardize_factors``. Missing source data leaves the factor
    column all-NaN so it is excluded from the active research set.
    """
    aux = aux or {}
    moneyflow = aux.get("moneyflow_ths")
    holder = aux.get("holder_structure")
    fund_portfolio = aux.get("fund_portfolio_features")
    basics_extra = aux.get("daily_basic_extra")

    df = _add_liquidity_flow_factor(df, moneyflow=moneyflow)
    df = _add_chip_concentration_factor(df, holder=holder)
    df = _add_public_fund_ownership_factors(df, fund_portfolio=fund_portfolio)
    return _add_dividend_ps_value_factor(df, basics_extra=basics_extra)
