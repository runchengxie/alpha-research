"""New-factor helpers sourced from locally-landed tushare datasets.

Each helper merges its auxiliary stock-day table into ``df`` (via ``_merge_aux``)
and assigns exactly one factor column.  Splitting the original monolithic
``_add_new_factors`` keeps every helper's McCabe complexity well under the
complexity ceiling.  Missing source data leaves the factor column all-NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._aux import _merge_aux


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

    Each sub-indicator is winsorized (1%/99%) cross-sectionally then z-scored,
    in the same spirit as ``_standardize_factors`` / ``_add_quality_factor``.
    Missing source data leaves the factor column all-NaN (it is then dropped).
    """
    aux = aux or {}
    moneyflow = aux.get("moneyflow_ths")
    holder = aux.get("holder_structure")
    basics_extra = aux.get("daily_basic_extra")

    df = _add_liquidity_flow_factor(df, moneyflow=moneyflow)
    df = _add_chip_concentration_factor(df, holder=holder)
    return _add_dividend_ps_value_factor(df, basics_extra=basics_extra)
