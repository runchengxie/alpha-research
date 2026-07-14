from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def _to_datetime_series(series: pd.Series) -> pd.Series:
    return cast(pd.Series, pd.to_datetime(series, errors="coerce"))


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(series, errors="coerce"))


def _compute_group_trailing_stat(
    dates: pd.Series,
    values: pd.Series,
    *,
    years: int,
    stat: str,
    min_periods: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=dates.index, dtype=float)
    if dates.empty:
        return result

    for row_index, current_date in dates.items():
        if pd.isna(current_date):
            continue
        window_start = current_date - pd.DateOffset(years=years)
        mask = (dates >= window_start) & (dates <= current_date)
        window = values.loc[mask].dropna()
        if len(window) < min_periods:
            continue
        if stat == "mean":
            result.loc[row_index] = float(window.mean())
        elif stat == "median":
            result.loc[row_index] = float(window.median())
        elif stat == "std":
            result.loc[row_index] = float(window.std(ddof=0))
        elif stat == "positive_ratio":
            result.loc[row_index] = float((window > 0).mean())
        else:
            raise ValueError(f"Unsupported trailing calendar stat: {stat}")

    return result


def _compute_group_cagr(
    dates: pd.Series,
    values: pd.Series,
    *,
    years: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=dates.index, dtype=float)
    if dates.empty:
        return result

    for row_index, current_date in dates.items():
        if pd.isna(current_date):
            continue
        current_value = values.loc[row_index]
        if pd.isna(current_value) or float(current_value) <= 0:
            continue

        anchor_target = current_date - pd.DateOffset(years=years)
        anchor_mask = (dates <= anchor_target) & values.notna()
        if not anchor_mask.any():
            continue

        anchor_date = dates.loc[anchor_mask].iloc[-1]
        anchor_value = values.loc[anchor_mask].iloc[-1]
        if pd.isna(anchor_date) or pd.isna(anchor_value) or float(anchor_value) <= 0:
            continue

        elapsed_years = (current_date - anchor_date).days / 365.25
        if elapsed_years <= 0:
            continue
        growth_rate = np.exp(np.log(current_value / anchor_value) / elapsed_years) - 1.0
        result.loc[row_index] = float(growth_rate)

    return result


def compute_trailing_calendar_window_stat(
    frame: pd.DataFrame,
    value_series: pd.Series,
    *,
    years: int,
    stat: str,
    min_periods: int = 3,
) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if frame.empty:
        return result

    trade_dates = _to_datetime_series(cast(pd.Series, frame["trade_date"]))
    values = _to_numeric_series(value_series)

    for _, index_values in frame.groupby("symbol", sort=False).groups.items():
        group_index = pd.Index(index_values)
        group_dates = trade_dates.loc[group_index]
        ordered_index = group_dates.sort_values(kind="stable").index
        ordered_dates = trade_dates.loc[ordered_index]
        ordered_values = values.loc[ordered_index]
        result.loc[ordered_index] = _compute_group_trailing_stat(
            ordered_dates,
            ordered_values,
            years=years,
            stat=stat,
            min_periods=min_periods,
        ).to_numpy(dtype=float)

    return result


def compute_calendar_cagr(
    frame: pd.DataFrame,
    value_series: pd.Series,
    *,
    years: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if frame.empty:
        return result

    trade_dates = _to_datetime_series(cast(pd.Series, frame["trade_date"]))
    values = _to_numeric_series(value_series)

    for _, index_values in frame.groupby("symbol", sort=False).groups.items():
        group_index = pd.Index(index_values)
        group_dates = trade_dates.loc[group_index]
        ordered_index = group_dates.sort_values(kind="stable").index
        ordered_dates = trade_dates.loc[ordered_index]
        ordered_values = values.loc[ordered_index]
        result.loc[ordered_index] = _compute_group_cagr(
            ordered_dates,
            ordered_values,
            years=years,
        ).to_numpy(dtype=float)

    return result


_FUNDAMENTAL_SOURCE_DEPENDENCIES = {
    "sales": ("revenue", "operating_revenue"),
    "profit_margin": ("net_profit", "revenue", "operating_revenue"),
    "profit_margin_std_3y": ("net_profit", "revenue", "operating_revenue"),
    "operating_margin": ("operating_profit", "revenue", "operating_revenue"),
    "cfo_margin": (
        "cash_flow_from_operating_activities",
        "revenue",
        "operating_revenue",
    ),
    "cfo_margin_avg_3y": (
        "cash_flow_from_operating_activities",
        "revenue",
        "operating_revenue",
    ),
    "cfo_to_profit": ("cash_flow_from_operating_activities", "net_profit"),
    "cfo_to_profit_median_3y": (
        "cash_flow_from_operating_activities",
        "net_profit",
    ),
    "positive_cfo_ratio_3y": ("cash_flow_from_operating_activities",),
    "positive_cfo_ratio_2y": ("cash_flow_from_operating_activities",),
    "positive_cfo_ratio_3y_min2": ("cash_flow_from_operating_activities",),
    "debt": ("short_term_debt", "long_term_loans"),
    "asset_turnover": ("revenue", "total_assets"),
    "roa": ("net_profit", "total_assets"),
    "leverage": ("total_liabilities", "total_assets"),
    "cfo_to_assets": ("cash_flow_from_operating_activities", "total_assets"),
    "debt_to_assets": ("short_term_debt", "long_term_loans", "total_assets"),
    "debt_to_equity": ("short_term_debt", "long_term_loans", "total_equity"),
    "cash_to_assets": ("cash_and_equivalents", "total_assets"),
    "goodwill_to_assets": ("goodwill", "total_assets"),
    "accrual_ratio": (
        "net_profit",
        "cash_flow_from_operating_activities",
        "total_assets",
    ),
    "receivables_to_revenue": ("accounts_receivable", "revenue"),
    "inventory_to_revenue": ("inventory", "revenue"),
    "working_capital_to_assets": (
        "accounts_receivable",
        "inventory",
        "accounts_payable",
        "total_assets",
    ),
    "net_debt_to_assets": (
        "short_term_debt",
        "long_term_loans",
        "cash_and_equivalents",
        "total_assets",
    ),
    "sales_cagr_3y": ("revenue", "operating_revenue"),
    "eps_cagr_3y": ("basic_earnings_per_share",),
    "valuation_age_days": ("valuation_trade_date",),
}


def _numeric_fundamental_series(fund_df: pd.DataFrame, name: str) -> pd.Series:
    if name not in fund_df.columns:
        return pd.Series(np.nan, index=fund_df.index, dtype=float)
    return pd.to_numeric(fund_df[name], errors="coerce")


def _safe_fundamental_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid_denominator = denominator.where(denominator.notna() & (denominator != 0))
    return (numerator / valid_denominator).replace([np.inf, -np.inf], np.nan)


def _needs_fundamental_feature(
    requested_feature_names: set[str],
    *feature_names: str,
) -> bool:
    return any(name in requested_feature_names for name in feature_names)


def _add_sales_field(fund_df: pd.DataFrame, requested_feature_names: set[str]) -> None:
    if not _needs_fundamental_feature(
        requested_feature_names,
        "sales",
        "delta_sales",
        "growth_sales",
        "profit_margin",
        "operating_margin",
        "cfo_margin",
        "sales_cagr_3y",
    ):
        return
    revenue = _numeric_fundamental_series(fund_df, "revenue")
    operating_revenue = _numeric_fundamental_series(fund_df, "operating_revenue")
    fund_df["sales"] = revenue.combine_first(operating_revenue)


def _add_debt_field(fund_df: pd.DataFrame, requested_feature_names: set[str]) -> None:
    if not _needs_fundamental_feature(
        requested_feature_names,
        "debt",
        "delta_debt",
        "growth_debt",
        "debt_to_assets",
        "debt_to_equity",
        "net_debt_to_assets",
    ):
        return
    short_term_debt = _numeric_fundamental_series(fund_df, "short_term_debt")
    long_term_loans = _numeric_fundamental_series(fund_df, "long_term_loans")
    debt = short_term_debt.fillna(0.0) + long_term_loans.fillna(0.0)
    fund_df["debt"] = debt.where(~(short_term_debt.isna() & long_term_loans.isna()))


def _add_margin_fields(fund_df: pd.DataFrame, requested_feature_names: set[str]) -> None:
    if _needs_fundamental_feature(requested_feature_names, "profit_margin", "profit_margin_std_3y"):
        fund_df["profit_margin"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "net_profit"),
            _numeric_fundamental_series(fund_df, "sales"),
        )
    if "operating_margin" in requested_feature_names:
        fund_df["operating_margin"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "operating_profit"),
            _numeric_fundamental_series(fund_df, "sales"),
        )
    if _needs_fundamental_feature(requested_feature_names, "cfo_margin", "cfo_margin_avg_3y"):
        fund_df["cfo_margin"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "cash_flow_from_operating_activities"),
            _numeric_fundamental_series(fund_df, "sales"),
        )
    if _needs_fundamental_feature(
        requested_feature_names,
        "cfo_to_profit",
        "cfo_to_profit_median_3y",
    ):
        fund_df["cfo_to_profit"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "cash_flow_from_operating_activities"),
            _numeric_fundamental_series(fund_df, "net_profit"),
        )


def _add_structure_ratio_fields(fund_df: pd.DataFrame, requested_feature_names: set[str]) -> None:
    if "asset_turnover" in requested_feature_names:
        fund_df["asset_turnover"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "revenue"),
            _numeric_fundamental_series(fund_df, "total_assets"),
        )
    if "roa" in requested_feature_names:
        fund_df["roa"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "net_profit"),
            _numeric_fundamental_series(fund_df, "total_assets"),
        )
    if "leverage" in requested_feature_names:
        fund_df["leverage"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "total_liabilities"),
            _numeric_fundamental_series(fund_df, "total_assets"),
        )
    if "cfo_to_assets" in requested_feature_names:
        fund_df["cfo_to_assets"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "cash_flow_from_operating_activities"),
            _numeric_fundamental_series(fund_df, "total_assets"),
        )
    if "debt_to_assets" in requested_feature_names:
        fund_df["debt_to_assets"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "debt"),
            _numeric_fundamental_series(fund_df, "total_assets"),
        )
    if "debt_to_equity" in requested_feature_names:
        fund_df["debt_to_equity"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "debt"),
            _numeric_fundamental_series(fund_df, "total_equity"),
        )
    if "cash_to_assets" in requested_feature_names:
        fund_df["cash_to_assets"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "cash_and_equivalents"),
            _numeric_fundamental_series(fund_df, "total_assets"),
        )
    if "goodwill_to_assets" in requested_feature_names:
        fund_df["goodwill_to_assets"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "goodwill"),
            _numeric_fundamental_series(fund_df, "total_assets"),
        )
    if "accrual_ratio" in requested_feature_names:
        accrual = _numeric_fundamental_series(fund_df, "net_profit") - _numeric_fundamental_series(
            fund_df, "cash_flow_from_operating_activities"
        )
        fund_df["accrual_ratio"] = _safe_fundamental_ratio(
            accrual,
            _numeric_fundamental_series(fund_df, "total_assets"),
        )
    if "receivables_to_revenue" in requested_feature_names:
        fund_df["receivables_to_revenue"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "accounts_receivable"),
            _numeric_fundamental_series(fund_df, "revenue"),
        )
    if "inventory_to_revenue" in requested_feature_names:
        fund_df["inventory_to_revenue"] = _safe_fundamental_ratio(
            _numeric_fundamental_series(fund_df, "inventory"),
            _numeric_fundamental_series(fund_df, "revenue"),
        )
    if "working_capital_to_assets" in requested_feature_names:
        working_capital = (
            _numeric_fundamental_series(fund_df, "accounts_receivable")
            + _numeric_fundamental_series(fund_df, "inventory")
            - _numeric_fundamental_series(fund_df, "accounts_payable")
        )
        fund_df["working_capital_to_assets"] = _safe_fundamental_ratio(
            working_capital,
            _numeric_fundamental_series(fund_df, "total_assets"),
        )
    if "net_debt_to_assets" in requested_feature_names:
        net_debt = _numeric_fundamental_series(fund_df, "debt") - _numeric_fundamental_series(
            fund_df, "cash_and_equivalents"
        )
        fund_df["net_debt_to_assets"] = _safe_fundamental_ratio(
            net_debt,
            _numeric_fundamental_series(fund_df, "total_assets"),
        )


def _add_delta_fields(fund_df: pd.DataFrame, requested_feature_names: set[str]) -> None:
    delta_base_features = sorted(
        {
            feat.removeprefix("delta_")
            for feat in requested_feature_names
            if feat.startswith("delta_")
        }
    )
    for base_feature in delta_base_features:
        if base_feature not in fund_df.columns:
            continue
        base_series = pd.to_numeric(fund_df[base_feature], errors="coerce")
        fund_df[f"delta_{base_feature}"] = base_series.groupby(fund_df["symbol"]).diff()


def _add_growth_fields(fund_df: pd.DataFrame, requested_feature_names: set[str]) -> None:
    growth_base_features = sorted(
        {
            feat.removeprefix("growth_")
            for feat in requested_feature_names
            if feat.startswith("growth_")
        }
    )
    for base_feature in growth_base_features:
        if base_feature not in fund_df.columns:
            continue
        current = pd.to_numeric(fund_df[base_feature], errors="coerce")
        previous = current.groupby(fund_df["symbol"]).shift()
        scale = ((current.abs() + previous.abs()) / 2.0).where(
            lambda values: values.notna() & (values != 0)
        )
        growth = (current - previous) / scale
        fund_df[f"growth_{base_feature}"] = growth.replace([np.inf, -np.inf], np.nan)


def _add_calendar_cagr_fields(fund_df: pd.DataFrame, requested_feature_names: set[str]) -> None:
    if "sales_cagr_3y" in requested_feature_names:
        fund_df["sales_cagr_3y"] = compute_calendar_cagr(fund_df, fund_df["sales"], years=3)
    if "eps_cagr_3y" in requested_feature_names:
        fund_df["eps_cagr_3y"] = compute_calendar_cagr(
            fund_df,
            _numeric_fundamental_series(fund_df, "basic_earnings_per_share"),
            years=3,
        )


def _add_trailing_window_fields(fund_df: pd.DataFrame, requested_feature_names: set[str]) -> None:
    specs = (
        ("cfo_margin_avg_3y", "cfo_margin", 3, "mean", 3),
        ("profit_margin_std_3y", "profit_margin", 3, "std", 3),
        ("cfo_to_profit_median_3y", "cfo_to_profit", 3, "median", 3),
        ("positive_cfo_ratio_3y", "cash_flow_from_operating_activities", 3, "positive_ratio", 3),
        ("positive_cfo_ratio_2y", "cash_flow_from_operating_activities", 2, "positive_ratio", 2),
        (
            "positive_cfo_ratio_3y_min2",
            "cash_flow_from_operating_activities",
            3,
            "positive_ratio",
            2,
        ),
    )
    for feature_name, source_name, years, stat, min_periods in specs:
        if feature_name not in requested_feature_names:
            continue
        source = (
            fund_df[source_name]
            if source_name in fund_df.columns
            else _numeric_fundamental_series(fund_df, source_name)
        )
        fund_df[feature_name] = compute_trailing_calendar_window_stat(
            fund_df,
            source,
            years=years,
            stat=stat,
            min_periods=min_periods,
        )


def derive_requested_fundamental_fields(
    fund_df: pd.DataFrame,
    requested_feature_names: set[str],
) -> pd.DataFrame:
    fund_df = fund_df.copy()
    _add_sales_field(fund_df, requested_feature_names)
    _add_debt_field(fund_df, requested_feature_names)
    _add_margin_fields(fund_df, requested_feature_names)
    _add_structure_ratio_fields(fund_df, requested_feature_names)
    if "days_since_report" in requested_feature_names:
        fund_df["report_trade_date"] = fund_df["trade_date"]
    _add_delta_fields(fund_df, requested_feature_names)
    _add_growth_fields(fund_df, requested_feature_names)
    _add_calendar_cagr_fields(fund_df, requested_feature_names)
    _add_trailing_window_fields(fund_df, requested_feature_names)
    return fund_df


def fundamental_source_fields(requested_feature_names: set[str]) -> set[str]:
    fields = set(requested_feature_names)
    for feature in list(requested_feature_names):
        if feature.startswith("delta_"):
            fields.add(feature.removeprefix("delta_"))
        if feature.startswith("growth_"):
            fields.add(feature.removeprefix("growth_"))
    for feature in list(fields):
        fields.update(_FUNDAMENTAL_SOURCE_DEPENDENCIES.get(feature, ()))
    return fields


__all__ = ["derive_requested_fundamental_fields", "fundamental_source_fields"]
