from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..pipeline.support import _parse_window_config, parse_feature_windows


def _numeric_column_or_nan(group: pd.DataFrame, column: str) -> pd.Series:
    if column in group.columns:
        return pd.to_numeric(group[column], errors="coerce")
    return pd.Series(np.nan, index=group.index, dtype=float)


def _has_columns(group: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return all(column in group.columns for column in columns)


def _safe_ratio(
    group: pd.DataFrame,
    needed: set[str],
    numerator_col: str,
    denominator_col: str,
    out_col: str,
) -> None:
    if out_col not in needed:
        return
    if numerator_col not in group.columns or denominator_col not in group.columns:
        return
    numerator = pd.to_numeric(group[numerator_col], errors="coerce")
    denominator = pd.to_numeric(group[denominator_col], errors="coerce")
    valid_denominator = denominator.where(denominator.notna() & (denominator != 0))
    ratio = numerator / valid_denominator
    group[out_col] = ratio.replace([np.inf, -np.inf], np.nan)


def _add_sma_features(
    group: pd.DataFrame,
    *,
    features: list[str],
    feature_params: dict,
    needed: set[str],
    price_series: pd.Series,
) -> None:
    sma_windows = set(parse_feature_windows(features, "sma_"))
    sma_windows.update(parse_feature_windows(features, "sma_", "_diff"))
    if not sma_windows:
        sma_windows = _parse_window_config(feature_params.get("sma_windows"))
    for win in sorted(sma_windows):
        group[f"sma_{win}"] = ta.sma(price_series, length=win)
        if f"sma_{win}_diff" in needed:
            group[f"sma_{win}_diff"] = group[f"sma_{win}"].pct_change()


def _add_rsi_features(
    group: pd.DataFrame,
    *,
    features: list[str],
    feature_params: dict,
    price_series: pd.Series,
) -> None:
    rsi_lengths = set(parse_feature_windows(features, "rsi_"))
    if not rsi_lengths:
        rsi_lengths = _parse_window_config(feature_params.get("rsi"))
    for length in sorted(rsi_lengths):
        group[f"rsi_{length}"] = ta.rsi(price_series, length=length)


def _add_macd_feature(
    group: pd.DataFrame,
    *,
    feature_params: dict,
    needed: set[str],
    price_series: pd.Series,
) -> None:
    if "macd_hist" not in needed:
        return
    macd_cfg = feature_params.get("macd", [12, 26, 9])
    macd_fast, macd_slow, macd_signal = macd_cfg
    macd = ta.macd(price_series, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    col_name = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"
    group["macd_hist"] = macd[col_name] if macd is not None and col_name in macd.columns else np.nan


def _add_volume_features(
    group: pd.DataFrame,
    *,
    features: list[str],
    feature_params: dict,
    needed: set[str],
) -> None:
    volume_windows = set(parse_feature_windows(features, "volume_sma", "_ratio"))
    if not volume_windows:
        volume_windows = _parse_window_config(feature_params.get("volume_sma_windows"))
    for win in sorted(volume_windows):
        volume_sma = ta.sma(group["vol"], length=win)
        if volume_sma is None:
            volume_sma = group["vol"].rolling(window=win).mean()
        group[f"volume_sma{win}"] = volume_sma
        if f"volume_sma{win}_ratio" in needed:
            group[f"volume_sma{win}_ratio"] = group["vol"] / group[f"volume_sma{win}"]


def _add_return_features(
    group: pd.DataFrame,
    *,
    features: list[str],
    feature_params: dict,
    price_series: pd.Series,
) -> None:
    ret_windows = set(parse_feature_windows(features, "ret_"))
    if not ret_windows:
        ret_windows = _parse_window_config(feature_params.get("ret_windows"))
    for win in sorted(ret_windows):
        group[f"ret_{win}"] = price_series.pct_change(win)


def _add_realized_volatility_features(
    group: pd.DataFrame,
    *,
    features: list[str],
    feature_params: dict,
    price_series: pd.Series,
) -> None:
    rv_windows = set(parse_feature_windows(features, "rv_"))
    if not rv_windows:
        rv_windows = _parse_window_config(feature_params.get("rv_windows"))
    if not rv_windows:
        return
    daily_return = price_series.pct_change().replace([np.inf, -np.inf], np.nan)
    for win in sorted(rv_windows):
        group[f"rv_{win}"] = daily_return.rolling(window=win).std(ddof=0)


def _add_technical_features(
    group: pd.DataFrame,
    *,
    features: list[str],
    feature_params: dict,
    needed: set[str],
    price_series: pd.Series,
) -> None:
    _add_sma_features(
        group,
        features=features,
        feature_params=feature_params,
        needed=needed,
        price_series=price_series,
    )
    _add_rsi_features(
        group,
        features=features,
        feature_params=feature_params,
        price_series=price_series,
    )
    _add_macd_feature(
        group,
        feature_params=feature_params,
        needed=needed,
        price_series=price_series,
    )
    _add_volume_features(
        group,
        features=features,
        feature_params=feature_params,
        needed=needed,
    )
    _add_return_features(
        group,
        features=features,
        feature_params=feature_params,
        price_series=price_series,
    )
    _add_realized_volatility_features(
        group,
        features=features,
        feature_params=feature_params,
        price_series=price_series,
    )
    if "log_vol" in needed:
        group["log_vol"] = np.log1p(group["vol"].clip(lower=0))


def _add_base_fundamental_fields(group: pd.DataFrame, needed: set[str]) -> None:
    if needed.intersection({"sales", "profit_margin", "operating_margin", "cfo_margin"}):
        revenue = _numeric_column_or_nan(group, "revenue")
        operating_revenue = _numeric_column_or_nan(group, "operating_revenue")
        group["sales"] = revenue.combine_first(operating_revenue)
    if needed.intersection({"debt", "debt_to_assets", "debt_to_equity", "net_debt_to_assets"}):
        short_term_debt = _numeric_column_or_nan(group, "short_term_debt")
        long_term_loans = _numeric_column_or_nan(group, "long_term_loans")
        debt = short_term_debt.fillna(0.0) + long_term_loans.fillna(0.0)
        group["debt"] = debt.where(~(short_term_debt.isna() & long_term_loans.isna()))


def _add_standard_fundamental_ratios(group: pd.DataFrame, needed: set[str]) -> None:
    ratio_specs = (
        ("net_profit", "sales", "profit_margin"),
        ("operating_profit", "sales", "operating_margin"),
        ("cash_flow_from_operating_activities", "sales", "cfo_margin"),
        ("cash_flow_from_operating_activities", "net_profit", "cfo_to_profit"),
        ("revenue", "total_assets", "asset_turnover"),
        ("net_profit", "total_assets", "roa"),
        ("total_liabilities", "total_assets", "leverage"),
        ("cash_flow_from_operating_activities", "total_assets", "cfo_to_assets"),
        ("debt", "total_assets", "debt_to_assets"),
        ("debt", "total_equity", "debt_to_equity"),
        ("cash_and_equivalents", "total_assets", "cash_to_assets"),
        ("goodwill", "total_assets", "goodwill_to_assets"),
        ("accounts_receivable", "revenue", "receivables_to_revenue"),
        ("inventory", "revenue", "inventory_to_revenue"),
    )
    for numerator_col, denominator_col, out_col in ratio_specs:
        _safe_ratio(group, needed, numerator_col, denominator_col, out_col)


def _add_accrual_ratio(group: pd.DataFrame, needed: set[str]) -> None:
    if "accrual_ratio" not in needed:
        return
    if not _has_columns(
        group,
        ("net_profit", "cash_flow_from_operating_activities", "total_assets"),
    ):
        return
    net_profit = pd.to_numeric(group["net_profit"], errors="coerce")
    cfo = pd.to_numeric(group["cash_flow_from_operating_activities"], errors="coerce")
    total_assets = pd.to_numeric(group["total_assets"], errors="coerce")
    valid_assets = total_assets.where(total_assets.notna() & (total_assets != 0))
    accrual = (net_profit - cfo) / valid_assets
    group["accrual_ratio"] = accrual.replace([np.inf, -np.inf], np.nan)


def _add_working_capital_ratio(group: pd.DataFrame, needed: set[str]) -> None:
    if "working_capital_to_assets" not in needed:
        return
    if not _has_columns(
        group,
        ("accounts_receivable", "inventory", "accounts_payable", "total_assets"),
    ):
        return
    receivables = pd.to_numeric(group["accounts_receivable"], errors="coerce")
    inventory = pd.to_numeric(group["inventory"], errors="coerce")
    payables = pd.to_numeric(group["accounts_payable"], errors="coerce")
    total_assets = pd.to_numeric(group["total_assets"], errors="coerce")
    valid_assets = total_assets.where(total_assets.notna() & (total_assets != 0))
    working_capital = receivables + inventory - payables
    ratio = working_capital / valid_assets
    group["working_capital_to_assets"] = ratio.replace([np.inf, -np.inf], np.nan)


def _add_net_debt_ratio(group: pd.DataFrame, needed: set[str]) -> None:
    if "net_debt_to_assets" not in needed:
        return
    if not _has_columns(group, ("debt", "cash_and_equivalents", "total_assets")):
        return
    debt = pd.to_numeric(group["debt"], errors="coerce")
    cash_and_equivalents = pd.to_numeric(group["cash_and_equivalents"], errors="coerce")
    total_assets = pd.to_numeric(group["total_assets"], errors="coerce")
    valid_assets = total_assets.where(total_assets.notna() & (total_assets != 0))
    net_debt = debt - cash_and_equivalents
    ratio = net_debt / valid_assets
    group["net_debt_to_assets"] = ratio.replace([np.inf, -np.inf], np.nan)


def _add_fundamental_features(group: pd.DataFrame, needed: set[str]) -> None:
    _add_base_fundamental_fields(group, needed)
    _add_standard_fundamental_ratios(group, needed)
    _add_accrual_ratio(group, needed)
    _add_working_capital_ratio(group, needed)
    _add_net_debt_ratio(group, needed)


def _add_forward_return_label(
    group: pd.DataFrame,
    *,
    price_col: str,
    target: str,
    label_shift_days: int,
    label_horizon_days: int,
    label_horizon_mode: str,
    label_next_rebalance_map: dict[pd.Timestamp, pd.Timestamp] | None,
) -> None:
    if label_shift_days > 0:
        shifted_price = group[price_col].shift(-label_shift_days)
    else:
        shifted_price = group[price_col]
    entry_price = shifted_price
    if label_horizon_mode == "next_rebalance" and label_next_rebalance_map is not None:
        exit_base = group["trade_date"].map(label_next_rebalance_map)
        shifted_by_date = pd.Series(shifted_price.values, index=group["trade_date"])
        exit_price = exit_base.map(shifted_by_date)
    else:
        exit_price = shifted_price.shift(-label_horizon_days)
    group[target] = exit_price / entry_price - 1.0


def engineer_symbol_features(
    group: pd.DataFrame,
    *,
    features: list[str],
    feature_params: dict,
    price_col: str,
    target: str,
    label_shift_days: int,
    label_horizon_days: int,
    label_horizon_mode: str,
    label_next_rebalance_map: dict[pd.Timestamp, pd.Timestamp] | None,
) -> pd.DataFrame:
    group = group.sort_values("trade_date").copy()
    needed = set(features)
    price_series = pd.to_numeric(group[price_col], errors="coerce")
    _add_technical_features(
        group,
        features=features,
        feature_params=feature_params,
        needed=needed,
        price_series=price_series,
    )
    _add_fundamental_features(group, needed)
    _add_forward_return_label(
        group,
        price_col=price_col,
        target=target,
        label_shift_days=label_shift_days,
        label_horizon_days=label_horizon_days,
        label_horizon_mode=label_horizon_mode,
        label_next_rebalance_map=label_next_rebalance_map,
    )
    return group
