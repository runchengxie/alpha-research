"""Factor computation from daily, valuation, and optional fundamental data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .helpers import add_new_factors, merge_sw_industry_pit

FACTOR_COLS = [
    "factor_size",
    "factor_value",
    "factor_momentum",
    "factor_quality",
    "factor_earnings_yield",
    "factor_lowvol",
    "factor_growth",
    "factor_leverage",
    "factor_beta",
    "factor_liquidity",
    # Factors from locally-landed tushare datasets (zero network traffic):
    "factor_liquidity_flow",  # moneyflow_ths: main-order net inflow
    "factor_chip_concentration",  # holder_structure: top10 float concentration
    "factor_institution_holding",  # holder_structure: top10 inst float hold ratio
    "factor_fund_breadth",  # fund top10: log count of funds listing the stock
    "factor_fund_breadth_change",  # fund top10: signed-log formation-date count change
    "factor_fund_ownership",  # fund top10: summed stock float-share holding ratio
    "factor_fund_ownership_change",  # fund top10: formation-date ratio change
    "factor_dividend_yield",  # daily_basic.dv_ttm (value group)
    "factor_ps_value",  # 1/ps_ttm (value group)
]

VALUE_GROUP = {"factor_value", "factor_earnings_yield", "factor_dividend_yield", "factor_ps_value"}

VALUE_CLUSTER_COL = "factor_value_cluster_z"
VALUE_CLUSTER_MEMBERS = (
    "factor_value",
    "factor_earnings_yield",
    "factor_dividend_yield",
    "factor_ps_value",
)

FUNDAMENTAL_COLS = ["roe", "roa", "netprofit_yoy", "or_yoy", "debt_to_assets"]
EARNINGS_STABILITY_COL = "earnings_stability_8q"
QUALITY_EXTRA_COLS = ["n_cashflow_act", "net_profit", EARNINGS_STABILITY_COL]


def _price_frame(daily: pd.DataFrame) -> pd.DataFrame:
    daily_cols = ["trade_date", "symbol", "close", "pct_chg", "amount"]
    df = daily[daily_cols].copy()
    df = df[df["amount"] > 0].copy()
    return df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _merge_daily_basics(df: pd.DataFrame, basics: pd.DataFrame) -> pd.DataFrame:
    basic_cols = ["trade_date", "symbol", "total_mv", "pb", "pe_ttm", "turnover_rate"]
    df = df.merge(
        basics[basic_cols],
        on=["trade_date", "symbol"],
        how="left",
    )
    df = df[df["total_mv"] > 0].copy()
    return df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _fundamental_columns(fina: pd.DataFrame) -> list[str]:
    wanted = FUNDAMENTAL_COLS + QUALITY_EXTRA_COLS
    return [column for column in wanted if column in fina.columns]


def _prepare_fundamentals(fina: pd.DataFrame) -> pd.DataFrame:
    aligned = fina.rename(columns={"ann_date": "align_date", "symbol": "_sym"}).copy()
    aligned["align_date"] = pd.to_datetime(aligned["align_date"])
    aligned = aligned.dropna(subset=["align_date"])
    sort_columns = ["_sym", "align_date"]
    if "end_date" in aligned.columns:
        sort_columns.append("end_date")
    aligned = aligned.sort_values(sort_columns)
    if "netprofit_yoy" in aligned.columns:
        aligned[EARNINGS_STABILITY_COL] = aligned.groupby("_sym", sort=False)[
            "netprofit_yoy"
        ].transform(lambda series: series.rolling(8, min_periods=4).std())
    aligned = aligned.drop_duplicates(["_sym", "align_date"], keep="last")
    return aligned.set_index("align_date")


def _assign_symbol_fundamentals(
    group: pd.DataFrame,
    symbol_fundamentals: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    if symbol_fundamentals.empty:
        return group

    group = group.copy()
    symbol_fundamentals = symbol_fundamentals.sort_index()
    report_dates = symbol_fundamentals.index.to_numpy()
    trade_dates = group["trade_date"].to_numpy()
    positions = np.searchsorted(report_dates, trade_dates, side="right") - 1
    valid = positions >= 0
    for column in columns:
        values = np.full(len(group), np.nan)
        source_values = symbol_fundamentals[column].to_numpy()
        values[valid] = source_values[positions[valid]]
        group[column] = values
    return group


def _merge_fundamentals(
    df: pd.DataFrame,
    fina: pd.DataFrame | None,
) -> tuple[pd.DataFrame, bool]:
    if fina is None or fina.empty:
        return df, False

    if "ann_date" not in fina.columns or "symbol" not in fina.columns:
        return df, False

    df = df.copy()
    df["_sym"] = df["symbol"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
    df["_idx"] = np.arange(len(df))
    aligned = _prepare_fundamentals(fina)
    columns = _fundamental_columns(aligned)
    if not columns:
        return df.drop(columns=["_idx", "_sym"]), False
    fundamentals_by_symbol = {
        symbol: group.drop(columns=["_sym"])
        for symbol, group in aligned.groupby("_sym", sort=False)
    }
    empty_fundamentals = aligned.iloc[0:0].drop(columns=["_sym"])

    grouped = [
        _assign_symbol_fundamentals(
            group,
            fundamentals_by_symbol.get(symbol, empty_fundamentals),
            columns,
        )
        for symbol, group in df.groupby("_sym", sort=False)
    ]
    merged = pd.concat(grouped, ignore_index=True)
    merged = merged.sort_values("_idx").drop(columns=["_idx", "_sym"])
    return merged, True


def _overlay_formation_fundamentals(
    df: pd.DataFrame,
    panel: pd.DataFrame | None,
) -> tuple[pd.DataFrame, bool]:
    """Prefer exact-date PIT v2 fields while retaining unavailable legacy inputs."""
    if panel is None or panel.empty:
        return df, False
    keys = ["trade_date", "symbol"]
    if panel.duplicated(keys).any():
        raise ValueError("formation_fundamentals has duplicate trade_date/symbol keys")
    fields = [
        column
        for column in (
            "roe",
            "roa",
            "debt_to_assets",
            "n_cashflow_act",
            "net_profit",
            "netprofit_yoy",
            "or_yoy",
            "quarterly_revenue_yoy",
        )
        if column in panel.columns
    ]
    if not fields:
        return df, False
    overlay = panel[[*keys, *fields]].rename(
        columns={column: f"{column}__pit_v2" for column in fields}
    )
    out = df.merge(overlay, on=keys, how="left", validate="one_to_one")
    for column in fields:
        pit_column = f"{column}__pit_v2"
        if column in out.columns:
            out[column] = out[pit_column].combine_first(out[column])
        else:
            out[column] = out[pit_column]
        out = out.drop(columns=pit_column)
    return out, any(out[column].notna().any() for column in fields)


def _add_daily_price_factors(df: pd.DataFrame) -> pd.DataFrame:
    df["ret_1d"] = df.groupby("symbol")["close"].pct_change()
    df["factor_momentum"] = df.groupby("symbol")["close"].transform(
        lambda series: series.pct_change(periods=21).shift(1)
    )

    df["vol20"] = df.groupby("symbol")["ret_1d"].transform(
        lambda series: series.rolling(21, min_periods=10).std().shift(1)
    )
    df["factor_lowvol"] = -df["vol20"]
    return df


def _add_daily_basic_factors(df: pd.DataFrame) -> pd.DataFrame:
    df["factor_size"] = np.log(df["total_mv"] + 1)

    df["pb_clean"] = df["pb"].where(df["pb"] > 0).clip(lower=0.01, upper=100)
    df["factor_value"] = 1.0 / df["pb_clean"]

    df["pe_clean"] = df["pe_ttm"].where(df["pe_ttm"] > 0).clip(lower=1, upper=500)
    df["factor_earnings_yield"] = 1.0 / df["pe_clean"]

    return df


def _winsorize(series: pd.Series, trade_dates: pd.Series) -> pd.Series:
    """Cross-sectional 1%/99% winsorization within each trade date."""
    grouped = series.groupby(trade_dates, sort=False)
    lower = grouped.transform(lambda values: values.quantile(0.01))
    upper = grouped.transform(lambda values: values.quantile(0.99))
    return series.clip(lower=lower, upper=upper, axis=0)


def _add_quality_factor(df: pd.DataFrame, *, has_fina: bool) -> pd.DataFrame:
    """Composite quality factor from ROE, leverage, earnings stability, OCF quality."""
    if not has_fina or "roe" not in df.columns:
        return df

    components: list[pd.Series] = []
    trade_dates = df["trade_date"]

    roe = df["roe"].astype(float)
    components.append(_winsorize(roe, trade_dates))

    if "debt_to_assets" in df.columns:
        lev = -df["debt_to_assets"].astype(float)
        components.append(_winsorize(lev, trade_dates))

    if EARNINGS_STABILITY_COL in df.columns:
        stability = -df[EARNINGS_STABILITY_COL].astype(float)
        components.append(_winsorize(stability, trade_dates))

    if {"n_cashflow_act", "net_profit"} <= set(df.columns):
        np_safe = df["net_profit"].replace(0, np.nan).clip(lower=1e-9)
        cq = df["n_cashflow_act"].astype(float) / np_safe
        cq = cq.where(df["net_profit"] > 0)
        components.append(_winsorize(cq, trade_dates))

    z_parts = []
    for comp in components:
        grouped = comp.groupby(df["trade_date"], sort=False)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        z = (comp - mean) / std
        z_parts.append(z.fillna(0.0))

    quality = sum(z_parts) / len(z_parts)
    df["factor_quality"] = quality.where(roe.notna())
    return df


def _add_fundamental_factors(df: pd.DataFrame, *, has_fina: bool) -> pd.DataFrame:
    if not has_fina:
        return df
    if {"netprofit_yoy", "or_yoy"} <= set(df.columns):
        df["g_np"] = df["netprofit_yoy"].clip(lower=-300, upper=500)
        df["g_or"] = df["or_yoy"].clip(lower=-200, upper=500)
        df["factor_growth"] = df[["g_np", "g_or"]].mean(axis=1)
    if "debt_to_assets" in df.columns:
        df["factor_leverage"] = -df["debt_to_assets"].clip(lower=0, upper=500)
    return df


def _add_beta_factor(df: pd.DataFrame) -> pd.DataFrame:
    """252-day CAPM β, using rolling-sum decomposition for speed."""
    df["ret"] = df["pct_chg"] / 100.0
    df["mkt_ret"] = df.groupby("trade_date")["ret"].transform("mean")
    df["ret_mkt"] = df["ret"] * df["mkt_ret"]
    df["mkt_ret_sq"] = df["mkt_ret"] * df["mkt_ret"]

    grouped = df.groupby("symbol", sort=False)
    window, min_periods = 252, 126
    rolling = {
        column: grouped[column].rolling(window, min_periods=min_periods)
        for column in ("ret_mkt", "ret", "mkt_ret", "mkt_ret_sq")
    }
    sum_rm = rolling["ret_mkt"].sum().reset_index(level=0, drop=True)
    sum_r = rolling["ret"].sum().reset_index(level=0, drop=True)
    sum_m = rolling["mkt_ret"].sum().reset_index(level=0, drop=True)
    sum_m2 = rolling["mkt_ret_sq"].sum().reset_index(level=0, drop=True)
    n = grouped["ret"].rolling(window, min_periods=min_periods).count()
    n = n.reset_index(level=0, drop=True)

    cov_num = sum_rm / n - (sum_r / n) * (sum_m / n)
    var_den = sum_m2 / n - (sum_m / n) ** 2
    raw_beta = cov_num / var_den.replace(0, np.nan)
    df["factor_beta"] = -raw_beta
    return df.drop(columns=["ret_mkt", "mkt_ret_sq"])


def _add_liquidity_factor(df: pd.DataFrame) -> pd.DataFrame:
    df["turn_clean"] = df["turnover_rate"].clip(lower=0.01, upper=100)
    df["factor_liquidity"] = -df["turn_clean"]
    return df


def _standardize_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    active = [column for column in FACTOR_COLS if column in df.columns]
    has_industry = "industry_l1" in df.columns and df["industry_l1"].notna().any()

    for column in active:
        grouped = df.groupby("trade_date", sort=False)[column]
        lower = grouped.transform(lambda series: series.quantile(0.01))
        upper = grouped.transform(lambda series: series.quantile(0.99))
        df[column] = df[column].clip(lower=lower, upper=upper, axis=0)

    for column in active:
        if has_industry:
            grp = df.groupby(["trade_date", "industry_l1"], sort=False, dropna=False)[column]
            demeaned = df[column] - grp.transform("mean")
            df[f"{column}_z"] = demeaned
        else:
            df[f"{column}_z"] = df[column].copy()

    for column in active:
        zcol = f"{column}_z"
        grouped = df.groupby("trade_date", sort=False)[zcol]
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        df[zcol] = (df[zcol] - mean) / std.replace(0, np.nan)
    return df


def compute_factors(
    daily: pd.DataFrame,
    basics: pd.DataFrame,
    fina: pd.DataFrame | None = None,
    cashflow: pd.DataFrame | None = None,
    *,
    aux: dict | None = None,
    sw_membership: pd.DataFrame | None = None,
    rebalance_dates: pd.DatetimeIndex | None = None,
    formation_fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute style factors per stock per date.

    If fina_indicator data is provided, Growth, Leverage and the composite
    Quality factor are aligned by announcement date so the factor frame does
    not look ahead. ``cashflow`` (OCF / net profit) is merged into ``fina``
    when supplied to feed the cashflow-quality sub-indicator.

    ``aux`` (optional) carries locally-landed tushare datasets keyed by name:
    ``moneyflow_ths``, ``holder_structure``, ``fund_top10_portfolio_features``
    and ``daily_basic_extra`` (dv_ttm / ps_ttm). These add auxiliary factors
    with zero network traffic. The ``fund_top10_portfolio_features`` slot
    expects a consistent per-fund top-10 public-fund panel already PIT
    materialized to the formation dates by the caller.

    ``sw_membership`` (optional) is the PIT SW-L1 membership long table from
    ``load_sw_industry_membership``. When present, every factor is demeaned
    within its L1 industry before the cross-sectional z-score.

    ``rebalance_dates`` optionally limits the expensive fundamentals, auxiliary
    and industry joins to formation dates after daily rolling price factors have
    been calculated. The workflow uses this path because portfolio membership
    changes only at month end.

    ``formation_fundamentals`` may provide exact formation-date PIT v2 fields.
    Non-null values override the corresponding legacy fundamentals. Growth
    remains on legacy ``netprofit_yoy`` / ``or_yoy`` because those fields are
    not present in the current PIT v2 contract.
    """
    if fina is not None and cashflow is not None and not cashflow.empty:
        merge_on = [c for c in ("symbol", "end_date", "ann_date") if c in cashflow.columns]
        fina = fina.merge(cashflow, on=merge_on, how="left")

    df = _price_frame(daily)
    df = _add_daily_price_factors(df)
    df = _add_beta_factor(df)
    if rebalance_dates is not None:
        normalized_dates = rebalance_dates.normalize()  # type: ignore
        df = df[df["trade_date"].isin(normalized_dates)].copy()
    df = _merge_daily_basics(df, basics)
    df = _add_daily_basic_factors(df)
    df = _add_liquidity_factor(df)
    df, has_fina = _merge_fundamentals(df, fina)
    df, has_pit_panel = _overlay_formation_fundamentals(df, formation_fundamentals)
    has_fina = has_fina or has_pit_panel
    df = _add_fundamental_factors(df, has_fina=has_fina)
    df = _add_quality_factor(df, has_fina=has_fina)
    df = add_new_factors(df, aux=aux)
    df = merge_sw_industry_pit(df, sw_membership)
    active = [column for column in FACTOR_COLS if column in df.columns]
    df = df[["trade_date", "symbol", "industry_l1", *active]].copy()
    df = _standardize_factors(df)
    value_members = [
        f"{column}_z" for column in VALUE_CLUSTER_MEMBERS if f"{column}_z" in df.columns
    ]
    if len(value_members) >= 2:
        df[VALUE_CLUSTER_COL] = df[value_members].mean(axis=1)
    active = [column for column in FACTOR_COLS if column in df.columns]
    n_factors = len(active)
    cov = df["industry_l1"].notna().mean() if "industry_l1" in df.columns else 0.0
    print(
        f"[factors] {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}, "
        f"{len(df)} rows, {df['symbol'].nunique()} stocks, {n_factors} factors, "
        f"industry coverage={cov:.1%}"
    )
    return df
