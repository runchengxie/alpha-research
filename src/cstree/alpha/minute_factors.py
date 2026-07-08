"""Minute-level factors ported from guan-factor-research-framework.

These factors require only 1-minute OHLCV bars (TuShare ``pro.mins``),
NOT Level-2 order-book data.  They are designed to drop into the
``cstree.alpha`` feature pipeline via ``feature_dataset``.

Factor groups
-------------
* morning_vwap      — opening auction VWAP signal
* volume_ratio      — relative volume vs 20-day average
* volume_perc_1_8   — intraday volume concentration (8 percentiles)
* smart_money       — q-factor: "smart money" VWAP / total VWAP

References
----------
Original implementations (with CuPy GPU acceleration):
  ~/code/guan-factor-research-framework/src/research/minute/
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# morning_vwap
# ═══════════════════════════════════════════════════════════════════════════════


def compute_morning_vwap(
    minute_bars: pd.DataFrame,
    *,
    open_minutes: int = 1,
    price_col: str = "close",
    volume_col: str = "vol",
) -> pd.Series:
    """Opening-period VWAP for each stock on each trading day.

    Args:
        minute_bars: Long-format DataFrame with columns
            ``trade_date``, ``symbol``, ``time`` (HHMMSS int),
            ``close``, ``vol``.
        open_minutes: Number of minutes after open to include.
        price_col, volume_col: Column names for price and volume.

    Returns:
        Series indexed by (trade_date, symbol) with morning VWAP values.
    """
    df = minute_bars.copy()
    # Filter to opening minutes (time < 093000 + open_minutes*100)
    open_cutoff = 93000 + open_minutes * 100
    if "time" in df.columns:
        df = df[df["time"] <= open_cutoff]

    if volume_col not in df.columns or price_col not in df.columns:
        return pd.Series(dtype=float, name="morning_vwap")

    df["_notional"] = df[price_col].astype(float) * df[volume_col].astype(float)
    grouped = df.groupby(["trade_date", "symbol"])
    total_notional = grouped["_notional"].sum()
    total_volume = grouped[volume_col].sum()
    vwap = total_notional / total_volume.replace(0, np.nan)
    vwap.name = "morning_vwap"
    return vwap


# ═══════════════════════════════════════════════════════════════════════════════
# volume_ratio
# ═══════════════════════════════════════════════════════════════════════════════


def compute_volume_ratio(
    daily: pd.DataFrame,
    *,
    volume_col: str = "vol",
    window: int = 20,
) -> pd.Series:
    """Volume relative to rolling N-day average.

    Args:
        daily: Long-format DataFrame with ``trade_date``, ``symbol``, ``vol``.
        volume_col: Column name for volume.
        window: Rolling window in trading days.

    Returns:
        Series: volume_ratio = vol / mean(vol over past N days).
    """
    df = daily[["trade_date", "symbol", volume_col]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()
    result = df.groupby("symbol")[volume_col].transform(
        lambda x: x / x.shift(1).rolling(window, min_periods=5).mean()
    )
    result.name = "volume_ratio"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# volume_perc_1_8
# ═══════════════════════════════════════════════════════════════════════════════


def compute_volume_perc(
    minute_bars: pd.DataFrame,
    *,
    volume_col: str = "vol",
    n_buckets: int = 8,
) -> pd.DataFrame:
    """Intraday volume concentration by percentile bucket.

    Divides the trading day into *n_buckets* equal-sized time windows
    and computes the fraction of daily volume in each, per stock per day.

    Args:
        minute_bars: Long-format with ``trade_date``, ``symbol``,
            ``time`` (HHMMSS int), ``vol``.
        n_buckets: Number of time buckets (default 8).

    Returns:
        DataFrame with columns ``volume_perc1`` … ``volume_percN``,
        indexed by (trade_date, symbol).
    """
    df = minute_bars[["trade_date", "symbol", "time", volume_col]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # Map time to bucket: 0930-1500 → 0..n_buckets-1
    trading_minutes = 240  # 4 hours
    df["minute_of_day"] = (df["time"] // 100 - 930) * 60 + (df["time"] % 100)  # rough
    df["minute_of_day"] = df["minute_of_day"].clip(0, trading_minutes - 1)
    df["bucket"] = (df["minute_of_day"] / trading_minutes * n_buckets).astype(int)
    df["bucket"] = df["bucket"].clip(0, n_buckets - 1)

    daily_vol = df.groupby(["trade_date", "symbol"])[volume_col].transform("sum")
    bucket_vol = df.groupby(["trade_date", "symbol", "bucket"])[volume_col].sum()
    perc = bucket_vol / daily_vol.groupby(["trade_date", "symbol"]).first()

    result = perc.unstack("bucket").fillna(0)
    result.columns = [f"volume_perc{i + 1}" for i in range(n_buckets)]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# smart_money q-factor
# ═══════════════════════════════════════════════════════════════════════════════


def compute_smart_money_q(
    minute_bars: pd.DataFrame,
    *,
    beta: float = 0.1,
    threshold: float = 0.2,
    price_col: str = "close",
    volume_col: str = "vol",
) -> pd.Series:
    """Smart-money q-factor: ratio of "informed" VWAP to overall VWAP.

    Identifies minutes with high |return| / volume^beta as "smart" money,
    limits to the top *threshold* fraction of daily volume, then computes
    VWAP of smart trades vs total VWAP.

    Args:
        minute_bars: Long-format with ``trade_date``, ``symbol``,
            ``open``, ``close``, ``vol``.
        beta: Exponent for volume penalty (0.1 = mild, 0.5 = strong).
        threshold: Fraction of daily volume to include as "smart".
        price_col, volume_col: Column names.

    Returns:
        Series indexed by (trade_date, symbol).  > 1 means smart money
        paid above VWAP (bullish); < 1 means below (bearish).
    """
    needed = ["trade_date", "symbol", "open", "close", volume_col]
    for col in needed:
        if col not in minute_bars.columns:
            raise ValueError(f"Missing column: {col}")

    df = minute_bars[needed].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # Return magnitude
    eps = 1e-10
    df["ret_abs"] = np.abs(
        (df["close"].astype(float) - df["open"].astype(float))
        / np.maximum(df["open"].astype(float), eps)
    )
    # Smart metric: |ret| / vol^beta
    df["s_metric"] = df["ret_abs"] / np.maximum(df[volume_col].astype(float) ** beta, eps)
    # Price * volume for notional
    df["pv"] = df["close"].astype(float) * df[volume_col].astype(float)

    results = []
    for (dt, sym), grp in df.groupby(["trade_date", "symbol"]):
        grp = grp.sort_values("s_metric", ascending=False)
        grp["cum_vol"] = grp[volume_col].cumsum()
        total_vol = grp[volume_col].sum()
        if total_vol <= 0:
            continue
        vol_cut = total_vol * threshold
        smart = grp[grp["cum_vol"] <= vol_cut]
        if smart.empty:
            continue
        smart_vwap = smart["pv"].sum() / smart[volume_col].sum()
        total_vwap = grp["pv"].sum() / total_vol
        q = smart_vwap / total_vwap if total_vwap > 0 else np.nan
        if np.isfinite(q):
            results.append({"trade_date": dt, "symbol": sym, "q_factor": q})

    if not results:
        return pd.Series(dtype=float, name="q_factor")
    out = pd.DataFrame(results).set_index(["trade_date", "symbol"])["q_factor"]
    out.name = "smart_money_q"
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Factor inventory (for documentation / feature discovery)
# ═══════════════════════════════════════════════════════════════════════════════

MINUTE_FACTOR_REGISTRY = {
    "morning_vwap": {
        "fn": "compute_morning_vwap",
        "description": "Opening-period VWAP (1-10 min after open)",
        "data": "TuShare pro.mins (1-min OHLCV)",
        "source": "guan: src/research/minute/morning_vwap.py",
    },
    "volume_ratio": {
        "fn": "compute_volume_ratio",
        "description": "Daily volume / 20-day average volume",
        "data": "TuShare pro.daily (daily volume)",
        "source": "guan: src/research/minute/volume_ratio.py",
    },
    "volume_perc": {
        "fn": "compute_volume_perc",
        "description": "Intraday volume concentration (8 time buckets)",
        "data": "TuShare pro.mins (1-min OHLCV)",
        "source": "guan: mf_qpgru / mf_distribution_22.py",
    },
    "smart_money_q": {
        "fn": "compute_smart_money_q",
        "description": "Smart-money VWAP / total VWAP ratio",
        "data": "TuShare pro.mins (1-min OHLCV)",
        "source": "guan: src/research/minute/smart_money.py",
    },
    # ── Additional factors (stubs — implementation in guan source) ──
    "late_skew_ret": {
        "description": "Return skew in final 30 min vs full day",
        "source": "guan: mf_qpgru",
    },
    "down_vol_perc": {
        "description": "Volume fraction during down-moves",
        "source": "guan: mf_qpgru",
    },
    "corr_volume_ret": {
        "description": "Correlation between volume and return across minutes",
        "source": "guan: mf_qpgru",
    },
    "corr_volume_amplitude": {
        "description": "Correlation between volume and price amplitude",
        "source": "guan: mf_qpgru",
    },
    "avg_trade_size": {
        "description": "Average trade size inferred from vol/count",
        "data": "TuShare pro.mins (requires trade_count)",
        "source": "guan: trades_minute_stats",
    },
    "trade_count_skew": {
        "description": "Skewness of trade count distribution",
        "data": "TuShare pro.mins (requires trade_count)",
        "source": "guan: trades_minute_stats",
    },
}


__all__ = [
    "MINUTE_FACTOR_REGISTRY",
    "compute_morning_vwap",
    "compute_smart_money_q",
    "compute_volume_perc",
    "compute_volume_ratio",
]
