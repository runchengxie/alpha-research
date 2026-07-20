# ruff: noqa: RUF002
"""Factor computation for StyleReplica-A80B20-v0.

All factors are computed as daily cross-sectional percentile ranks (0–1 range),
which avoids unit conflicts between different factor dimensions.

Factors computed:
- RESVOL:     residual volatility (imported from .resvol)
- LIQUIDITY:  average daily turnover rate over 20 days
- SIZE:       log market capitalization (negated → smaller = higher)
- BETA:       rolling 60-day market beta
- MOM20:      20-day momentum (return)
- MOM120:     120-day momentum (return)
- INDUSTRY_MOM: recent industry-level momentum
- VOL_CONVERGENCE: -(Vol_20 / Vol_120)  (for B-leg)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from .resvol import compute_resvol_factor


def _cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide DataFrame to cross-sectional percentile ranks (0–1)."""
    return frame.rank(axis=1, method="average", pct=True, na_option="bottom")


def _winsorize_frame(frame: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Winsorize each column at the given quantiles."""
    result = frame.copy()
    for col in result.columns:
        col_data = result[col].dropna()
        if len(col_data) < 3:
            continue
        lo = col_data.quantile(lower)
        hi = col_data.quantile(upper)
        result[col] = result[col].clip(lo, hi)
    return result


# ── Returns helpers ────────────────────────────────────────────────────────────


def compute_returns_from_prices(
    price_panel: pd.DataFrame,
    *,
    method: Literal["simple", "log"] = "simple",
) -> pd.DataFrame:
    """Compute daily returns from a wide price panel."""
    if method == "log":
        return np.log(price_panel / price_panel.shift(1))
    return price_panel.pct_change()


# ── Factor: Beta ───────────────────────────────────────────────────────────────


def compute_beta_factor(
    returns_panel: pd.DataFrame,
    market_returns: pd.Series | None = None,
    *,
    window: int = 60,
    min_obs: int = 40,
) -> pd.DataFrame:
    """Compute rolling 60-day market beta for each stock.

    Returns a wide DataFrame (dates × symbols) of beta values.
    Higher beta → higher score for A-leg.
    """
    if market_returns is None:
        market_returns = returns_panel.mean(axis=1)

    symbols = list(returns_panel.columns)
    dates = returns_panel.index
    mkt = market_returns.reindex(dates).values.astype(float)
    n = len(dates)

    result = pd.DataFrame(np.nan, index=dates, columns=symbols, dtype=float)

    for symbol in symbols:
        r = returns_panel[symbol].values.astype(float)
        for i in range(min_obs - 1, n):
            start = max(0, i - window + 1)
            y = r[start : i + 1]
            x = mkt[start : i + 1]
            valid = np.isfinite(y) & np.isfinite(x)
            if valid.sum() < min_obs:
                continue
            yv = y[valid]
            xv = x[valid]
            x_centered = xv - xv.mean()
            denominator = float(np.dot(x_centered, x_centered))
            if denominator < 1e-12:
                continue
            # OLS covariance and variance must use the same normalization.
            # The centered-sum form avoids the former np.cov(ddof=1) /
            # np.var(ddof=0) mismatch, which inflated beta by n/(n-1).
            beta = float(np.dot(yv - yv.mean(), x_centered) / denominator)
            result.iloc[i, result.columns.get_loc(symbol)] = beta

    return result


# ── Factor: Size ───────────────────────────────────────────────────────────────


def compute_size_factor(
    market_cap_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compute log market cap. Negate so smaller stocks get higher rank.

    Args:
        market_cap_panel: Wide DataFrame (dates × symbols) of market cap in CNY.

    Returns:
        Wide DataFrame of -log(market_cap).
    """
    size = cast(pd.DataFrame, np.log(market_cap_panel.replace(0, np.nan)))
    return -size  # negate: smaller stocks → higher values


# ── Factor: Liquidity ──────────────────────────────────────────────────────────


def compute_liquidity_factor(
    turnover_panel: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.DataFrame:
    """Compute average daily turnover rate over `window` days.

    Args:
        turnover_panel: Wide DataFrame (dates × symbols) of daily turnover rate (%).
        window: Rolling average window.

    Returns:
        Wide DataFrame of average turnover.
    """
    return cast(
        pd.DataFrame,
        turnover_panel.rolling(window=window, min_periods=max(5, window // 2)).mean(),
    )


# ── Factor: Momentum ───────────────────────────────────────────────────────────


def compute_momentum_factor(
    price_panel: pd.DataFrame,
    *,
    window: int = 20,
    log_return: bool = True,
) -> pd.DataFrame:
    """Compute N-day momentum for each stock.

    Args:
        price_panel: Wide DataFrame (dates × symbols) of adjusted close prices.
        window: Lookback window in trading days.
        log_return: If True, use log returns; otherwise simple returns.

    Returns:
        Wide DataFrame of N-day returns.
    """
    if log_return:
        return np.log(price_panel / price_panel.shift(window))
    return price_panel.pct_change(periods=window)


# ── Factor: Industry Momentum ──────────────────────────────────────────────────


def compute_industry_momentum(
    price_panel: pd.DataFrame,
    industry_map: pd.Series,
    *,
    window: int = 20,
) -> pd.DataFrame:
    """Compute industry-level momentum and assign to each stock.

    Industry momentum is the equal-weighted average return of all stocks
    in that industry over the lookback window. Each stock inherits its
    industry's momentum value.

    Args:
        price_panel: Wide DataFrame (dates × symbols) of prices.
        industry_map: Series indexed by symbol, values = industry labels.
        window: Lookback window in trading days.

    Returns:
        Wide DataFrame (dates × symbols) of industry momentum values.
    """
    returns = price_panel.pct_change()
    industry_returns: dict[str, pd.Series] = {}

    for industry_name in industry_map.dropna().unique():
        matching = cast(pd.Series, industry_map.loc[industry_map.eq(industry_name)])
        members = matching.index
        member_cols = [s for s in members if s in returns.columns]
        if not member_cols:
            continue
        industry_returns[industry_name] = returns[member_cols].mean(axis=1)

    if not industry_returns:
        return pd.DataFrame(np.nan, index=price_panel.index, columns=price_panel.columns)

    ind_wide = pd.DataFrame(industry_returns, index=price_panel.index)
    ind_mom = ind_wide.pct_change(periods=window)

    result = pd.DataFrame(np.nan, index=price_panel.index, columns=price_panel.columns)
    for symbol in price_panel.columns:
        ind = industry_map.get(symbol)
        if ind and ind in ind_mom.columns:
            result[symbol] = ind_mom[ind]

    return result


# ── Factor: Volatility Convergence (B-leg) ─────────────────────────────────────


def compute_vol_convergence_factor(
    returns_panel: pd.DataFrame,
    *,
    short_window: int = 20,
    long_window: int = 120,
    min_obs_short: int = 15,
    min_obs_long: int = 80,
) -> pd.DataFrame:
    """Compute volatility convergence: -(Vol_short / Vol_long).

    Positive values mean short-term volatility is declining relative to
    long-term volatility — stocks that are "calming down".

    Args:
        returns_panel: Wide DataFrame (dates × symbols) of daily returns.
        short_window: Short-term volatility window (default 20).
        long_window: Long-term volatility window (default 120).
        min_obs_short: Minimum observations for short vol.
        min_obs_long: Minimum observations for long vol.

    Returns:
        Wide DataFrame of volatility convergence values.
    """
    vol_short = returns_panel.rolling(window=short_window, min_periods=min_obs_short).std()
    vol_long = returns_panel.rolling(window=long_window, min_periods=min_obs_long).std()

    ratio = vol_short / vol_long.replace(0, np.nan)
    return -ratio  # negative ratio → short vol < long vol → positive convergence


# ── Factor: Low RESVOL (B-leg) ─────────────────────────────────────────────────


def compute_low_resvol_factor(
    resvol_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Negate RESVOL so lower residual volatility → higher score.

    Used for B-leg where we want low-volatility stocks.
    """
    return -resvol_panel


# ── Factor: Volume Activity (minute-level, A-leg) ──────────────────────────────


def compute_volume_activity_factor(
    factor_root: str | Path | None = None,
    *,
    factor_name: str = "volume_volatility",
) -> pd.DataFrame | None:
    """Load pre-computed minute-level volume activity factor.

    Reads from the factor panel directory produced by the minute factor pipeline.
    Falls back to None if the data is unavailable.

    Args:
        factor_root: Path to minute factor results directory.
        factor_name: Which volume activity variant to use.
                     Options: volume_volatility, log_volume_volatility,
                     diff_abs_mean_volume, peak_count_1std, peak_count_2std.

    Returns:
        Wide DataFrame (dates × symbols) of volume activity values, or None.
    """
    import os

    root = Path(
        factor_root
        or os.environ.get("MINUTE_FACTOR_ROOT", "")
        or "artifacts/a_share_minute_factor_top200_202510_202604/factor_results"
    )
    panel_path = root / "mf_volatility_32_full" / f"{factor_name}.parquet"
    if not panel_path.exists():
        return None

    df = pd.read_parquet(panel_path)
    if "date" in df.columns and "ticker" in df.columns:
        value_cols = [c for c in df.columns if c not in ("date", "ticker")]
        if len(value_cols) == 1:
            df = df.pivot(index="date", columns="ticker", values=value_cols[0])
    if df.index.name != "date" and "date" in str(df.index.name or "").lower():
        df.index.name = "date"
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.columns = pd.Index(df.columns).astype(str)
    return df.sort_index()


# ── Factor: Hermite Stability (B-leg) ──────────────────────────────────────────


def compute_hermite_stability_factor(
    volume_activity_panel: pd.DataFrame | None = None,
    factor_root: str | Path | None = None,
    *,
    window: int = 60,
    min_periods: int = 36,
    variant: str = "closeness",
    ddof: int = 0,
) -> pd.DataFrame | None:
    """Compute cross-day Hermite stability from a daily volume-activity panel.

    The index is expected to be trade dates and each cell one stock-day's
    aggregated volume-activity value. Rolling moments therefore describe the
    cross-day time series, not the distribution of minute bars within one day:
    - ``closeness``: -log(1 + h3² + h4²) — HIGHER = more stable over time
    - ``compression``: log(1+E_long) - log(1+E_short) — POSITIVE = getting less stable

    B-leg prefers HIGH closeness (stable behavior).

    Args:
        volume_activity_panel: Pre-computed volume activity wide DataFrame.
        factor_root: Path to minute factor results (used if panel is None).
        window: Rolling window for Hermite transform.
        min_periods: Minimum observations required.
        variant: "closeness" or "compression".
        ddof: Rolling standard-deviation convention. StyleReplica retains its
            historical population default (0); DailyWatch20 currently uses 1.

    Returns:
        Wide DataFrame of Hermite stability values, or None.
    """
    panel = volume_activity_panel
    if panel is None:
        panel = compute_volume_activity_factor(factor_root)
    if panel is None or panel.empty:
        return None

    # Rolling z-score
    roll_mean = panel.rolling(window, min_periods=min_periods).mean()
    if ddof not in {0, 1}:
        raise ValueError("ddof must be 0 or 1")
    roll_std = panel.rolling(window, min_periods=min_periods).std(ddof=ddof)
    roll_std = roll_std.where(roll_std > 1e-8, np.nan)
    z = ((panel - roll_mean) / roll_std).clip(-8.0, 8.0)

    # Hermite polynomials h3 (skewness proxy) and h4 (kurtosis proxy)
    z2 = z * z
    h3_raw = (z * z2 - 3.0 * z) / np.sqrt(6.0)
    h4_raw = (z2 * z2 - 6.0 * z2 + 3.0) / np.sqrt(24.0)
    h3 = h3_raw.rolling(window, min_periods=min_periods).mean()
    h4 = h4_raw.rolling(window, min_periods=min_periods).mean()
    energy = h3.pow(2) + h4.pow(2)

    if variant == "compression":
        # Compute short-window energy too
        w_short = max(20, window // 3)
        mp_short = max(12, min_periods // 3)
        rs_mean = panel.rolling(w_short, min_periods=mp_short).mean()
        rs_std = panel.rolling(w_short, min_periods=mp_short).std(ddof=ddof)
        rs_std = rs_std.where(rs_std > 1e-8, np.nan)
        zs = ((panel - rs_mean) / rs_std).clip(-8.0, 8.0)
        zs2 = zs * zs
        hs3 = ((zs * zs2 - 3.0 * zs) / np.sqrt(6.0)).rolling(w_short, min_periods=mp_short).mean()
        hs4_raw2 = (zs2 * zs2 - 6.0 * zs2 + 3.0) / np.sqrt(24.0)
        hs4 = hs4_raw2.rolling(w_short, min_periods=mp_short).mean()
        energy_short = hs3.pow(2) + hs4.pow(2)
        return np.log1p(energy) - np.log1p(energy_short)  # positive = getting less stable

    # Default: closeness = -log(1 + energy)
    return -np.log1p(energy)  # higher = more stable across daily observations


# ── Composite factorization ────────────────────────────────────────────────────


def compute_all_style_factors(
    price_panel: pd.DataFrame,
    *,
    turnover_panel: pd.DataFrame | None = None,
    market_cap_panel: pd.DataFrame | None = None,
    market_returns: pd.Series | None = None,
    industry_map: pd.Series | None = None,
    minute_factor_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute all style factors needed for StyleReplica scoring.

    Args:
        price_panel: Wide DataFrame (dates × symbols) of adjusted close prices.
        turnover_panel: Optional wide DataFrame of daily turnover rates (%).
        market_cap_panel: Optional wide DataFrame of market capitalizations.
        market_returns: Optional market return series.
        industry_map: Optional symbol → industry label mapping.

    Returns:
        Dictionary mapping factor names to wide DataFrames:
        - ``resvol``
        - ``beta``
        - ``size``
        - ``liquidity``
        - ``mom20``
        - ``mom120``
        - ``industry_mom``
        - ``vol_convergence``
    """
    returns = compute_returns_from_prices(price_panel)

    factors: dict[str, pd.DataFrame] = {}

    # RESVOL
    factors["resvol"] = compute_resvol_factor(returns, market_returns=market_returns)

    # Beta
    factors["beta"] = compute_beta_factor(returns, market_returns=market_returns)

    # Size
    if market_cap_panel is not None and not market_cap_panel.empty:
        factors["size"] = compute_size_factor(market_cap_panel)
    else:
        factors["size"] = pd.DataFrame(0.0, index=price_panel.index, columns=price_panel.columns)

    # Liquidity
    if turnover_panel is not None and not turnover_panel.empty:
        factors["liquidity"] = compute_liquidity_factor(turnover_panel)
    else:
        empty_liq = pd.DataFrame(0.0, index=price_panel.index, columns=price_panel.columns)
        factors["liquidity"] = empty_liq

    # Momentum
    factors["mom20"] = compute_momentum_factor(price_panel, window=20)
    factors["mom120"] = compute_momentum_factor(price_panel, window=120)

    # Industry momentum
    if industry_map is not None and not industry_map.empty:
        factors["industry_mom"] = compute_industry_momentum(price_panel, industry_map)
    else:
        factors["industry_mom"] = pd.DataFrame(
            0.0, index=price_panel.index, columns=price_panel.columns
        )

    # Vol convergence (for B-leg)
    factors["vol_convergence"] = compute_vol_convergence_factor(returns)

    # Minute-level factors (optional — require pre-computed panels)
    if minute_factor_root:
        va = compute_volume_activity_factor(minute_factor_root)
        if va is not None:
            factors["volume_activity"] = va
            hs = compute_hermite_stability_factor(va)
            if hs is not None:
                factors["hermite_stability"] = hs

    return factors
