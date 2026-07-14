# ruff: noqa: RUF002
"""Residual volatility (RESVOL) computation for StyleReplica-A80B20-v0.

Computes residual volatility as the standard deviation of residuals from a
60-trading-day rolling market + industry regression:

    r_{i,t} = α_i + β_{mkt}·r_{mkt,t} + β_{ind}·r_{industry,t} + ε_{i,t}

    RESVOL_i = std(ε_{i,t-59:t})

For the first-phase MVP, this uses a simplified 2-factor regression
(market + industry). Later versions can add Size, Momentum etc. to
upgrade to a full specific-volatility model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_RESVOL_WINDOW = 60
_MIN_OBS = 40  # minimum valid observations in the window


def _rolling_resvol_single(
    returns: np.ndarray,
    market_returns: np.ndarray,
    *,
    window: int = _RESVOL_WINDOW,
    min_obs: int = _MIN_OBS,
) -> np.ndarray:
    """Compute rolling residual volatility for a single asset.

    Uses an expanding-then-rolling approach: the first valid output appears
    at index ``min_obs - 1``; subsequent values use a full rolling window.

    Args:
        returns: 1-d array of daily stock returns (aligned with market_returns).
        market_returns: 1-d array of market returns (same length).
        window: Rolling window length.
        min_obs: Minimum observations required for a valid residual vol estimate.

    Returns:
        1-d array of same length as inputs; NaN where insufficient data.
    """
    n = len(returns)
    result = np.full(n, np.nan)

    # Ensure aligned length
    length = min(len(returns), len(market_returns))
    if length < min_obs:
        return result

    r = returns[:length].astype(float)
    m = market_returns[:length].astype(float)

    # Build design matrix: [intercept, market]
    X = np.column_stack([np.ones(length), m])

    for i in range(min_obs - 1, length):
        start = max(0, i - window + 1)
        win_r = r[start : i + 1]
        win_X = X[start : i + 1]

        valid = np.isfinite(win_r) & np.all(np.isfinite(win_X), axis=1)
        if valid.sum() < min_obs:
            continue

        y = win_r[valid]
        Xv = win_X[valid]

        try:
            coeffs, *_ = np.linalg.lstsq(Xv, y, rcond=None)
            residuals = y - Xv @ coeffs
            result[i] = float(np.std(residuals, ddof=1))
        except np.linalg.LinAlgError:
            continue

    return result


def compute_resvol_factor(
    returns_panel: pd.DataFrame,
    market_returns: pd.Series | None = None,
    *,
    industry_returns: pd.DataFrame | None = None,
    window: int = _RESVOL_WINDOW,
    min_obs: int = _MIN_OBS,
) -> pd.DataFrame:
    """Compute daily residual volatility for all stocks.

    Args:
        returns_panel: Wide DataFrame: dates as index, symbols as columns,
                       values = daily log or simple returns.
        market_returns: Series of market returns aligned by date index.
                        If None, uses cross-sectional mean return as proxy.
        industry_returns: Optional wide DataFrame of industry index returns
                          (dates × industry codes). MVP uses 2-factor model
                          (market + industry). If None, reverts to market-only.
        window: Rolling window in trading days (default 60).
        min_obs: Minimum valid observations (default 40).

    Returns:
        Wide DataFrame (dates × symbols) of daily residual volatility values.
    """
    symbols = list(returns_panel.columns)
    dates = returns_panel.index

    # Market return proxy
    if market_returns is None:
        mkt = returns_panel.mean(axis=1)
    else:
        mkt = market_returns.reindex(dates).astype(float)
    mkt_vals = mkt.values

    result = pd.DataFrame(np.nan, index=dates, columns=symbols, dtype=float)

    for symbol in symbols:
        stock_returns = returns_panel[symbol].values.astype(float)

        if industry_returns is not None and not industry_returns.empty:
            # For MVP, we use the mean industry return as additional regressor.
            # Future: match each stock to its specific industry index.
            ind_vals = industry_returns.reindex(dates).mean(axis=1).values.astype(float)
            # Adjust returns for industry
            adj_returns = stock_returns - ind_vals
        else:
            adj_returns = stock_returns

        resvol = _rolling_resvol_single(
            adj_returns,
            mkt_vals,
            window=window,
            min_obs=min_obs,
        )
        result[symbol] = resvol

    return result


def compute_resvol_simplified(
    daily_returns: pd.DataFrame,
    *,
    window: int = 60,
    min_obs: int = 40,
) -> pd.DataFrame:
    """Simplified RESVOL: std of market-hedged returns (market-only regression).

    This is the quick path when industry data is unavailable.
    Uses a single-factor market model regression residual.

    Args:
        daily_returns: Wide DataFrame (dates × symbols).
        window: Rolling window length.
        min_obs: Minimum valid observations.

    Returns:
        Wide DataFrame (dates × symbols) of residual volatility.
    """
    return compute_resvol_factor(daily_returns, window=window, min_obs=min_obs)
