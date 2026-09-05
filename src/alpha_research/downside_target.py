"""Fixed exchange-session downside-risk labels after next-close entry."""

import numpy as np
import pandas as pd


def trailing_downside_rms(prices, decision_date, horizon, calendar):
    """Observed downside RMS through decision close; unavailable gaps stay NaN."""
    dates = pd.DatetimeIndex(calendar)
    if (
        not isinstance(horizon, int)
        or horizon < 1
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
    ):
        raise ValueError("positive integer horizon and ordered unique calendar required")
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique:
        raise ValueError("prices require unique datetime keys")
    end = int(np.searchsorted(dates.to_numpy(), np.datetime64(decision_date), side="right")) - 1
    start = end - horizon
    if start < 0:
        return np.nan
    values = prices.reindex(dates[start : end + 1]).to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        return np.nan
    return float(np.sqrt(np.square(np.minimum(values[1:] / values[:-1] - 1, 0)).mean()))


def next_close_downside_target(prices, decision_date, horizon, calendar):
    """Return daily downside RMS and maturity date, or NaN when unavailable.

    Requires horizon+1 observed positive closing marks starting at next close.
    Positive-return sessions remain in the RMS denominator. Missing prices do
    not extend the horizon or imply a zero return. No annualization is applied.
    """
    dates = pd.DatetimeIndex(calendar)
    if (
        not isinstance(horizon, int)
        or horizon < 1
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
    ):
        raise ValueError("positive integer horizon and ordered unique calendar required")
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique:
        raise ValueError("prices require unique datetime keys")
    entry = int(np.searchsorted(dates.to_numpy(), np.datetime64(decision_date), side="right"))
    stop = entry + horizon
    if stop >= len(dates):
        return np.nan, pd.NaT
    end = dates[stop]
    window = prices.reindex(dates[entry : stop + 1]).to_numpy(dtype=float)
    if not np.isfinite(window).all() or (window <= 0).any():
        return np.nan, end
    returns = window[1:] / window[:-1] - 1
    return float(np.sqrt(np.square(np.minimum(returns, 0)).mean())), end
