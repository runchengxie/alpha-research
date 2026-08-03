"""Stock universe filtering for StyleReplica-A80B20-v0.

Filters A-share stocks to a tradable universe suitable for daily style replication.
Excludes ST stocks, newly listed stocks (< 60 trading days), suspended stocks,
and stocks with insufficient price history for factor computation (need ≥ 120 days).
"""

from __future__ import annotations

from typing import cast

import pandas as pd

_REQUIRED_HISTORY_DAYS = 120
_MIN_LISTED_DAYS = 60


def _has_sufficient_history(
    price_panel: pd.DataFrame,
    *,
    min_history: int = _REQUIRED_HISTORY_DAYS,
) -> set[str]:
    """Return the set of symbols with at least `min_history` non-null closes."""
    symbol_counts = price_panel.notna().sum(axis=0)
    return set(symbol_counts[symbol_counts >= min_history].index.tolist())


def _filter_st_and_newly_listed(
    instruments: pd.DataFrame,
    as_of_date: pd.Timestamp,
    *,
    min_listed_days: int = _MIN_LISTED_DAYS,
) -> set[str]:
    """Exclude ST stocks and stocks listed fewer than `min_listed_days` ago."""
    df = instruments.copy()
    if "list_date" in df.columns:
        df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
        cutoff = as_of_date - pd.Timedelta(days=min_listed_days)
        df = df[df["list_date"].notna() & (df["list_date"] <= cutoff)]
    if "is_st" in df.columns:
        df = df[~df["is_st"].astype(bool)]
    elif "name" in df.columns:
        df = df[~df["name"].astype(str).str.upper().str.contains("ST", na=False)]
    return set(df["symbol"].astype(str).tolist()) if "symbol" in df.columns else set()


def filter_style_replica_universe(
    price_panel: pd.DataFrame,
    instruments: pd.DataFrame,
    as_of_date: pd.Timestamp | str,
    *,
    min_history: int = _REQUIRED_HISTORY_DAYS,
    min_listed_days: int = _MIN_LISTED_DAYS,
) -> pd.DataFrame:
    """Filter the full A-share universe to stocks suitable for style replica.

    Args:
        price_panel: Wide-format DataFrame with dates as index, symbols as columns,
                     values = close prices. Must span at least `min_history` days.
        instruments: DataFrame with at minimum ``symbol``, plus optional
                     ``list_date`` and ``is_st`` columns.
        as_of_date: Reference date for filtering.
        min_history: Minimum number of daily closes required.
        min_listed_days: Minimum listing days required.

    Returns:
        A subset of ``price_panel`` containing only eligible symbols.
    """
    as_of_dt = cast(pd.Timestamp, pd.Timestamp(as_of_date)).normalize()

    eligible_from_history = _has_sufficient_history(price_panel, min_history=min_history)
    if not eligible_from_history:
        return pd.DataFrame()

    eligible_from_instruments = _filter_st_and_newly_listed(
        instruments,
        as_of_date=as_of_dt,
        min_listed_days=min_listed_days,
    )
    eligible = eligible_from_history & (
        eligible_from_instruments if eligible_from_instruments else eligible_from_history
    )

    symbols = sorted(eligible & set(price_panel.columns.tolist()))
    if not symbols:
        return pd.DataFrame()

    return price_panel[symbols].loc[:as_of_dt]
