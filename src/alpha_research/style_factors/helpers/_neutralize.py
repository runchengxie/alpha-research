"""PIT SW-L1 industry merge helper for style_factors neutralization.

Attaches ``industry_l1`` to each (symbol, trade_date) via point-in-time interval
matching against the SW membership long table.  Kept in its own module so the
original ``_add_new_factors`` complexity hotspot is eliminated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def merge_sw_industry_pit(
    df: pd.DataFrame,
    sw_membership: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach ``industry_l1`` to each (symbol, trade_date) via PIT interval match.

    ``sw_membership`` is the long table from ``load_sw_industry_membership``:
    ``[symbol, in_date, out_date, industry_l1]``.  A stock on a trade_date gets
    the L1 industry whose ``in_date <= trade_date <= out_date`` (open end when
    out_date is NaT).  Stocks/dates with no match get NaN and are neutralized as
    a single residual group.
    """
    df = df.copy()
    df["industry_l1"] = np.nan
    if sw_membership is None or sw_membership.empty:
        return df

    mem = sw_membership.copy()
    mem["in_date"] = pd.to_datetime(mem["in_date"], errors="coerce")
    mem["out_date"] = pd.to_datetime(mem["out_date"], errors="coerce")

    # Per symbol, interval-match each trade_date.  Loop over symbols is fine:
    # membership rows per symbol are few.
    by_symbol = {sym: g.sort_values("in_date") for sym, g in mem.groupby("symbol", sort=False)}
    industry = np.full(len(df), np.nan, dtype=object)
    df_sym = df["symbol"].to_numpy()
    df_td = df["trade_date"].to_numpy()
    for i in range(len(df)):
        g = by_symbol.get(df_sym[i])
        if g is None or g.empty:
            continue
        td = df_td[i]
        start = g["in_date"].to_numpy()
        end = g["out_date"].to_numpy()
        end_mask = np.array([pd.isna(e) or td <= e for e in end])
        hit = (td >= start) & end_mask
        if hit.any():
            industry[i] = g["industry_l1"].to_numpy()[hit][-1]
    df["industry_l1"] = industry
    return df
