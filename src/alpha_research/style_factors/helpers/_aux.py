"""Exact-date auxiliary stock-day panel merge for style_factors."""

from __future__ import annotations

import pandas as pd


def _merge_aux(
    df: pd.DataFrame,
    aux: pd.DataFrame | None,
    columns: list[str],
) -> pd.DataFrame:
    """Merge a daily auxiliary panel without carrying observations forward.

    Money-flow and event rows are date-specific.  Forward-filling them changes
    the signal meaning and can make a one-day event persist indefinitely.
    Holder and daily-basic feature assets are already materialized by trade date,
    so their missing dates should also remain missing and visible to coverage QA.
    """
    if aux is None or aux.empty:
        for column in columns:
            df[column] = pd.NA
        return df
    aux = aux.copy()
    aux["trade_date"] = pd.to_datetime(aux["trade_date"])
    aux = aux[
        aux["trade_date"].isin(df["trade_date"].unique())
        & aux["symbol"].isin(df["symbol"].unique())
    ].copy()
    available = [column for column in columns if column in aux.columns]
    source = aux[["symbol", "trade_date", *available]].drop_duplicates(
        ["symbol", "trade_date"], keep="last"
    )
    merged = df.merge(source, on=["symbol", "trade_date"], how="left", sort=False)
    for column in columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged
