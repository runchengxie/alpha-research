"""Signal-ranking churn diagnostics.

These helpers measure how the top-ranked membership set changes through time.
They deliberately do not model portfolio buffers, weights, tradability or
transaction costs; those semantics belong to ``portfolio-backtester``.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from .symbols import canonicalize_symbol_columns


def estimate_topk_membership_churn(
    data: pd.DataFrame,
    score_col: str,
    k: int,
    evaluation_dates: list[pd.Timestamp],
    *,
    rank_offset: int = 0,
) -> pd.Series:
    """Return the fraction of Top-K membership replaced at each evaluation date."""

    if data is None or data.empty or k <= 0:
        return pd.Series(dtype=float, name="topk_membership_churn")
    if score_col not in data.columns:
        raise ValueError(f"Missing score column: {score_col}")
    if rank_offset < 0:
        raise ValueError("rank_offset must be >= 0")

    work = canonicalize_symbol_columns(data, context="Signal churn data")
    groups = {
        pd.Timestamp(pd.to_datetime(cast(Any, date))): group
        for date, group in work.groupby("trade_date")
    }
    previous: set[str] | None = None
    records: list[tuple[pd.Timestamp, float]] = []
    for raw_date in evaluation_dates:
        date = pd.to_datetime(cast(Any, raw_date))
        day = groups.get(date)
        if day is None or day.empty:
            continue
        ranked = (
            day.dropna(subset=[score_col])
            .sort_values(score_col, ascending=False)["symbol"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        selected = set(ranked[rank_offset : rank_offset + k])
        if len(selected) < k:
            continue
        if previous is not None:
            overlap = len(selected & previous)
            records.append((date, 1.0 - overlap / k))
        previous = selected

    if not records:
        return pd.Series(dtype=float, name="topk_membership_churn")
    return pd.Series(
        [value for _, value in records],
        index=pd.Index([date for date, _ in records], name="trade_date"),
        name="topk_membership_churn",
    )


__all__ = ["estimate_topk_membership_churn"]
