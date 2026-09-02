"""Symbol-column normalization used by the public alpha-research package."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

SYMBOL_COL = "symbol"
LEGACY_SYMBOL_COLUMNS = ("ts_code", "stock_ticker")
DEFAULT_SYMBOL_PRIORITY = ("symbol", "ts_code", "stock_ticker", "order_book_id")


def _column_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame.loc[:, column]
    return values.iloc[:, 0] if isinstance(values, pd.DataFrame) else values


def _clean_symbol_series(values: pd.Series) -> pd.Series:
    return values.where(values.notna(), "").astype(str).str.strip()


def resolve_symbol_series(
    frame: pd.DataFrame,
    *,
    context: str,
    priority: Sequence[str] = DEFAULT_SYMBOL_PRIORITY,
) -> pd.Series:
    present = [column for column in priority if column in frame.columns]
    if not present:
        raise SystemExit(f"{context} is missing symbol/stock_ticker/ts_code/order_book_id.")
    merged = _clean_symbol_series(_column_series(frame, present[0]))
    for column in present[1:]:
        values = _clean_symbol_series(_column_series(frame, column))
        merged = merged.where(merged != "", values)
    return merged


def canonicalize_symbol_columns(
    frame: pd.DataFrame,
    *,
    context: str,
    priority: Sequence[str] = DEFAULT_SYMBOL_PRIORITY,
    drop_order_book_id: bool = False,
) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[SYMBOL_COL] = resolve_symbol_series(normalized, context=context, priority=priority)
    drop_columns = [*LEGACY_SYMBOL_COLUMNS]
    if drop_order_book_id:
        drop_columns.append("order_book_id")
    result = normalized.drop(columns=drop_columns, errors="ignore")
    result.attrs = dict(getattr(frame, "attrs", {}))
    return result
