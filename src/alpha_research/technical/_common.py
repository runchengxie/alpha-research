from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd


def require_columns(frame: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {', '.join(missing)}")


def validate_window(window: int, name: str) -> None:
    if isinstance(window, bool) or window <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_min_periods(min_periods: int | None, window: int) -> int:
    value = window if min_periods is None else min_periods
    if isinstance(value, bool) or value <= 0 or value > window:
        raise ValueError("min_periods must be between 1 and window")
    return value


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def grouped_transform(
    values: pd.Series,
    groups: pd.Series | None,
    function: Callable[[pd.Series], pd.Series],
) -> pd.Series:
    if groups is None:
        return function(values)
    return values.groupby(groups, sort=False).transform(function)
