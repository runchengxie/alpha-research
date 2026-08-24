from __future__ import annotations

import pandas as pd

from ._common import (
    grouped_transform,
    numeric_column,
    require_columns,
    validate_min_periods,
    validate_window,
)


def true_range(
    frame: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    group_col: str | None = None,
    output_name: str = "true_range",
) -> pd.Series:
    """Calculate point-in-time true range without crossing symbol boundaries."""

    required = [high_col, low_col, close_col]
    if group_col is not None:
        required.append(group_col)
    require_columns(frame, required, "price frame")

    high = numeric_column(frame, high_col)
    low = numeric_column(frame, low_col)
    close = numeric_column(frame, close_col)
    if group_col is None:
        previous_close = close.shift(1)
    else:
        previous_close = close.groupby(frame[group_col], sort=False).shift(1)

    components = pd.DataFrame(
        {
            "range": high - low,
            "high_gap": (high - previous_close).abs(),
            "low_gap": (low - previous_close).abs(),
        },
        index=frame.index,
    )
    result = components.max(axis=1, skipna=True)
    result.name = output_name
    return result


def average_true_range(
    frame: pd.DataFrame,
    *,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    group_col: str | None = None,
    min_periods: int | None = None,
    output_name: str = "atr",
) -> pd.Series:
    """Calculate a simple-moving-average ATR with point-in-time windows."""

    validate_window(window, "window")
    minimum = validate_min_periods(min_periods, window)
    ranges = true_range(
        frame,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        group_col=group_col,
    )

    def rolling_mean(values: pd.Series) -> pd.Series:
        return values.rolling(window, min_periods=minimum).mean()

    result = grouped_transform(ranges, frame[group_col] if group_col else None, rolling_mean)
    result.name = output_name
    return result


__all__ = ["average_true_range", "true_range"]
