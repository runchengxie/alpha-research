from __future__ import annotations

import pandas as pd

from ._common import (
    grouped_transform,
    numeric_column,
    require_columns,
    validate_min_periods,
    validate_window,
)


def rolling_high(
    frame: pd.DataFrame,
    *,
    window: int = 20,
    high_col: str = "high",
    group_col: str | None = None,
    exclude_current: bool = True,
    min_periods: int | None = None,
    output_name: str = "rolling_high",
) -> pd.Series:
    """Calculate a rolling high, optionally shifted to exclude the current bar."""

    validate_window(window, "window")
    minimum = validate_min_periods(min_periods, window)
    required = [high_col]
    if group_col is not None:
        required.append(group_col)
    require_columns(frame, required, "price frame")
    highs = numeric_column(frame, high_col)

    def rolling_max(values: pd.Series) -> pd.Series:
        result = values.rolling(window, min_periods=minimum).max()
        return result.shift(1) if exclude_current else result

    result = grouped_transform(highs, frame[group_col] if group_col else None, rolling_max)
    result.name = output_name
    return result


__all__ = ["rolling_high"]
