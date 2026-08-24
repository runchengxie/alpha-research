from __future__ import annotations

import pandas as pd

from ._common import (
    grouped_transform,
    numeric_column,
    require_columns,
    validate_min_periods,
    validate_window,
)


def volume_weighted_cost(
    frame: pd.DataFrame,
    *,
    window: int,
    price_col: str = "close",
    volume_col: str = "volume",
    amount_col: str | None = None,
    amount_divisor: float = 1.0,
    group_col: str | None = None,
    min_periods: int | None = None,
    output_name: str = "volume_weighted_cost",
) -> pd.Series:
    """Calculate rolling volume-weighted cost for an asset or an index.

    With amount_col, the numerator is the rolling amount sum. Without it, the
    numerator is the rolling sum of price times volume.
    """

    validate_window(window, "window")
    minimum = validate_min_periods(min_periods, window)
    if amount_divisor <= 0:
        raise ValueError("amount_divisor must be positive")

    required = [volume_col]
    if amount_col is None:
        required.append(price_col)
    else:
        required.append(amount_col)
    if group_col is not None:
        required.append(group_col)
    require_columns(frame, required, "price frame")

    volume = numeric_column(frame, volume_col)
    if amount_col is None:
        numerator = numeric_column(frame, price_col) * volume
    else:
        numerator = numeric_column(frame, amount_col)

    def rolling_sum(values: pd.Series) -> pd.Series:
        return values.rolling(window, min_periods=minimum).sum()

    groups = frame[group_col] if group_col else None
    numerator_sum = grouped_transform(numerator, groups, rolling_sum)
    volume_sum = grouped_transform(volume, groups, rolling_sum)
    result = numerator_sum.div(volume_sum).div(amount_divisor)
    result = result.mask(volume_sum <= 0)
    result.name = output_name
    return result


__all__ = ["volume_weighted_cost"]
