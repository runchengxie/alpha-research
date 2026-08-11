from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from .split import select_train_window_dates

logger = logging.getLogger("alpha_research")


def build_trade_date_slices(
    frame: pd.DataFrame,
    *,
    date_col: str = "trade_date",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[pd.Timestamp, int]]:
    ordered = frame.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    date_values = ordered[date_col].to_numpy()
    if date_values.size == 0:
        empty = np.array([], dtype=int)
        return ordered, np.array([], dtype="datetime64[ns]"), empty, empty, {}

    dates, start_rows = np.unique(date_values, return_index=True)
    end_rows = np.empty_like(start_rows)
    if len(start_rows) > 1:
        end_rows[:-1] = start_rows[1:]
    end_rows[-1] = len(ordered)
    date_to_pos = {pd.to_datetime(date): idx for idx, date in enumerate(dates)}
    return ordered, dates, start_rows, end_rows, date_to_pos


def slice_trade_date_range(
    ordered: pd.DataFrame,
    start_rows: np.ndarray,
    end_rows: np.ndarray,
    start_pos: int,
    end_pos: int,
) -> pd.DataFrame:
    if start_pos < 0 or end_pos < start_pos:
        return ordered.iloc[0:0].copy()
    return ordered.iloc[start_rows[start_pos] : end_rows[end_pos]].copy()


def slice_trade_dates(
    ordered: pd.DataFrame,
    start_rows: np.ndarray,
    end_rows: np.ndarray,
    date_to_pos: dict[pd.Timestamp, int],
    dates: np.ndarray | list[pd.Timestamp] | tuple[pd.Timestamp, ...],
    *,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    del date_col
    if len(dates) == 0:
        return ordered.iloc[0:0].copy()
    unique_dates = []
    seen_dates: set[pd.Timestamp] = set()
    for date in pd.to_datetime(dates):
        date_ts = cast(pd.Timestamp, pd.Timestamp(date))
        if date_ts in seen_dates:
            continue
        seen_dates.add(date_ts)
        unique_dates.append(date_ts)
    if not unique_dates:
        return ordered.iloc[0:0].copy()

    first = unique_dates[0]
    last = unique_dates[-1]
    start_pos = date_to_pos.get(first)
    end_pos = date_to_pos.get(last)
    if start_pos is not None and end_pos is not None:
        expected_len = end_pos - start_pos + 1
        if expected_len == len(unique_dates):
            return slice_trade_date_range(ordered, start_rows, end_rows, start_pos, end_pos)

    positions = sorted({date_to_pos[date] for date in unique_dates if date in date_to_pos})
    if not positions:
        return ordered.iloc[0:0].copy()

    ranges: list[tuple[int, int]] = []
    range_start = positions[0]
    range_end = positions[0]
    for pos in positions[1:]:
        if pos == range_end + 1:
            range_end = pos
            continue
        ranges.append((range_start, range_end))
        range_start = pos
        range_end = pos
    ranges.append((range_start, range_end))

    parts = [ordered.iloc[start_rows[start] : end_rows[end]] for start, end in ranges]
    if len(parts) == 1:
        return parts[0].copy()
    return pd.concat(parts, ignore_index=True)


def apply_model_train_window(
    train_dates_input: np.ndarray | list[pd.Timestamp] | tuple[pd.Timestamp, ...],
    *,
    label: str,
    train_window_mode: str,
    train_window_size: int | None,
    train_window_unit: str,
) -> np.ndarray:
    train_dates_array = np.asarray(pd.to_datetime(train_dates_input), dtype="datetime64[ns]")
    if train_dates_array.size == 0:
        return train_dates_array
    windowed_dates = select_train_window_dates(
        train_dates_array,
        mode=train_window_mode,
        size=train_window_size,
        unit=train_window_unit,
    )
    if (
        train_window_mode == "rolling"
        and windowed_dates.size > 0
        and windowed_dates.size < train_dates_array.size
    ):
        logger.info(
            "Applied model.train_window to %s: using %s/%s dates (%s -> %s).",
            label,
            len(windowed_dates),
            len(train_dates_array),
            pd.to_datetime(windowed_dates[0]).strftime("%Y-%m-%d"),
            pd.to_datetime(windowed_dates[-1]).strftime("%Y-%m-%d"),
        )
    return windowed_dates


def slice_with_train_window(
    ordered: pd.DataFrame,
    start_rows: np.ndarray,
    end_rows: np.ndarray,
    date_to_pos: dict[pd.Timestamp, int],
    train_dates_input: np.ndarray | list[pd.Timestamp] | tuple[pd.Timestamp, ...],
    *,
    label: str,
    train_window_mode: str,
    train_window_size: int | None,
    train_window_unit: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    windowed_dates = apply_model_train_window(
        train_dates_input,
        label=label,
        train_window_mode=train_window_mode,
        train_window_size=train_window_size,
        train_window_unit=train_window_unit,
    )
    if windowed_dates.size == 0:
        return ordered.iloc[0:0].copy(), windowed_dates
    frame = slice_trade_dates(
        ordered,
        start_rows,
        end_rows,
        date_to_pos,
        windowed_dates,
    )
    return frame, windowed_dates


_build_trade_date_slices = build_trade_date_slices
_slice_trade_date_range = slice_trade_date_range
_slice_trade_dates = slice_trade_dates
_apply_model_train_window = apply_model_train_window
_slice_with_train_window = slice_with_train_window


__all__ = [
    "_apply_model_train_window",
    "_build_trade_date_slices",
    "_slice_trade_date_range",
    "_slice_trade_dates",
    "_slice_with_train_window",
    "apply_model_train_window",
    "build_trade_date_slices",
    "slice_trade_date_range",
    "slice_trade_dates",
    "slice_with_train_window",
]
