"""CPCV group assignment and split construction (private helpers).

Re-exported from ``alpha_research.cpcv`` so existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any, cast

import pandas as pd

from ._cpcv_dates import LabelEventWindow, _as_date_tuple, _date_key, _lookup_shifted_date


@dataclass(frozen=True)
class CPCVSplit:
    split_id: int
    test_groups: tuple[int, ...]
    train_groups: tuple[int, ...]
    train_dates_raw: tuple[pd.Timestamp, ...]
    train_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]
    purged_train_dates: tuple[pd.Timestamp, ...]
    embargoed_train_dates: tuple[pd.Timestamp, ...]
    purge_mode: str
    status: str = "ok"


def assign_cpcv_groups(dates: Any, n_groups: int) -> dict[int, tuple[pd.Timestamp, ...]]:
    date_values = _as_date_tuple(dates)
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2.")
    if len(date_values) < n_groups:
        raise ValueError("n_groups cannot exceed the number of eligible dates.")

    groups: dict[int, tuple[pd.Timestamp, ...]] = {}
    base_size, extra = divmod(len(date_values), n_groups)
    cursor = 0
    for group_id in range(n_groups):
        size = base_size + (1 if group_id < extra else 0)
        groups[group_id] = date_values[cursor : cursor + size]
        cursor += size
    return groups


def expected_cpcv_path_count(n_groups: int, test_groups: int) -> int:
    if test_groups < 1 or test_groups >= n_groups:
        raise ValueError("test_groups must satisfy 1 <= test_groups < n_groups.")
    return comb(n_groups - 1, test_groups - 1)


def build_label_event_windows(
    signal_dates: Any,
    *,
    all_trade_dates: Any,
    horizon_mode: str,
    horizon_days: int,
    shift_days: int,
    next_rebalance_map: dict[Any, Any] | None = None,
) -> tuple[dict[pd.Timestamp, LabelEventWindow], str]:
    dates = _as_date_tuple(signal_dates)
    trade_dates = _as_date_tuple(all_trade_dates)
    if not dates or not trade_dates:
        return {}, "fallback_gap"

    mode = str(horizon_mode or "fixed").strip().lower()
    next_map = {
        _date_key(key): _date_key(value)
        for key, value in (next_rebalance_map or {}).items()
        if not pd.isna(pd.to_datetime(key, errors="coerce"))
        and not pd.isna(pd.to_datetime(value, errors="coerce"))
    }
    windows: dict[pd.Timestamp, LabelEventWindow] = {}
    for signal_date in dates:
        label_start = _lookup_shifted_date(signal_date, trade_dates, shift_days)
        if label_start is None:
            continue
        if mode == "next_rebalance":
            exit_signal = next_map.get(signal_date)
            if exit_signal is None:
                continue
            label_end = _lookup_shifted_date(exit_signal, trade_dates, shift_days)
        else:
            label_end = _lookup_shifted_date(
                signal_date, trade_dates, int(horizon_days) + int(shift_days)
            )
        if label_end is None:
            continue
        if label_end < label_start:
            label_start, label_end = label_end, label_start
        windows[signal_date] = LabelEventWindow(
            signal_date=signal_date,
            label_start=label_start,
            label_end=label_end,
        )
    purge_mode = "event_window" if len(windows) == len(dates) else "fallback_gap"
    return windows, purge_mode


def _intervals_overlap(left: LabelEventWindow, right: LabelEventWindow) -> bool:
    return left.label_start <= right.label_end and right.label_start <= left.label_end


def _apply_event_purge(
    train_dates: tuple[pd.Timestamp, ...],
    test_dates: tuple[pd.Timestamp, ...],
    event_windows: dict[pd.Timestamp, LabelEventWindow],
    *,
    embargo_days: int,
) -> tuple[tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...], str]:
    test_windows = [event_windows[date] for date in test_dates if date in event_windows]
    if len(test_windows) != len(test_dates):
        return train_dates, (), (), "fallback_gap"

    purged: list[pd.Timestamp] = []
    embargoed: list[pd.Timestamp] = []
    kept: list[pd.Timestamp] = []
    embargo_delta = cast(pd.Timedelta, pd.Timedelta(days=max(0, int(embargo_days))))
    for train_date in train_dates:
        train_window = event_windows.get(train_date)
        if train_window is None:
            purged.append(train_date)
            continue
        if any(_intervals_overlap(train_window, test_window) for test_window in test_windows):
            purged.append(train_date)
            continue
        if embargo_delta > pd.Timedelta(0) and any(
            test_window.label_end
            < train_window.signal_date
            <= test_window.label_end + embargo_delta
            for test_window in test_windows
        ):
            embargoed.append(train_date)
            continue
        kept.append(train_date)
    return tuple(kept), tuple(purged), tuple(embargoed), "event_window"


def _apply_gap_purge(
    train_dates: tuple[pd.Timestamp, ...],
    test_dates: tuple[pd.Timestamp, ...],
    all_dates: tuple[pd.Timestamp, ...],
    *,
    gap_steps: int,
) -> tuple[tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...]]:
    if gap_steps <= 0:
        return train_dates, (), ()
    positions = {date: idx for idx, date in enumerate(all_dates)}
    test_positions = [positions[date] for date in test_dates if date in positions]
    if not test_positions:
        return train_dates, (), ()
    min_test = min(test_positions)
    max_test = max(test_positions)
    purged: list[pd.Timestamp] = []
    kept: list[pd.Timestamp] = []
    for train_date in train_dates:
        pos = positions.get(train_date)
        if pos is None:
            purged.append(train_date)
            continue
        if min_test - gap_steps <= pos <= max_test + gap_steps:
            purged.append(train_date)
            continue
        kept.append(train_date)
    return tuple(kept), tuple(purged), ()


def build_cpcv_splits(
    dates: Any,
    *,
    n_groups: int,
    test_groups: int,
    event_windows: dict[pd.Timestamp, LabelEventWindow] | None = None,
    embargo_days: int = 0,
    fallback_gap_steps: int = 0,
    min_train_dates: int = 1,
    min_test_dates: int = 1,
) -> tuple[dict[int, tuple[pd.Timestamp, ...]], list[CPCVSplit]]:
    if test_groups < 1 or test_groups >= n_groups:
        raise ValueError("test_groups must satisfy 1 <= test_groups < n_groups.")
    all_dates = _as_date_tuple(dates)
    groups = assign_cpcv_groups(all_dates, n_groups)
    group_ids = tuple(groups)
    splits: list[CPCVSplit] = []
    for split_id, test_group_tuple in enumerate(combinations(group_ids, test_groups), start=1):
        train_group_tuple = tuple(group for group in group_ids if group not in test_group_tuple)
        test_dates = tuple(date for group in test_group_tuple for date in groups[group])
        train_dates_raw = tuple(date for group in train_group_tuple for date in groups[group])
        if event_windows:
            train_dates, purged, embargoed, purge_mode = _apply_event_purge(
                train_dates_raw,
                test_dates,
                event_windows,
                embargo_days=embargo_days,
            )
            if purge_mode == "fallback_gap":
                train_dates, purged, embargoed = _apply_gap_purge(
                    train_dates_raw,
                    test_dates,
                    all_dates,
                    gap_steps=fallback_gap_steps,
                )
        else:
            train_dates, purged, embargoed = _apply_gap_purge(
                train_dates_raw,
                test_dates,
                all_dates,
                gap_steps=fallback_gap_steps,
            )
            purge_mode = "fallback_gap" if fallback_gap_steps > 0 else "none"
        status = (
            "ok"
            if len(train_dates) >= min_train_dates and len(test_dates) >= min_test_dates
            else "insufficient_data"
        )
        splits.append(
            CPCVSplit(
                split_id=split_id,
                test_groups=tuple(int(group) for group in test_group_tuple),
                train_groups=tuple(int(group) for group in train_group_tuple),
                train_dates_raw=train_dates_raw,
                train_dates=tuple(sorted(train_dates)),
                test_dates=tuple(sorted(test_dates)),
                purged_train_dates=tuple(sorted(purged)),
                embargoed_train_dates=tuple(sorted(embargoed)),
                purge_mode=purge_mode,
                status=status,
            )
        )
    return groups, splits


def build_cpcv_paths(
    valid_splits: list[CPCVSplit],
    *,
    n_groups: int,
    test_groups: int,
) -> list[list[CPCVSplit]]:
    path_count = expected_cpcv_path_count(n_groups, test_groups)
    paths: list[list[CPCVSplit]] = [[] for _ in range(path_count)]
    covered: list[set[int]] = [set() for _ in range(path_count)]
    for split in sorted(valid_splits, key=lambda item: (item.test_groups, item.split_id)):
        for path_idx, group_set in enumerate(covered):
            if all(group not in group_set for group in split.test_groups):
                paths[path_idx].append(split)
                group_set.update(split.test_groups)
                break
    return paths
