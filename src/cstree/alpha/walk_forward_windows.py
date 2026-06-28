from __future__ import annotations

import numpy as np


def build_walk_forward_windows(
    all_dates: np.ndarray,
    test_size: float,
    n_windows: int,
    step_size: float | None,
    gap_days: int,
    anchor_end: bool,
) -> list[dict]:
    n_dates = len(all_dates)
    if n_dates == 0:
        return []
    if test_size <= 0:
        return []
    test_len = int(test_size) if test_size >= 1 else int(n_dates * test_size)
    test_len = max(1, test_len)
    step = step_size
    if step is None:
        step = test_len
    elif 0 < float(step) < 1:
        step = int(n_dates * float(step))
    step = max(1, int(step))

    if anchor_end:
        first_test_start = n_dates - test_len - step * (n_windows - 1)
    else:
        first_test_start = int(n_dates * (1 - test_size))
    windows = []
    for idx in range(n_windows):
        test_start = first_test_start + idx * step
        test_end = test_start + test_len
        if test_start < 0 or test_end > n_dates:
            continue
        train_end = max(0, test_start - gap_days)
        train_dates = all_dates[:train_end]
        test_dates = all_dates[test_start:test_end]
        if len(train_dates) == 0 or len(test_dates) == 0:
            continue
        windows.append(
            {
                "window": idx + 1,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "train_dates": train_dates,
                "test_dates": test_dates,
            }
        )
    return windows
