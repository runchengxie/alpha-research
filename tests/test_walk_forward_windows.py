from __future__ import annotations

import numpy as np
import pandas as pd

from cstree.alpha.walk_forward_windows import build_walk_forward_windows


def test_walk_forward_windows_anchor_end_with_gap() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="B").to_numpy()

    windows = build_walk_forward_windows(
        dates,
        test_size=2,
        n_windows=2,
        step_size=2,
        gap_days=1,
        anchor_end=True,
    )

    assert [window["window"] for window in windows] == [1, 2]
    assert np.array_equal(windows[0]["train_dates"], dates[:5])
    assert np.array_equal(windows[0]["test_dates"], dates[6:8])
    assert np.array_equal(windows[1]["train_dates"], dates[:7])
    assert np.array_equal(windows[1]["test_dates"], dates[8:10])
