import numpy as np
import pandas as pd

from alpha_research.split import (
    _apply_event_window_purge_indices,
    _build_label_event_windows,
    build_sample_weight,
    select_train_window_dates,
    time_series_cv_ic,
)


def test_time_series_cv_gap_skips_all():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "symbol": ["A", "B"] * len(dates),
            "f1": [1.0] * (len(dates) * 2),
            "target": [0.1] * (len(dates) * 2),
        }
    )
    scores = time_series_cv_ic(
        df,
        features=["f1"],
        target_col="target",
        n_splits=3,
        embargo_days=10,
        purge_days=10,
        model_params={"n_estimators": 1, "max_depth": 1, "learning_rate": 0.1},
        signal_direction=1.0,
    )
    assert scores == []


def test_build_sample_weight_date_equal():
    dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])
    df = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["A", "B", "A"],
            "f1": [1.0, 2.0, 3.0],
            "target": [0.1, 0.2, 0.3],
        }
    )
    weights = build_sample_weight(df, "date_equal")
    assert weights is not None
    df = df.assign(weight=weights)
    sums = df.groupby("trade_date")["weight"].sum().values
    assert all(abs(value - 1.0) < 1e-12 for value in sums)


def test_build_sample_weight_exp_decay_increases_recent_dates():
    dates = pd.to_datetime(
        [
            "2020-01-01",
            "2020-01-01",
            "2020-01-02",
            "2020-01-02",
            "2020-01-03",
        ]
    )
    df = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["A", "B", "A", "B", "A"],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    weights = build_sample_weight(
        df,
        "exp_decay",
        params={"halflife": 1.0},
    )
    assert weights is not None
    totals = df.assign(weight=weights).groupby("trade_date")["weight"].sum()
    assert totals.iloc[2] > totals.iloc[1] > totals.iloc[0]
    assert np.isclose(float(totals.mean()), 1.0)


def test_select_train_window_dates_supports_dates_unit():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    selected = select_train_window_dates(dates.to_numpy(), mode="rolling", size=3, unit="dates")
    assert pd.to_datetime(selected).tolist() == list(dates[-3:])


def test_select_train_window_dates_supports_years_unit():
    dates = pd.to_datetime(["2020-03-31", "2021-03-31", "2022-03-31", "2023-03-31", "2024-03-31"])
    selected = select_train_window_dates(dates.to_numpy(), mode="rolling", size=2, unit="years")
    assert pd.to_datetime(selected).tolist() == list(dates[-3:])


def test_event_window_purge_removes_overlapping_train_dates():
    dates = pd.date_range("2020-01-01", periods=8, freq="D").to_numpy()
    all_trade_dates = pd.date_range("2020-01-01", periods=12, freq="D").to_numpy()
    windows, status = _build_label_event_windows(
        dates,
        all_trade_dates=all_trade_dates,
        horizon_mode="fixed",
        horizon_days=3,
        shift_days=0,
    )

    kept, used_event_window = _apply_event_window_purge_indices(
        np.array([0, 1, 2, 3]),
        np.array([4, 5]),
        dates,
        windows,
        embargo_days=0,
    )

    assert status == "event_window"
    assert used_event_window is True
    assert kept.tolist() == [0]


def test_time_series_cv_supports_model_cfg():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    rows = []
    for idx, date in enumerate(dates):
        rows.append({"trade_date": date, "symbol": "A", "f1": 0.0, "target": 0.0 + idx * 0.01})
        rows.append({"trade_date": date, "symbol": "B", "f1": 1.0, "target": 1.0 + idx * 0.01})
    df = pd.DataFrame(rows)

    scores = time_series_cv_ic(
        df,
        features=["f1"],
        target_col="target",
        n_splits=3,
        embargo_days=0,
        purge_days=0,
        model_cfg={"type": "ridge", "params": {"alpha": 1.0}},
        signal_direction=1.0,
    )

    assert len(scores) == 3
    assert all(np.isfinite(scores))


def test_time_series_cv_supports_separate_fit_target_col():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    rows = []
    for date in dates:
        rows.append(
            {
                "trade_date": date,
                "symbol": "A",
                "f1": 0.0,
                "target": 1.0,
                "train_target": 0.0,
            }
        )
        rows.append(
            {
                "trade_date": date,
                "symbol": "B",
                "f1": 1.0,
                "target": 0.0,
                "train_target": 1.0,
            }
        )
    df = pd.DataFrame(rows)

    scores = time_series_cv_ic(
        df,
        features=["f1"],
        target_col="target",
        fit_target_col="train_target",
        eval_target_col="target",
        n_splits=3,
        embargo_days=0,
        purge_days=0,
        model_cfg={"type": "ridge", "params": {"alpha": 1.0}},
        signal_direction=1.0,
    )

    assert len(scores) == 3
    assert all(score < 0 for score in scores if np.isfinite(score))


def test_time_series_cv_supports_ranker_date_equal_weights():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    rows = []
    for idx, date in enumerate(dates):
        rows.append({"trade_date": date, "symbol": "A", "f1": 0.1, "target": 0.0 + idx * 0.01})
        rows.append({"trade_date": date, "symbol": "B", "f1": 0.9, "target": 1.0 + idx * 0.01})
        rows.append({"trade_date": date, "symbol": "C", "f1": 0.5, "target": 0.5 + idx * 0.01})
    df = pd.DataFrame(rows)

    scores = time_series_cv_ic(
        df,
        features=["f1"],
        target_col="target",
        n_splits=3,
        embargo_days=0,
        purge_days=0,
        model_cfg={
            "type": "xgb_ranker",
            "params": {
                "n_estimators": 5,
                "max_depth": 2,
                "learning_rate": 0.1,
                "objective": "rank:pairwise",
                "random_state": 7,
            },
        },
        signal_direction=1.0,
        sample_weight_mode="date_equal",
    )

    assert len(scores) == 3
    assert all(np.isfinite(scores))


def test_time_series_cv_supports_custom_date_col_with_unsorted_rows():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    rows = []
    for idx, date in enumerate(dates):
        rows.append(
            {
                "trade_dt": date,
                "symbol": "A",
                "f1": 0.0,
                "target": 0.0 + idx * 0.01,
            }
        )
        rows.append(
            {
                "trade_dt": date,
                "symbol": "B",
                "f1": 1.0,
                "target": 1.0 + idx * 0.01,
            }
        )
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)

    scores = time_series_cv_ic(
        df,
        features=["f1"],
        target_col="target",
        n_splits=3,
        embargo_days=0,
        purge_days=0,
        model_cfg={"type": "ridge", "params": {"alpha": 1.0}},
        signal_direction=1.0,
        date_col="trade_dt",
    )

    assert len(scores) == 3
    assert all(np.isfinite(scores))


def test_time_series_cv_supports_exp_decay_and_train_window():
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    rows = []
    for idx, date in enumerate(dates):
        rows.append({"trade_date": date, "symbol": "A", "f1": 0.0, "target": 0.1 + idx * 0.01})
        rows.append({"trade_date": date, "symbol": "B", "f1": 1.0, "target": 0.9 + idx * 0.01})
    df = pd.DataFrame(rows)

    scores = time_series_cv_ic(
        df,
        features=["f1"],
        target_col="target",
        n_splits=3,
        embargo_days=0,
        purge_days=0,
        model_cfg={"type": "ridge", "params": {"alpha": 1.0}},
        signal_direction=1.0,
        sample_weight_mode="exp_decay",
        sample_weight_params={"halflife": 3.0},
        train_window_mode="rolling",
        train_window_size=5,
        train_window_unit="dates",
    )

    assert len(scores) == 3
    assert all(np.isfinite(scores))


def test_time_series_cv_avoids_trade_date_isin(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    rows = []
    for idx, date in enumerate(dates):
        rows.append({"trade_date": date, "symbol": "A", "f1": 0.0, "target": 0.0 + idx * 0.01})
        rows.append({"trade_date": date, "symbol": "B", "f1": 1.0, "target": 1.0 + idx * 0.01})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)

    original_isin = pd.Series.isin
    call_count = 0

    def _count_isin(series, values):
        nonlocal call_count
        if series.name == "trade_date":
            call_count += 1
        return original_isin(series, values)

    monkeypatch.setattr(pd.Series, "isin", _count_isin)

    scores = time_series_cv_ic(
        df,
        features=["f1"],
        target_col="target",
        n_splits=3,
        embargo_days=0,
        purge_days=0,
        model_cfg={"type": "ridge", "params": {"alpha": 1.0}},
        signal_direction=1.0,
    )

    assert len(scores) == 3
    assert all(np.isfinite(scores))
    assert call_count == 0
