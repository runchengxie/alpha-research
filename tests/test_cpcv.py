from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from alpha_research import cpcv


def test_cpcv_group_and_split_counts():
    dates = pd.date_range("2020-01-31", periods=16, freq="ME")

    groups, splits = cpcv.build_cpcv_splits(dates, n_groups=8, test_groups=2)

    assert len(groups) == 8
    assert {len(value) for value in groups.values()} == {2}
    assert len(splits) == 28
    assert cpcv.expected_cpcv_path_count(8, 2) == 7
    assert splits[0].test_groups == (0, 1)


def test_cpcv_rejects_invalid_shape():
    dates = pd.date_range("2020-01-31", periods=8, freq="ME")

    with pytest.raises(ValueError, match="test_groups"):
        cpcv.build_cpcv_splits(dates, n_groups=4, test_groups=4)


def test_cpcv_event_window_purge_fixed_and_next_rebalance():
    trade_dates = pd.date_range("2020-01-01", periods=12, freq="D")
    signal_dates = trade_dates[:6]

    fixed_windows, fixed_mode = cpcv.build_label_event_windows(
        signal_dates,
        all_trade_dates=trade_dates,
        horizon_mode="fixed",
        horizon_days=3,
        shift_days=1,
    )
    assert fixed_mode == "event_window"
    assert fixed_windows[
        cast(pd.Timestamp, pd.Timestamp("2020-01-01"))
    ].label_start == pd.Timestamp("2020-01-02")
    assert fixed_windows[cast(pd.Timestamp, pd.Timestamp("2020-01-01"))].label_end == pd.Timestamp(
        "2020-01-05"
    )

    _, splits = cpcv.build_cpcv_splits(
        signal_dates[:4],
        n_groups=4,
        test_groups=1,
        event_windows=fixed_windows,
    )
    split_testing_second_date = next(
        split for split in splits if split.test_dates == (pd.Timestamp("2020-01-02"),)
    )
    assert pd.Timestamp("2020-01-01") in split_testing_second_date.purged_train_dates

    next_windows, next_mode = cpcv.build_label_event_windows(
        signal_dates[:4],
        all_trade_dates=trade_dates,
        horizon_mode="next_rebalance",
        horizon_days=3,
        shift_days=1,
        next_rebalance_map={
            pd.Timestamp("2020-01-01"): pd.Timestamp("2020-01-03"),
            pd.Timestamp("2020-01-02"): pd.Timestamp("2020-01-04"),
            pd.Timestamp("2020-01-03"): pd.Timestamp("2020-01-05"),
            pd.Timestamp("2020-01-04"): pd.Timestamp("2020-01-06"),
        },
    )
    assert next_mode == "event_window"
    assert next_windows[cast(pd.Timestamp, pd.Timestamp("2020-01-01"))].label_end == pd.Timestamp(
        "2020-01-04"
    )


def test_cpcv_marks_insufficient_data_after_purge():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    windows, _ = cpcv.build_label_event_windows(
        dates,
        all_trade_dates=pd.date_range("2020-01-01", periods=8, freq="D"),
        horizon_mode="fixed",
        horizon_days=6,
        shift_days=0,
    )

    _, splits = cpcv.build_cpcv_splits(
        dates,
        n_groups=4,
        test_groups=1,
        event_windows=windows,
        min_train_dates=1,
    )

    assert any(split.status == "insufficient_data" for split in splits)


def _fake_context(dates):
    request = SimpleNamespace(
        data=SimpleNamespace(all_dates=np.asarray(dates, dtype="datetime64[ns]")),
        model=SimpleNamespace(n_splits=2),
        backtest=SimpleNamespace(backtest_trading_days_per_year=12),
        feature_target=SimpleNamespace(target="future_return"),
        period=SimpleNamespace(n_quantiles=5),
    )
    return {
        "train_eval_request": request,
        "split_state": {
            "final_oos_dates": np.asarray(dates[-2:], dtype="datetime64[ns]"),
            "embargo_days": 0,
            "effective_gap_steps": 0,
        },
        "dataset_state": {
            "all_dates_model_full": np.asarray(dates, dtype="datetime64[ns]"),
            "reference_trade_dates": np.asarray(dates, dtype="datetime64[ns]"),
        },
        "panel_state": {"label_next_rebalance_map": {}},
        "date_label_settings": {
            "LABEL_HORIZON_MODE": "fixed",
            "LABEL_HORIZON_DAYS": 1,
            "LABEL_SHIFT_DAYS": 0,
        },
    }


def test_cpcv_report_schema_excludes_final_oos_by_default(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-31", periods=10, freq="ME")

    def fake_evaluate(_context, split):
        rows = []
        for date in split.test_dates:
            for idx in range(5):
                rows.append(
                    {
                        "trade_date": date,
                        "symbol": f"s{idx}",
                        "future_return": idx / 100.0,
                        "signal_eval": idx,
                    }
                )
        date = split.test_dates[0]
        net = pd.Series([0.01 + split.split_id / 1000.0], index=[date], name="net_return")
        gross = pd.Series([0.012 + split.split_id / 1000.0], index=[date], name="gross_return")
        turnover = pd.Series([0.2], index=[date], name="turnover")
        return {
            "status": "ok",
            "split": split,
            "eval_scored": pd.DataFrame(rows),
            "net_series": net,
            "gross_series": gross,
            "turnover_series": turnover,
            "benchmark_series": pd.Series([0.002], index=[date], name="benchmark_return"),
            "active_series": pd.Series([0.008], index=[date], name="active_return"),
            "period_info": [
                {"entry_idx": split.split_id, "exit_idx": split.split_id + 1, "exit_date": date}
            ],
            "bt_stats": {"avg_cost_drag": 0.001},
        }

    monkeypatch.setattr(cpcv, "_evaluate_split", fake_evaluate)

    summary = cpcv.run_cpcv_audit(
        _fake_context(dates),
        n_groups=4,
        test_groups=2,
        embargo_days=None,
        include_final_oos=False,
        out_dir=tmp_path,
    )

    assert summary["split_count"] == 6
    assert summary["path_count"] == 3
    assert summary["excluded_final_oos_dates"] == 2
    assert (tmp_path / "cpcv_splits.csv").exists()
    assert (tmp_path / "cpcv_path_returns.csv").exists()
    assert (tmp_path / "cpcv_path_metrics.csv").exists()
    assert (tmp_path / "cpcv_summary.json").exists()
    assert "sharpe_median" in summary
    assert "positive_sharpe_ratio" in summary


def test_cpcv_path_returns_collapse_duplicate_exit_dates():
    date = pd.Timestamp("2020-03-31")
    split_results = [
        {
            "split": SimpleNamespace(split_id=1, test_dates=(date,)),
            "eval_scored": pd.DataFrame(),
            "net_series": pd.Series([0.01], index=[date], name="net_return"),
            "gross_series": pd.Series([0.012], index=[date], name="gross_return"),
            "turnover_series": pd.Series([0.2], index=[date], name="turnover"),
            "benchmark_series": pd.Series([0.002], index=[date], name="benchmark_return"),
            "active_series": pd.Series([0.008], index=[date], name="active_return"),
            "period_info": [{"entry_idx": 0, "exit_idx": 1, "exit_date": date}],
            "bt_stats": {"avg_cost_drag": 0.001},
        },
        {
            "split": SimpleNamespace(split_id=2, test_dates=(date,)),
            "eval_scored": pd.DataFrame(),
            "net_series": pd.Series([0.03], index=[date], name="net_return"),
            "gross_series": pd.Series([0.032], index=[date], name="gross_return"),
            "turnover_series": pd.Series([0.4], index=[date], name="turnover"),
            "benchmark_series": pd.Series([0.006], index=[date], name="benchmark_return"),
            "active_series": pd.Series([0.024], index=[date], name="active_return"),
            "period_info": [{"entry_idx": 1, "exit_idx": 2, "exit_date": date}],
            "bt_stats": {"avg_cost_drag": 0.003},
        },
    ]

    _, return_rows = cpcv._path_metric_row(
        1,
        split_results,
        target_col="future_return",
        n_quantiles=5,
        trading_days_per_year=12,
    )

    assert len(return_rows) == 1
    assert return_rows[0]["net_return"] == pytest.approx(0.02)
    assert return_rows[0]["benchmark_return"] == pytest.approx(0.004)
    assert return_rows[0]["active_return"] == pytest.approx(0.016)
