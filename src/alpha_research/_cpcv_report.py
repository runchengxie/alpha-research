"""CPCV report-row assembly and JSON serialization (private helpers).

Re-exported from ``alpha_research.cpcv`` so existing imports keep working.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._cpcv_dates import _format_date
from ._cpcv_eval import _collapse_series_by_date
from ._cpcv_groups import CPCVSplit
from .metrics import (
    daily_ic_series,
    quantile_returns,
    summarize_active_returns,
    summarize_ic,
)
from .return_metrics import summarize_period_returns


def _split_to_row(split: CPCVSplit) -> dict[str, Any]:
    return {
        "split_id": split.split_id,
        "test_groups": "|".join(str(group) for group in split.test_groups),
        "train_groups": "|".join(str(group) for group in split.train_groups),
        "train_start": _format_date(split.train_dates[0]) if split.train_dates else None,
        "train_end": _format_date(split.train_dates[-1]) if split.train_dates else None,
        "test_start": _format_date(split.test_dates[0]) if split.test_dates else None,
        "test_end": _format_date(split.test_dates[-1]) if split.test_dates else None,
        "train_dates_raw": len(split.train_dates_raw),
        "train_dates": len(split.train_dates),
        "test_dates": len(split.test_dates),
        "purged_train_dates": len(split.purged_train_dates),
        "embargoed_train_dates": len(split.embargoed_train_dates),
        "purge_mode": split.purge_mode,
        "status": split.status,
    }


def _path_metric_row(
    path_id: int,
    split_results: list[dict[str, Any]],
    *,
    target_col: str,
    n_quantiles: int,
    trading_days_per_year: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not split_results:
        return None, []
    split_ids = [int(item["split"].split_id) for item in split_results]
    test_dates = sorted({date for item in split_results for date in item["split"].test_dates})
    eval_frames = [item["eval_scored"] for item in split_results if not item["eval_scored"].empty]
    eval_frame = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    net_series = _collapse_series_by_date(
        pd.concat([item["net_series"] for item in split_results]).sort_index()
    )
    gross_series = _collapse_series_by_date(
        pd.concat([item["gross_series"] for item in split_results]).sort_index()
    )
    turnover_series = _collapse_series_by_date(
        pd.concat([item["turnover_series"] for item in split_results]).sort_index()
    )
    benchmark_series = _collapse_series_by_date(
        pd.concat([item["benchmark_series"] for item in split_results]).sort_index()
    )
    active_series = _collapse_series_by_date(
        pd.concat([item["active_series"] for item in split_results]).sort_index()
    )
    period_info = [period for item in split_results for period in item["period_info"]]
    period_info = sorted(period_info, key=lambda item: item.get("exit_date"))
    stats = summarize_period_returns(net_series, period_info, trading_days_per_year)

    if not eval_frame.empty:
        if target_col not in eval_frame.columns:
            candidates = [col for col in eval_frame.columns if col.endswith("return")]
            target_col = candidates[0] if candidates else "future_return"
        ic_stats = summarize_ic(daily_ic_series(eval_frame, target_col, "signal_eval"))
        q = quantile_returns(eval_frame, "signal_eval", target_col, n_quantiles)
        q_mean = q.mean() if not q.empty else pd.Series(dtype=float)
        long_short = float(q_mean.iloc[-1] - q_mean.iloc[0]) if not q_mean.empty else np.nan
    else:
        ic_stats = {}
        long_short = np.nan
    active_stats = None
    if not active_series.empty and not benchmark_series.empty:
        active_stats, _ = summarize_active_returns(
            net_series,
            benchmark_series,
            stats.get("periods_per_year", np.nan),
        )
    row = {
        "path_id": path_id,
        "split_ids": "|".join(str(split_id) for split_id in split_ids),
        "test_start": _format_date(test_dates[0]) if test_dates else None,
        "test_end": _format_date(test_dates[-1]) if test_dates else None,
        "observation_count": len(net_series) if not net_series.empty else len(eval_frame),
        "sharpe": stats.get("sharpe"),
        "total_return": stats.get("total_return"),
        "ann_return": stats.get("ann_return"),
        "ann_vol": stats.get("ann_vol"),
        "max_drawdown": stats.get("max_drawdown"),
        "ic_mean": ic_stats.get("mean"),
        "ic_ir": ic_stats.get("ir"),
        "long_short": long_short,
        "avg_turnover": float(turnover_series.mean()) if not turnover_series.empty else np.nan,
        "avg_cost_drag": np.nan,
        "active_total_return": (active_stats or {}).get("active_total_return"),
        "information_ratio": (active_stats or {}).get("information_ratio"),
        "tracking_error": (active_stats or {}).get("tracking_error"),
    }
    split_costs = [
        item["bt_stats"].get("avg_cost_drag")
        for item in split_results
        if item.get("bt_stats") and pd.notna(item["bt_stats"].get("avg_cost_drag"))
    ]
    if split_costs:
        row["avg_cost_drag"] = float(np.mean(split_costs))

    return_rows: list[dict[str, Any]] = []
    index_values = sorted(
        set(net_series.index).union(gross_series.index).union(benchmark_series.index)
    )
    for date in index_values:
        net_value = net_series.get(date, np.nan)
        gross_value = gross_series.get(date, np.nan)
        benchmark_value = benchmark_series.get(date, np.nan)
        return_rows.append(
            {
                "path_id": path_id,
                "date": _format_date(date),
                "net_return": float(net_value) if pd.notna(net_value) else np.nan,
                "gross_return": float(gross_value) if pd.notna(gross_value) else np.nan,
                "benchmark_return": float(benchmark_value) if pd.notna(benchmark_value) else np.nan,
                "active_return": float(net_value - benchmark_value)
                if pd.notna(net_value) and pd.notna(benchmark_value)
                else np.nan,
            }
        )
    return row, return_rows
