from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from . import cpcv as cpcv_module
from .date_slices import _build_trade_date_slices

logger = logging.getLogger("cstree")


@dataclass(frozen=True)
class _AuditDateScope:
    context: dict[str, Any]
    request: Any
    eligible_dates: tuple[pd.Timestamp, ...]
    final_oos_dates: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class _AuditPurgeSetup:
    event_windows: dict[pd.Timestamp, Any]
    purge_mode: str
    effective_embargo: int | None


@dataclass(frozen=True)
class _AuditPathRows:
    metrics: list[dict[str, Any]]
    returns: list[dict[str, Any]]


def _replace_request_data_dates(
    request: Any,
    *,
    df_model_sorted: pd.DataFrame,
    all_dates: Any,
    start_rows: Any,
    end_rows: Any,
    date_to_pos: dict[pd.Timestamp, int],
) -> Any:
    data = replace(
        request.data,
        df_model_sorted=df_model_sorted,
        all_dates=all_dates,
        all_date_start_rows=start_rows,
        all_date_end_rows=end_rows,
        all_date_to_pos=date_to_pos,
    )
    return replace(request, data=data)


def _resolve_audit_date_scope(
    context: dict[str, Any],
    *,
    include_final_oos: bool,
) -> _AuditDateScope:
    request = context["train_eval_request"]
    split_state = context["split_state"]
    dataset_state = context["dataset_state"]
    all_dates = (
        dataset_state["all_dates_model_full"] if include_final_oos else request.data.all_dates
    )
    eligible_dates = cpcv_module._as_date_tuple(all_dates)
    final_oos_dates = cpcv_module._as_date_tuple(split_state.get("final_oos_dates", []))
    if not include_final_oos and final_oos_dates:
        final_set = set(final_oos_dates)
        eligible_dates = tuple(date for date in eligible_dates if date not in final_set)

    if include_final_oos:
        (
            df_model_sorted,
            date_values,
            start_rows,
            end_rows,
            date_to_pos,
        ) = _build_trade_date_slices(dataset_state["df_model_all"])
        request = _replace_request_data_dates(
            request,
            df_model_sorted=df_model_sorted,
            all_dates=date_values,
            start_rows=start_rows,
            end_rows=end_rows,
            date_to_pos=date_to_pos,
        )
        context = {**context, "train_eval_request": request}
    return _AuditDateScope(
        context=context,
        request=request,
        eligible_dates=eligible_dates,
        final_oos_dates=final_oos_dates,
    )


def _resolve_audit_purge_setup(
    context: dict[str, Any],
    *,
    request: Any,
    eligible_dates: tuple[pd.Timestamp, ...],
    embargo_days: int | None,
) -> _AuditPurgeSetup:
    split_state = context["split_state"]
    dataset_state = context["dataset_state"]
    panel_state = context["panel_state"]
    date_settings = context["date_label_settings"]
    all_trade_dates = dataset_state.get("reference_trade_dates")
    if all_trade_dates is None or len(all_trade_dates) == 0:
        all_trade_dates = sorted(request.data.df_full["trade_date"].unique())
    event_windows, purge_mode = cpcv_module.build_label_event_windows(
        eligible_dates,
        all_trade_dates=all_trade_dates,
        horizon_mode=date_settings["LABEL_HORIZON_MODE"],
        horizon_days=int(date_settings["LABEL_HORIZON_DAYS"]),
        shift_days=int(date_settings["LABEL_SHIFT_DAYS"]),
        next_rebalance_map=panel_state.get("label_next_rebalance_map"),
    )
    if purge_mode == "fallback_gap":
        logger.warning("CPCV event-window purge unavailable; using fallback gap purge.")
    effective_embargo = split_state.get("embargo_days", 0) if embargo_days is None else embargo_days
    return _AuditPurgeSetup(
        event_windows=event_windows,
        purge_mode=purge_mode,
        effective_embargo=effective_embargo,
    )


def _build_audit_splits(
    context: dict[str, Any],
    *,
    request: Any,
    eligible_dates: tuple[pd.Timestamp, ...],
    purge_setup: _AuditPurgeSetup,
    n_groups: int,
    test_groups: int,
) -> list[Any]:
    split_state = context["split_state"]
    _groups, splits = cpcv_module.build_cpcv_splits(
        eligible_dates,
        n_groups=n_groups,
        test_groups=test_groups,
        event_windows=purge_setup.event_windows
        if purge_setup.purge_mode == "event_window"
        else None,
        embargo_days=int(purge_setup.effective_embargo or 0),
        fallback_gap_steps=int(split_state.get("effective_gap_steps", 0) or 0),
        min_train_dates=max(2, int(request.model.n_splits) + 1),
        min_test_dates=1,
    )
    return splits


def _evaluate_audit_splits(
    context: dict[str, Any],
    splits: list[Any],
) -> tuple[dict[int, dict[str, Any]], list[Any]]:
    split_results: dict[int, dict[str, Any]] = {}
    for split in splits:
        result = cpcv_module._evaluate_split(context, split)
        split_results[split.split_id] = result

    valid_splits = [
        split
        for split in splits
        if split.status == "ok" and split_results.get(split.split_id, {}).get("status") == "ok"
    ]
    return split_results, valid_splits


def _build_audit_path_rows(
    request: Any,
    split_results: dict[int, dict[str, Any]],
    valid_splits: list[Any],
    *,
    n_groups: int,
    test_groups: int,
) -> _AuditPathRows:
    paths = cpcv_module.build_cpcv_paths(valid_splits, n_groups=n_groups, test_groups=test_groups)
    path_metric_rows: list[dict[str, Any]] = []
    path_return_rows: list[dict[str, Any]] = []
    for path_idx, path_splits in enumerate(paths, start=1):
        results = [split_results[split.split_id] for split in path_splits]
        metric_row, return_rows = cpcv_module._path_metric_row(
            path_idx,
            results,
            target_col=request.feature_target.target,
            n_quantiles=request.period.n_quantiles,
            trading_days_per_year=request.backtest.backtest_trading_days_per_year,
        )
        if metric_row is not None:
            path_metric_rows.append(metric_row)
            path_return_rows.extend(return_rows)
    return _AuditPathRows(metrics=path_metric_rows, returns=path_return_rows)


def _build_audit_summary(
    *,
    n_groups: int,
    test_groups: int,
    splits: list[Any],
    valid_splits: list[Any],
    date_scope: _AuditDateScope,
    purge_setup: _AuditPurgeSetup,
    include_final_oos: bool,
    path_metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible_dates = date_scope.eligible_dates
    final_oos_dates = date_scope.final_oos_dates
    return {
        "n_groups": int(n_groups),
        "test_groups": int(test_groups),
        "split_count": len(splits),
        "valid_split_count": len(valid_splits),
        "path_count": cpcv_module.expected_cpcv_path_count(n_groups, test_groups),
        "eligible_date_count": len(eligible_dates),
        "eligible_start": cpcv_module._format_date(eligible_dates[0]) if eligible_dates else None,
        "eligible_end": cpcv_module._format_date(eligible_dates[-1]) if eligible_dates else None,
        "include_final_oos": bool(include_final_oos),
        "excluded_final_oos_dates": len(final_oos_dates) if not include_final_oos else 0,
        "purge_mode": purge_setup.purge_mode,
        "embargo_days": int(purge_setup.effective_embargo or 0),
        **cpcv_module._summarize_cpcv(path_metric_rows),
    }


def _write_audit_reports(
    out_dir: Path,
    *,
    splits: list[Any],
    path_rows: _AuditPathRows,
    summary: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([cpcv_module._split_to_row(split) for split in splits]).to_csv(
        out_dir / "cpcv_splits.csv",
        index=False,
    )
    pd.DataFrame(path_rows.returns).to_csv(out_dir / "cpcv_path_returns.csv", index=False)
    pd.DataFrame(path_rows.metrics).to_csv(out_dir / "cpcv_path_metrics.csv", index=False)
    (out_dir / "cpcv_summary.json").write_text(
        json.dumps(cpcv_module._to_jsonable(summary), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def run_cpcv_audit(
    context: dict[str, Any],
    *,
    n_groups: int,
    test_groups: int,
    embargo_days: int | None,
    include_final_oos: bool,
    out_dir: Path,
) -> dict[str, Any]:
    date_scope = _resolve_audit_date_scope(context, include_final_oos=include_final_oos)
    purge_setup = _resolve_audit_purge_setup(
        date_scope.context,
        request=date_scope.request,
        eligible_dates=date_scope.eligible_dates,
        embargo_days=embargo_days,
    )
    splits = _build_audit_splits(
        date_scope.context,
        request=date_scope.request,
        eligible_dates=date_scope.eligible_dates,
        purge_setup=purge_setup,
        n_groups=n_groups,
        test_groups=test_groups,
    )
    split_results, valid_splits = _evaluate_audit_splits(date_scope.context, splits)
    path_rows = _build_audit_path_rows(
        date_scope.request,
        split_results,
        valid_splits,
        n_groups=n_groups,
        test_groups=test_groups,
    )
    summary = _build_audit_summary(
        n_groups=n_groups,
        test_groups=test_groups,
        splits=splits,
        valid_splits=valid_splits,
        date_scope=date_scope,
        purge_setup=purge_setup,
        include_final_oos=include_final_oos,
        path_metric_rows=path_rows.metrics,
    )
    _write_audit_reports(out_dir, splits=splits, path_rows=path_rows, summary=summary)
    return summary
