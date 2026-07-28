"""Combinatorial purged cross-validation (CPCV) orchestration.

This module is a thin public surface for the CPCV implementation. The
historical single-file implementation has been split into private submodules
(``_cpcv_dates`` / ``_cpcv_groups`` / ``_cpcv_eval`` / ``_cpcv_report``) to keep
individual files smaller while preserving the exact public and private symbol
surface. Everything below is re-exported so existing imports from
``alpha_research.cpcv`` (and ``cpcv_module.<symbol>`` references inside
``cpcv_audit`` / ``artifact_cpcv``) keep working unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ._cpcv_dates import (
    LabelEventWindow,
    _as_date_tuple,
    _date_key,
    _format_date,
    _format_dates,
    _lookup_shifted_date,
    _to_jsonable,
)
from ._cpcv_eval import (
    _backtest_direction,
    _collapse_series_by_date,
    _empty_split_backtest_metrics,
    _evaluate_split,
    _evaluate_split_backtest,
    _evaluate_split_eval,
    _fit_split_model,
    _frame_for_dates,
    _prepare_split_frames,
    _resolve_cv_signal_direction,
    _resolve_train_ic_signal_direction,
    _sample_rebalance_frame,
    _score_frame,
    _score_with_request,
    _series_stat,
    _summarize_cpcv,
)
from ._cpcv_groups import (
    CPCVSplit,
    _apply_event_purge,
    _apply_gap_purge,
    _intervals_overlap,
    assign_cpcv_groups,
    build_cpcv_paths,
    build_cpcv_splits,
    build_label_event_windows,
    expected_cpcv_path_count,
)
from ._cpcv_report import _path_metric_row, _split_to_row

__all__ = [
    "CPCVSplit",
    "LabelEventWindow",
    "_apply_event_purge",
    "_apply_gap_purge",
    "_as_date_tuple",
    "_backtest_direction",
    "_collapse_series_by_date",
    "_date_key",
    "_empty_split_backtest_metrics",
    "_evaluate_split",
    "_evaluate_split_backtest",
    "_evaluate_split_eval",
    "_fit_split_model",
    "_format_date",
    "_format_dates",
    "_frame_for_dates",
    "_intervals_overlap",
    "_lookup_shifted_date",
    "_path_metric_row",
    "_prepare_split_frames",
    "_resolve_cv_signal_direction",
    "_resolve_train_ic_signal_direction",
    "_sample_rebalance_frame",
    "_score_frame",
    "_score_with_request",
    "_series_stat",
    "_split_to_row",
    "_summarize_cpcv",
    "_to_jsonable",
    "add_cpcv_args",
    "assign_cpcv_groups",
    "build_cpcv_paths",
    "build_cpcv_splits",
    "build_label_event_windows",
    "expected_cpcv_path_count",
    "run",
    "run_cpcv_audit",
]


def run_cpcv_audit(
    context: dict[str, Any],
    *,
    n_groups: int,
    test_groups: int,
    embargo_days: int | None,
    include_final_oos: bool,
    out_dir: Path,
) -> dict[str, Any]:
    from .cpcv_audit import run_cpcv_audit as _run_cpcv_audit

    return _run_cpcv_audit(
        context,
        n_groups=n_groups,
        test_groups=test_groups,
        embargo_days=embargo_days,
        include_final_oos=include_final_oos,
        out_dir=out_dir,
    )


def _default_out_dir(config_ref: str | Path | None) -> Path:
    tag = "default" if config_ref is None else Path(str(config_ref)).stem.replace(".", "_")
    return Path("artifacts") / "reports" / f"cpcv_{tag}"


def add_cpcv_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        default=None,
        help="Pipeline config path, artifact_cpcv config path, or built-in name.",
    )
    parser.add_argument(
        "--n-groups", type=int, default=8, help="Number of chronological CPCV groups."
    )
    parser.add_argument(
        "--test-groups", type=int, default=2, help="Number of groups tested per split."
    )
    parser.add_argument(
        "--embargo-days", type=int, default=None, help="Optional CPCV embargo days override."
    )
    parser.add_argument("--out", default=None, help="Output report directory.")
    parser.add_argument(
        "--include-final-oos",
        action="store_true",
        help="Include final OOS dates in the CPCV audit instead of reserving them.",
    )
    parser.add_argument(
        "--fail-on-quality",
        choices=["none", "info", "warning", "error"],
        default=None,
        help="Optional quality gate threshold forwarded to pipeline preparation.",
    )
    parser.add_argument("--artifacts-root", default=None, help="Optional artifacts root override.")
    return parser


def run(args: argparse.Namespace) -> int:
    from .artifact_cpcv import is_artifact_cpcv_config, run_artifact_cpcv

    out_dir = Path(args.out).expanduser() if args.out else _default_out_dir(args.config)
    if is_artifact_cpcv_config(args.config):
        summary = run_artifact_cpcv(
            args.config,
            n_groups=args.n_groups,
            test_groups=args.test_groups,
            embargo_days=args.embargo_days,
            out_dir=out_dir,
        )
        print(
            json.dumps(
                _to_jsonable({"output_dir": str(out_dir), **summary}),
                ensure_ascii=True,
                indent=2,
            )
        )
        return 0

    prepare_research_context = getattr(args, "prepare_research_context", None)
    if prepare_research_context is None:
        raise SystemExit(
            "Pipeline-backed CPCV requires a prepare_research_context adapter. "
            "Use the strategy CLI provided by strategy-pipeline or an artifact_cpcv config."
        )
    context = prepare_research_context(
        args.config,
        fail_on_quality=args.fail_on_quality,
        artifacts_root=args.artifacts_root,
    )
    summary = run_cpcv_audit(
        context,
        n_groups=args.n_groups,
        test_groups=args.test_groups,
        embargo_days=args.embargo_days,
        include_final_oos=bool(args.include_final_oos),
        out_dir=out_dir,
    )
    print(
        json.dumps(
            _to_jsonable({"output_dir": str(out_dir), **summary}), ensure_ascii=True, indent=2
        )
    )
    return 0
