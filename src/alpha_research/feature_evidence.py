from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from ._feature_evidence_ablation import (
    _families,
    _read_stability,
    _run_summary_row,
    _walk_forward_test_ic_mean,
    generate_ablation_jobs,
    summarize_ablation_results,
)
from ._feature_evidence_importance import (
    _cross_sectional_zscore,
    _factor_ic_input_path,
    _finite_or_nan,
    _load_factor_ic_frame,
    _permute_within_date,
    _to_float,
    _topk_metric,
    factor_ic_report,
    permutation_active_return_importance,
)
from ._feature_evidence_io import (
    _features_from_base_config,
    _first_non_empty,
    _get_nested,
    _load_json,
    _load_yaml,
    _resolve_feature_list,
    _resolve_input_path,
    _resolve_path,
    _safe_name,
    _section,
    _set_nested,
    _write_yaml,
)


def correlation_audit_report(
    config: dict[str, Any],
    *,
    config_dir: Path,
) -> list[dict[str, Any]]:
    from .feature_correlation import correlation_audit_report as _impl

    return _impl(config, config_dir=config_dir)


def drop_column_importance_report(
    config: dict[str, Any],
    *,
    config_dir: Path,
) -> list[dict[str, Any]]:
    from .feature_correlation import drop_column_importance_report as _impl

    return _impl(config, config_dir=config_dir)


def _write_rows(
    rows: list[dict[str, Any]], *, output_csv: Path | None, output_json: Path | None
) -> None:
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(rows, ensure_ascii=True, indent=2, default=str), encoding="utf-8"
        )


def add_feature_evidence_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "mode",
        choices=[
            "generate-ablation",
            "summarize-ablation",
            "permutation-importance",
            "factor-ic",
            "sfi",
            "correlation-audit",
            "drop-column-importance",
        ],
        help="Feature evidence workflow to run.",
    )
    parser.add_argument("--config", required=True, help="Feature evidence YAML config.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--output-json", default=None, help="Output JSON path.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    return parser


def run(args: argparse.Namespace) -> Any:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    config_path = _resolve_path(args.config)
    assert config_path is not None
    config = _load_yaml(config_path)
    cfg = _section(config)
    output_csv = _resolve_path(
        args.output or cfg.get("output_csv") or cfg.get("output"), base_dir=config_path.parent
    )
    output_json = _resolve_path(
        args.output_json or cfg.get("output_json"), base_dir=config_path.parent
    )

    if args.mode == "generate-ablation":
        result = generate_ablation_jobs(config, config_dir=config_path.parent)
        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8"
            )
        if not output_json:
            print(json.dumps(result, ensure_ascii=True, indent=2))
        return result

    if args.mode == "summarize-ablation":
        rows = summarize_ablation_results(config, config_dir=config_path.parent)
    elif args.mode == "drop-column-importance":
        rows = drop_column_importance_report(config, config_dir=config_path.parent)
    elif args.mode == "permutation-importance":
        rows = permutation_active_return_importance(config, config_dir=config_path.parent)
    elif args.mode in {"factor-ic", "sfi"}:
        rows = factor_ic_report(config, config_dir=config_path.parent)
    else:
        rows = correlation_audit_report(config, config_dir=config_path.parent)

    if output_csv is None and output_json is None:
        print(json.dumps(rows, ensure_ascii=True, indent=2, default=str))
    else:
        _write_rows(rows, output_csv=output_csv, output_json=output_json)
    return rows


__all__ = [
    "_cross_sectional_zscore",
    "_factor_ic_input_path",
    "_families",
    "_features_from_base_config",
    "_finite_or_nan",
    "_first_non_empty",
    "_get_nested",
    "_load_factor_ic_frame",
    "_load_json",
    "_load_yaml",
    "_permute_within_date",
    "_read_stability",
    "_resolve_feature_list",
    "_resolve_input_path",
    "_resolve_path",
    "_run_summary_row",
    "_safe_name",
    "_section",
    "_set_nested",
    "_to_float",
    "_topk_metric",
    "_walk_forward_test_ic_mean",
    "_write_rows",
    "_write_yaml",
    "add_feature_evidence_args",
    "correlation_audit_report",
    "drop_column_importance_report",
    "factor_ic_report",
    "generate_ablation_jobs",
    "permutation_active_return_importance",
    "run",
    "summarize_ablation_results",
]
