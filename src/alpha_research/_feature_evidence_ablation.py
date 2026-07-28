from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._feature_evidence_importance import _to_float
from ._feature_evidence_io import (
    _families,
    _get_nested,
    _load_json,
    _load_yaml,
    _resolve_path,
    _safe_name,
    _section,
    _write_yaml,
)


def generate_ablation_jobs(config: dict[str, Any], *, config_dir: Path) -> dict[str, Any]:
    cfg = _section(config)
    base_config_path = _resolve_path(cfg.get("base_config"), base_dir=config_dir)
    if base_config_path is None:
        raise SystemExit("feature_evidence.base_config is required for generate-ablation.")
    base_cfg = _load_yaml(base_config_path)
    features_cfg = base_cfg.get("features") if isinstance(base_cfg.get("features"), dict) else {}
    feature_list = features_cfg.get("list")
    if not isinstance(feature_list, list) or not feature_list:
        raise SystemExit("Base config must include features.list for ablation generation.")
    feature_list_text = [str(item) for item in feature_list]
    families = _families(cfg.get("families"))

    output_dir = _resolve_path(
        cfg.get("output_dir") or "artifacts/sweeps/feature_evidence", base_dir=config_dir
    )
    assert output_dir is not None
    configs_dir = output_dir / "configs"
    jobs_path = output_dir / "jobs.csv"
    run_name_prefix = str(cfg.get("run_name_prefix") or "feature_ablation_")
    run_output_dir = cfg.get("runs_dir")

    rows: list[dict[str, Any]] = []

    def _write_variant(family: str, removed: list[str], cfg_payload: dict[str, Any]) -> None:
        run_name = run_name_prefix + _safe_name(family)
        eval_cfg = cfg_payload.setdefault("eval", {})
        if isinstance(eval_cfg, dict):
            eval_cfg["run_name"] = run_name
            if run_output_dir:
                eval_cfg["output_dir"] = str(run_output_dir)
        out_path = configs_dir / f"{_safe_name(family)}.yml"
        _write_yaml(out_path, cfg_payload)
        rows.append(
            {
                "family": family,
                "run_name": run_name,
                "config_path": str(out_path),
                "removed_features": ",".join(removed),
                "removed_count": len(removed),
            }
        )

    baseline = copy.deepcopy(base_cfg)
    _write_variant("baseline", [], baseline)
    for family, remove_features in families.items():
        missing = sorted(set(remove_features) - set(feature_list_text))
        retained = [feature for feature in feature_list_text if feature not in set(remove_features)]
        variant_cfg = copy.deepcopy(base_cfg)
        variant_cfg.setdefault("features", {})["list"] = retained
        variant_cfg.setdefault("metadata", {})["feature_ablation"] = {
            "family": family,
            "removed_features": remove_features,
            "missing_features": missing,
            "base_config": str(base_config_path),
        }
        _write_variant(f"minus_{family}", remove_features, variant_cfg)

    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    with jobs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["family", "run_name", "config_path", "removed_features", "removed_count"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return {"output_dir": str(output_dir), "jobs_csv": str(jobs_path), "jobs": rows}


def _read_stability(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    path_text = _get_nested(summary, "walk_forward", "feature_stability_file")
    candidates: list[Path] = []
    if path_text:
        raw = Path(str(path_text)).expanduser()
        candidates.extend([raw if raw.is_absolute() else run_dir / raw, Path.cwd() / raw])
    candidates.append(run_dir / "walk_forward_feature_stability.csv")
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            return {
                "available": False,
                "path": str(path),
                "top_k_hit_rate": None,
                "nonzero_hit_rate": None,
            }
        return {
            "available": True,
            "path": str(path),
            "top_k_hit_rate": _to_float(frame.get("top_k_hit_rate", pd.Series(dtype=float)).max()),
            "nonzero_hit_rate": _to_float(
                frame.get("nonzero_hit_rate", pd.Series(dtype=float)).max()
            ),
        }
    return {"available": False, "path": None, "top_k_hit_rate": None, "nonzero_hit_rate": None}


def _run_summary_row(entry: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    summary_path = _resolve_path(
        entry.get("summary_file") or entry.get("summary_path"), base_dir=base_dir
    )
    run_dir = _resolve_path(entry.get("run_dir"), base_dir=base_dir)
    if summary_path is None:
        if run_dir is None:
            raise SystemExit("Each ablation run requires summary_file or run_dir.")
        summary_path = run_dir / "summary.json"
    if run_dir is None:
        run_dir = summary_path.parent
    summary = _load_json(summary_path)
    stability = _read_stability(run_dir, summary)
    return {
        "family": str(entry.get("family") or entry.get("name") or run_dir.name),
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "eval_ic_mean": _to_float(_get_nested(summary, "eval", "ic", "mean")),
        "eval_ic_ir": _to_float(_get_nested(summary, "eval", "ic", "ir")),
        "eval_long_short": _to_float(_get_nested(summary, "eval", "long_short")),
        "walk_forward_test_ic_mean": _walk_forward_test_ic_mean(summary),
        "final_oos_ic_mean": _to_float(_get_nested(summary, "final_oos", "ic", "mean")),
        "backtest_sharpe": _to_float(_get_nested(summary, "backtest", "stats", "sharpe")),
        "backtest_max_drawdown": _to_float(
            _get_nested(summary, "backtest", "stats", "max_drawdown")
        ),
        "backtest_avg_turnover": _to_float(
            _get_nested(summary, "backtest", "stats", "avg_turnover")
        ),
        "backtest_avg_cost_drag": _to_float(
            _get_nested(summary, "backtest", "stats", "avg_cost_drag")
        ),
        "active_information_ratio": _to_float(
            _get_nested(summary, "backtest", "active", "information_ratio")
        ),
        "flag_constant_prediction": bool(
            _get_nested(summary, "eval", "constant_prediction") or False
        ),
        "flag_zero_feature_importance": bool(
            _get_nested(summary, "eval", "zero_feature_importance") or False
        ),
        "feature_stability_available": stability["available"],
        "feature_stability_top_k_hit_rate": stability["top_k_hit_rate"],
        "feature_stability_nonzero_hit_rate": stability["nonzero_hit_rate"],
        "feature_stability_path": stability["path"],
    }


def _walk_forward_test_ic_mean(summary: dict[str, Any]) -> float | None:
    results = _get_nested(summary, "walk_forward", "results")
    if not isinstance(results, list):
        return None
    values: list[float] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").lower() != "ok":
            continue
        value = _to_float(_get_nested(item, "test_ic", "mean"))
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else None


def summarize_ablation_results(config: dict[str, Any], *, config_dir: Path) -> list[dict[str, Any]]:
    cfg = _section(config)
    runs = cfg.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit("feature_evidence.runs must be a non-empty list for summarize-ablation.")
    rows = [
        _run_summary_row(entry, base_dir=config_dir) for entry in runs if isinstance(entry, dict)
    ]
    baseline = next((row for row in rows if row["family"] == "baseline"), rows[0] if rows else None)
    if baseline:
        for row in rows:
            for metric in (
                "eval_ic_ir",
                "eval_long_short",
                "walk_forward_test_ic_mean",
                "final_oos_ic_mean",
                "backtest_sharpe",
                "backtest_avg_turnover",
                "backtest_avg_cost_drag",
                "active_information_ratio",
            ):
                base_value = _to_float(baseline.get(metric))
                value = _to_float(row.get(metric))
                row[f"delta_{metric}_vs_baseline"] = (
                    value - base_value if value is not None and base_value is not None else None
                )
    return rows
