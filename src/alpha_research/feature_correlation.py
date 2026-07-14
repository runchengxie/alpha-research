from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_evidence import (
    _first_non_empty,
    _resolve_feature_list,
    _resolve_input_path,
    _section,
    _to_float,
    summarize_ablation_results,
)


def _load_feature_matrix(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        data = pd.read_parquet(path)
    else:
        data = pd.read_csv(path)
    if "trade_date" not in data.columns:
        index_names = [name for name in data.index.names if name is not None]
        if "trade_date" in index_names:
            data = data.reset_index()
    if "trade_date" in data.columns:
        data = data.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"])
    return data


def _union_find_clusters(features: list[str], pairs: list[tuple[str, str]]) -> dict[str, int]:
    parent = {feature: feature for feature in features}

    def find(feature: str) -> str:
        while parent[feature] != feature:
            parent[feature] = parent[parent[feature]]
            feature = parent[feature]
        return feature

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in pairs:
        union(left, right)
    root_ids = {
        root: idx + 1 for idx, root in enumerate(sorted({find(feature) for feature in features}))
    }
    return {feature: root_ids[find(feature)] for feature in features}


def correlation_audit_report(
    config: dict[str, Any],
    *,
    config_dir: Path,
) -> list[dict[str, Any]]:
    cfg = _section(config)
    input_path = _resolve_input_path(
        _first_non_empty(
            cfg.get("correlation_file"),
            cfg.get("factor_ic_file"),
            cfg.get("dataset_file"),
            cfg.get("scored_file"),
        ),
        base_dir=config_dir,
    )
    if input_path is None or not input_path.exists():
        raise SystemExit(
            "feature_evidence.correlation_file, dataset_file, factor_ic_file, "
            "or scored_file is required for correlation-audit."
        )
    data = _load_feature_matrix(input_path)
    features = _resolve_feature_list(cfg, config_dir=config_dir, prefer_base_config=True)
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        missing_text = ", ".join(missing_features)
        raise SystemExit(f"Missing feature columns for correlation-audit: {missing_text}")

    method = str(cfg.get("correlation_method") or "spearman").strip().lower()
    threshold = float(cfg.get("correlation_threshold", cfg.get("threshold", 0.90)))
    if not 0.0 < threshold <= 1.0:
        raise SystemExit("feature_evidence.correlation_threshold must be in (0, 1].")
    corr = data[features].apply(pd.to_numeric, errors="coerce").corr(method=method)
    high_pairs: list[tuple[str, str]] = []
    pair_rows: list[dict[str, Any]] = []
    for left_idx, left in enumerate(features):
        for right in features[left_idx + 1 :]:
            value = _to_float(corr.loc[left, right])
            abs_value = abs(value) if value is not None else np.nan
            is_high = bool(np.isfinite(abs_value) and abs_value >= threshold)
            if is_high:
                high_pairs.append((left, right))
            pair_rows.append(
                {
                    "row_type": "pair",
                    "feature_a": left,
                    "feature_b": right,
                    "corr": value,
                    "abs_corr": abs_value,
                    "method": method,
                    "threshold": threshold,
                    "is_high_corr": is_high,
                    "cluster_id": None,
                    "cluster_size": None,
                    "input_file": str(input_path),
                }
            )
    clusters = _union_find_clusters(features, high_pairs)
    cluster_sizes = {
        cluster_id: sum(1 for value in clusters.values() if value == cluster_id)
        for cluster_id in set(clusters.values())
    }
    for row in pair_rows:
        if row["is_high_corr"]:
            row["cluster_id"] = clusters[str(row["feature_a"])]
            row["cluster_size"] = cluster_sizes[int(row["cluster_id"])]
    cluster_rows = [
        {
            "row_type": "cluster",
            "feature_a": feature,
            "feature_b": None,
            "corr": None,
            "abs_corr": None,
            "method": method,
            "threshold": threshold,
            "is_high_corr": cluster_sizes[cluster_id] > 1,
            "cluster_id": cluster_id,
            "cluster_size": cluster_sizes[cluster_id],
            "input_file": str(input_path),
        }
        for feature, cluster_id in sorted(clusters.items(), key=lambda item: (item[1], item[0]))
    ]
    return cluster_rows + pair_rows


def drop_column_importance_report(
    config: dict[str, Any],
    *,
    config_dir: Path,
) -> list[dict[str, Any]]:
    rows = summarize_ablation_results(config, config_dir=config_dir)
    out: list[dict[str, Any]] = []
    for row in rows:
        family = str(row.get("family") or "")
        if family == "baseline":
            continue
        result = dict(row)
        result["importance_kind"] = "drop_column"
        result["dropped_group"] = family.removeprefix("minus_")
        for metric in (
            "eval_ic_ir",
            "eval_long_short",
            "walk_forward_test_ic_mean",
            "final_oos_ic_mean",
            "backtest_sharpe",
            "active_information_ratio",
        ):
            delta = _to_float(row.get(f"delta_{metric}_vs_baseline"))
            result[f"drop_importance_{metric}"] = -delta if delta is not None else None
        out.append(result)
    return out
