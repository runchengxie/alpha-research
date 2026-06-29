from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .metrics import daily_ic_series, quantile_returns, summarize_ic


def _resolve_path(path_text: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if path_text is None:
        return None
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        return (base_dir / candidate).resolve()
    return (Path.cwd() / candidate).resolve()


def _resolve_input_path(
    path_text: str | Path | None, *, base_dir: Path | None = None
) -> Path | None:
    path = _resolve_path(path_text, base_dir=base_dir)
    if path is None or path.exists():
        return path
    candidate = Path(path_text).expanduser()
    if not candidate.is_absolute():
        cwd_path = (Path.cwd() / candidate).resolve()
        if cwd_path.exists():
            return cwd_path
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"YAML file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to parse YAML file: {path} ({exc})") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise SystemExit(f"YAML root must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"JSON file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _set_nested(payload: dict[str, Any], path: str, value: Any) -> None:
    current = payload
    parts = [part for part in str(path).split(".") if part]
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    if parts:
        current[parts[-1]] = value


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _section(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("feature_evidence", config.get("feature_ablation", config))
    if not isinstance(raw, dict):
        raise SystemExit("feature_evidence must be a mapping.")
    return raw


def _families(raw: Any) -> dict[str, list[str]]:
    if isinstance(raw, dict):
        return {
            str(name): [str(item) for item in values]
            for name, values in raw.items()
            if isinstance(values, list)
        }
    if isinstance(raw, list):
        out: dict[str, list[str]] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise SystemExit("feature families list items must be mappings.")
            name = str(item.get("name") or "").strip()
            features = item.get("features")
            if not name or not isinstance(features, list):
                raise SystemExit("Each feature family requires name and features.")
            out[name] = [str(feature) for feature in features]
        return out
    raise SystemExit("feature_evidence.families must be a mapping or list.")


def _features_from_base_config(cfg: dict[str, Any], *, config_dir: Path) -> list[str]:
    base_config_path = _resolve_input_path(cfg.get("base_config"), base_dir=config_dir)
    if base_config_path is None or not base_config_path.exists():
        return []
    base_cfg = _load_yaml(base_config_path)
    features_cfg = base_cfg.get("features") if isinstance(base_cfg.get("features"), dict) else {}
    feature_list = features_cfg.get("list")
    if not isinstance(feature_list, list):
        return []
    return [str(item) for item in feature_list]


def _resolve_feature_list(
    cfg: dict[str, Any],
    *,
    config_dir: Path,
    prefer_base_config: bool,
) -> list[str]:
    features_raw = cfg.get("features")
    if isinstance(features_raw, list):
        return [str(feature) for feature in features_raw]
    if features_raw is not None:
        raise SystemExit("feature_evidence.features must be a list when provided.")

    if prefer_base_config:
        base_features = _features_from_base_config(cfg, config_dir=config_dir)
        if base_features:
            return base_features

    if cfg.get("families"):
        families = _families(cfg.get("families"))
        features = sorted({feature for values in families.values() for feature in values})
        if features:
            return features

    if not prefer_base_config:
        base_features = _features_from_base_config(cfg, config_dir=config_dir)
        if base_features:
            return base_features

    raise SystemExit(
        "feature_evidence.features, feature_evidence.base_config with features.list, "
        "or feature_evidence.families is required."
    )


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name).strip("_")


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


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


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


def _topk_metric(
    data: pd.DataFrame, score_col: str, target_col: str, top_k: int
) -> tuple[float, int]:
    values: list[float] = []
    for _, group in data.dropna(subset=[score_col, target_col]).groupby("trade_date"):
        if group.shape[0] < top_k:
            continue
        values.append(float(group.nlargest(top_k, score_col)[target_col].mean()))
    return (float(np.mean(values)) if values else np.nan, len(values))


def _cross_sectional_zscore(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    pieces: list[pd.Series] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        grouped = values.groupby(frame["trade_date"])
        z = grouped.transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
        )
        pieces.append(z.fillna(0.0))
    if not pieces:
        return pd.Series(0.0, index=frame.index)
    return pd.concat(pieces, axis=1).mean(axis=1)


def _permute_within_date(data: pd.DataFrame, column: str, rng: np.random.Generator) -> pd.Series:
    out = pd.Series(index=data.index, dtype=float)
    for _, idx in data.groupby("trade_date").groups.items():
        values = pd.to_numeric(data.loc[idx, column], errors="coerce").to_numpy(dtype=float)
        out.loc[idx] = rng.permutation(values)
    return out


def _load_factor_ic_frame(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    if "trade_date" not in data.columns:
        index_names = [name for name in data.index.names if name is not None]
        if "trade_date" in index_names:
            data = data.reset_index()
    if "trade_date" not in data.columns:
        raise SystemExit("Factor IC input must include trade_date as a column or index level.")
    data = data.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    return data


def _finite_or_nan(value: Any) -> float:
    number = _to_float(value)
    return float(number) if number is not None else np.nan


def _factor_ic_input_path(cfg: dict[str, Any], *, config_dir: Path) -> Path:
    path = _resolve_input_path(
        _first_non_empty(
            cfg.get("factor_ic_file"),
            cfg.get("dataset_file"),
            cfg.get("scored_file"),
        ),
        base_dir=config_dir,
    )
    if path is None or not path.exists():
        raise SystemExit(
            "feature_evidence.factor_ic_file, dataset_file, or scored_file is required "
            "for factor-ic."
        )
    return path


def factor_ic_report(
    config: dict[str, Any],
    *,
    config_dir: Path,
) -> list[dict[str, Any]]:
    cfg = _section(config)
    input_path = _factor_ic_input_path(cfg, config_dir=config_dir)
    data = _load_factor_ic_frame(input_path)
    target_col = str(cfg.get("target_col") or "future_return")
    n_quantiles = int(cfg.get("n_quantiles") or 5)
    if n_quantiles < 2:
        raise SystemExit("feature_evidence.n_quantiles must be >= 2 for factor-ic.")
    if target_col not in data.columns:
        raise SystemExit(f"Missing target column: {target_col}")

    features = _resolve_feature_list(cfg, config_dir=config_dir, prefer_base_config=True)
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise SystemExit(
            "Missing feature columns for factor-ic: "
            + ", ".join(missing_features)
            + ". Use dataset.parquet or another factor_ic_file that includes the feature columns."
        )

    total_rows = int(data[target_col].notna().sum())
    rows: list[dict[str, Any]] = []
    for feature in features:
        subset = data[["trade_date", target_col, feature]].copy()
        valid = subset.dropna(subset=[target_col, feature])
        valid_rows = int(valid.shape[0])
        coverage = float(valid_rows / total_rows) if total_rows > 0 else np.nan

        ic_series = daily_ic_series(valid, target_col, feature)
        ic_stats = summarize_ic(ic_series)
        pearson_ic_series = daily_ic_series(valid, target_col, feature, method="pearson")
        pearson_ic_stats = summarize_ic(pearson_ic_series)

        quantile_ts = quantile_returns(valid, feature, target_col, n_quantiles)
        quantile_mean = quantile_ts.mean() if not quantile_ts.empty else pd.Series(dtype=float)
        q1_return = (
            float(quantile_mean.iloc[0])
            if not quantile_mean.empty and np.isfinite(quantile_mean.iloc[0])
            else np.nan
        )
        qN_return = (
            float(quantile_mean.iloc[-1])
            if not quantile_mean.empty and np.isfinite(quantile_mean.iloc[-1])
            else np.nan
        )
        long_short = (
            float(qN_return - q1_return)
            if np.isfinite(q1_return) and np.isfinite(qN_return)
            else np.nan
        )
        positive_ic_ratio = (
            float((ic_series.dropna() > 0).mean()) if not ic_series.dropna().empty else np.nan
        )

        rows.append(
            {
                "feature": feature,
                "n": int(ic_stats["n"]),
                "ic_mean": _finite_or_nan(ic_stats["mean"]),
                "ic_std": _finite_or_nan(ic_stats["std"]),
                "ic_ir": _finite_or_nan(ic_stats["ir"]),
                "t_stat": _finite_or_nan(ic_stats["t_stat"]),
                "p_value": _finite_or_nan(ic_stats["p_value"]),
                "pearson_ic_mean": _finite_or_nan(pearson_ic_stats["mean"]),
                "pearson_ic_std": _finite_or_nan(pearson_ic_stats["std"]),
                "pearson_ic_ir": _finite_or_nan(pearson_ic_stats["ir"]),
                "pearson_t_stat": _finite_or_nan(pearson_ic_stats["t_stat"]),
                "pearson_p_value": _finite_or_nan(pearson_ic_stats["p_value"]),
                "q1_return": q1_return,
                "qN_return": qN_return,
                "long_short": long_short,
                "coverage": coverage,
                "positive_ic_ratio": positive_ic_ratio,
                "valid_rows": valid_rows,
                "total_rows": total_rows,
                "n_quantiles": n_quantiles,
                "input_file": str(input_path),
                "target_col": target_col,
            }
        )
    rows.sort(
        key=lambda row: (
            -abs(row["ic_mean"]) if np.isfinite(row["ic_mean"]) else float("inf"),
            row["feature"],
        )
    )
    return rows


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


def permutation_active_return_importance(
    config: dict[str, Any],
    *,
    config_dir: Path,
) -> list[dict[str, Any]]:
    cfg = _section(config)
    scored_path = _resolve_input_path(cfg.get("scored_file"), base_dir=config_dir)
    if scored_path is None or not scored_path.exists():
        raise SystemExit("feature_evidence.scored_file is required for permutation importance.")
    data = pd.read_parquet(scored_path)
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    score_col = str(cfg.get("score_col") or cfg.get("signal_col") or "signal_backtest")
    target_col = str(cfg.get("target_col") or "future_return")
    top_k = int(cfg.get("top_k") or 10)
    if top_k <= 0:
        raise SystemExit("feature_evidence.top_k must be positive.")
    missing = [col for col in ("trade_date", score_col, target_col) if col not in data.columns]
    if missing:
        raise SystemExit("Missing required scored columns: " + ", ".join(missing))

    features = _resolve_feature_list(cfg, config_dir=config_dir, prefer_base_config=False)
    families = _families(cfg.get("families", {})) if cfg.get("families") else {}
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise SystemExit("Missing feature columns: " + ", ".join(missing_features))

    seed = int(cfg.get("seed") or 42)
    n_repeats = int(cfg.get("n_repeats") or 5)
    if n_repeats <= 0:
        raise SystemExit("feature_evidence.n_repeats must be positive.")

    baseline_metric, n_dates = _topk_metric(data, score_col, target_col, top_k)
    rows: list[dict[str, Any]] = []

    def _importance(name: str, kind: str, columns: list[str]) -> dict[str, Any]:
        feature_score_col = "__feature_proxy_score"
        working = data.copy()
        working[feature_score_col] = _cross_sectional_zscore(working, columns)
        feature_metric, feature_dates = _topk_metric(working, feature_score_col, target_col, top_k)
        permuted_metrics: list[float] = []
        for repeat in range(n_repeats):
            permuted = data.copy()
            rng = np.random.default_rng(seed + repeat)
            for column in columns:
                permuted[column] = _permute_within_date(permuted, column, rng)
            permuted[feature_score_col] = _cross_sectional_zscore(permuted, columns)
            metric, _ = _topk_metric(permuted, feature_score_col, target_col, top_k)
            if np.isfinite(metric):
                permuted_metrics.append(metric)
        permuted_metric = float(np.mean(permuted_metrics)) if permuted_metrics else np.nan
        return {
            "name": name,
            "kind": kind,
            "features": ",".join(columns),
            "feature_count": len(columns),
            "top_k": top_k,
            "n_dates": feature_dates,
            "baseline_score_metric": baseline_metric,
            "baseline_score_n_dates": n_dates,
            "feature_metric": feature_metric,
            "permuted_metric": permuted_metric,
            "permutation_importance": (
                feature_metric - permuted_metric
                if np.isfinite(feature_metric) and np.isfinite(permuted_metric)
                else np.nan
            ),
            "delta_vs_baseline_score": (
                feature_metric - baseline_metric
                if np.isfinite(feature_metric) and np.isfinite(baseline_metric)
                else np.nan
            ),
        }

    for feature in features:
        rows.append(_importance(feature, "feature", [feature]))
    for family, columns in families.items():
        valid_columns = [column for column in columns if column in data.columns]
        if valid_columns:
            rows.append(_importance(family, "family", valid_columns))
    return rows


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
