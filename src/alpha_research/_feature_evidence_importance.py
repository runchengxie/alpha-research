from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._feature_evidence_io import (
    _families,
    _first_non_empty,
    _resolve_feature_list,
    _resolve_input_path,
    _section,
)
from .metrics import daily_ic_series, quantile_returns, summarize_ic


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


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
