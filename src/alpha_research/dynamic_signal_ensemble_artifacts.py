from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .dynamic_signal_ensemble_types import DynamicSignalEnsembleResult


def stock_scores_to_long(
    scores: pd.DataFrame,
    *,
    score_col: str = "dynamic_ensemble_score",
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=pd.Index(["trade_date", "symbol", score_col]))
    out = scores.stack(future_stack=True).rename(score_col).reset_index()
    out.columns = ["trade_date", "symbol", score_col]
    return out.dropna(subset=[score_col]).reset_index(drop=True)


def write_dynamic_ensemble_artifacts(
    result: DynamicSignalEnsembleResult,
    *,
    output_dir: Path,
    score_col: str = "dynamic_ensemble_score",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "dynamic_scores.parquet"
    weights_path = output_dir / "stock_weights.parquet"
    factor_weights_path = output_dir / "factor_weights.parquet"
    factor_monitor_path = output_dir / "factor_monitor.csv"
    portfolio_monitor_path = output_dir / "portfolio_monitor.csv"
    direction_path = output_dir / "direction_calibration.csv"
    summary_path = output_dir / "dynamic_signal_ensemble_summary.json"

    stock_scores_to_long(result.stock_scores, score_col=score_col).to_parquet(
        scores_path,
        index=False,
    )
    result.stock_weights.to_parquet(weights_path)
    result.factor_weights.to_parquet(factor_weights_path)
    result.factor_monitor.to_csv(factor_monitor_path, index=False)
    result.portfolio_monitor.to_csv(portfolio_monitor_path, index=False)
    result.direction_calibration.to_csv(direction_path, index=False)
    paths = {
        "scores_file": str(scores_path),
        "stock_weights_file": str(weights_path),
        "factor_weights_file": str(factor_weights_path),
        "factor_monitor_file": str(factor_monitor_path),
        "portfolio_monitor_file": str(portfolio_monitor_path),
        "direction_calibration_file": str(direction_path),
        "summary_file": str(summary_path),
    }
    summary = {**result.summary, "files": paths}
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, default=str),
        encoding="utf-8",
    )
    return paths
