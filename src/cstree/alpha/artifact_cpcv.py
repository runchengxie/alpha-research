from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..backtesting.metrics import summarize_period_returns
from ..metrics import summarize_active_returns
from ..pipeline.eval_benchmark import build_benchmark_series
from . import cpcv as cpcv_module
from .transform import apply_score_postprocess


@dataclass(frozen=True)
class ArtifactCPCVInputs:
    cfg: dict[str, Any]
    config_path: Path
    run_dir: Path | None
    periods: pd.DataFrame
    scored: pd.DataFrame
    benchmark_returns: pd.Series
    positions_rows: int | None


def _resolve_path(path_text: Any, *, base_dir: Path) -> Path | None:
    if path_text is None or str(path_text).strip() == "":
        return None
    candidate = Path(str(path_text)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    by_base = (base_dir / candidate).resolve()
    if by_base.exists():
        return by_base
    return (Path.cwd() / candidate).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def is_artifact_cpcv_config(config_ref: str | Path | None) -> bool:
    if config_ref is None:
        return False
    path = Path(str(config_ref)).expanduser()
    if not path.exists() or not path.is_file():
        return False
    return isinstance(_load_yaml(path).get("artifact_cpcv"), dict)


def _artifact_payload(path: Path) -> dict[str, Any]:
    payload = _load_yaml(path)
    cfg = payload.get("artifact_cpcv")
    if not isinstance(cfg, dict):
        raise SystemExit("artifact_cpcv config must include an artifact_cpcv mapping.")
    return cfg


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Artifact CPCV input not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _date_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, errors="coerce")
    compact = text.str.fullmatch(r"\d{8}")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return parsed.dt.normalize()


def _normalize_periods(periods: pd.DataFrame) -> pd.DataFrame:
    required = {"rebalance_date", "entry_date", "exit_date", "net_return"}
    missing = sorted(required - set(periods.columns))
    if missing:
        raise SystemExit("Artifact CPCV periods file is missing: " + ", ".join(missing))
    out = periods.copy()
    out["rebalance_ts"] = _date_series(out["rebalance_date"])
    out["entry_ts"] = _date_series(out["entry_date"])
    out["exit_ts"] = _date_series(out["exit_date"])
    for col in ("net_return", "gross_return", "turnover", "total_cost"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["rebalance_ts", "entry_ts", "exit_ts", "net_return"])
    if "entry_idx" not in out.columns:
        out["entry_idx"] = range(out.shape[0])
    if "exit_idx" not in out.columns:
        out["exit_idx"] = pd.to_numeric(out["entry_idx"], errors="coerce").fillna(0).astype(int) + 1
    if "planned_exit_idx" not in out.columns:
        out["planned_exit_idx"] = out["exit_idx"]
    if "planned_exit_date" not in out.columns:
        out["planned_exit_date"] = out["exit_date"]
    return out.sort_values(["rebalance_ts", "exit_ts"]).reset_index(drop=True)


def _load_positions_rows(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return int(_read_table(path).shape[0])


def _read_returns_file(path: Path | None) -> pd.Series:
    if path is None or not path.exists():
        return pd.Series(dtype=float, name="benchmark_return")
    frame = _read_table(path)
    date_col = next(
        (col for col in ("trade_date", "date", "period_end", "exit_date") if col in frame), None
    )
    ret_col = next(
        (col for col in ("benchmark_return", "return", "net_return") if col in frame), None
    )
    if date_col is None or ret_col is None:
        raise SystemExit("Benchmark returns file needs a date column and return column.")
    series = pd.Series(
        pd.to_numeric(frame[ret_col], errors="coerce").to_numpy(dtype=float),
        index=_date_series(frame[date_col]),
        name="benchmark_return",
    )
    return series.dropna().sort_index()


def _load_scored(path: Path | None, cfg: dict[str, Any]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    scored = _read_table(path)
    date_col = str(cfg.get("scored_date_col", "trade_date"))
    score_col = str(cfg.get("score_col", "signal_eval"))
    target_col = str(cfg.get("target_col", "future_return"))
    required = {date_col, score_col, target_col}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise SystemExit("Artifact CPCV scored file is missing: " + ", ".join(missing))
    scored = scored.copy()
    scored["trade_date"] = _date_series(scored[date_col])
    postprocess = cfg.get("score_postprocess") or {}
    if postprocess:
        if not isinstance(postprocess, dict):
            raise SystemExit("artifact_cpcv.score_postprocess must be a mapping.")
        scored["signal_eval"] = apply_score_postprocess(
            scored,
            score_col,
            method=str(postprocess.get("method", "none")),
            columns=[str(col) for col in postprocess.get("columns", [])],
            strength=float(postprocess.get("strength", 1.0)),
            min_obs=postprocess.get("min_obs"),
        )
    else:
        scored["signal_eval"] = pd.to_numeric(scored[score_col], errors="coerce")
    scored[target_col] = pd.to_numeric(scored[target_col], errors="coerce")
    return scored.dropna(subset=["trade_date", "signal_eval", target_col])


def _load_inputs(config_path: Path) -> ArtifactCPCVInputs:
    cfg = _artifact_payload(config_path)
    base_dir = config_path.parent
    run_dir = _resolve_path(cfg.get("run_dir"), base_dir=base_dir)
    input_base = run_dir or base_dir
    periods_path = _resolve_path(
        cfg.get("periods_file") or "backtest_periods.csv",
        base_dir=input_base,
    )
    scored_path = _resolve_path(cfg.get("scored_file"), base_dir=input_base)
    positions_path = _resolve_path(
        cfg.get("positions_file") or "positions_by_rebalance.csv",
        base_dir=input_base,
    )
    benchmark_path = _resolve_path(cfg.get("benchmark_returns_file"), base_dir=input_base)
    if periods_path is None:
        raise SystemExit("artifact_cpcv.periods_file is required.")
    return ArtifactCPCVInputs(
        cfg=cfg,
        config_path=config_path,
        run_dir=run_dir,
        periods=_normalize_periods(_read_table(periods_path)),
        scored=_load_scored(scored_path, cfg),
        benchmark_returns=_read_returns_file(benchmark_path),
        positions_rows=_load_positions_rows(positions_path),
    )


def _event_windows(periods: pd.DataFrame) -> dict[pd.Timestamp, cpcv_module.LabelEventWindow]:
    windows: dict[pd.Timestamp, cpcv_module.LabelEventWindow] = {}
    for row in periods.itertuples(index=False):
        windows[pd.Timestamp(row.rebalance_ts)] = cpcv_module.LabelEventWindow(
            signal_date=pd.Timestamp(row.rebalance_ts),
            label_start=pd.Timestamp(row.entry_ts),
            label_end=pd.Timestamp(row.exit_ts),
        )
    return windows


def _period_info(periods: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in periods.to_dict("records"):
        item = dict(row)
        item["rebalance_date"] = pd.Timestamp(item["rebalance_ts"])
        item["entry_date"] = pd.Timestamp(item["entry_ts"])
        item["exit_date"] = pd.Timestamp(item["exit_ts"])
        rows.append(item)
    return rows


def _series_from_periods(periods: pd.DataFrame, column: str, name: str) -> pd.Series:
    if column not in periods.columns:
        return pd.Series(dtype=float, name=name)
    return pd.Series(
        periods[column].to_numpy(dtype=float),
        index=pd.to_datetime(periods["exit_ts"]),
        name=name,
    ).dropna()


def _eval_scored_for_dates(
    inputs: ArtifactCPCVInputs,
    dates: tuple[pd.Timestamp, ...],
) -> pd.DataFrame:
    if inputs.scored.empty:
        return pd.DataFrame()
    target_col = str(inputs.cfg.get("target_col", "future_return"))
    frame = inputs.scored.loc[inputs.scored["trade_date"].isin(set(dates))].copy()
    if target_col != "future_return" and target_col in frame.columns:
        frame["future_return"] = frame[target_col]
    return frame


def _split_result(inputs: ArtifactCPCVInputs, split: cpcv_module.CPCVSplit) -> dict[str, Any]:
    if split.status != "ok":
        return {"status": split.status, "split": split}
    periods = inputs.periods.loc[inputs.periods["rebalance_ts"].isin(set(split.test_dates))].copy()
    if periods.empty:
        return {"status": "insufficient_data", "split": split}
    info = _period_info(periods)
    net = _series_from_periods(periods, "net_return", "net_return")
    gross = _series_from_periods(periods, "gross_return", "gross_return")
    turnover = _series_from_periods(periods, "turnover", "turnover")
    stats = summarize_period_returns(
        net,
        info,
        int(inputs.cfg.get("trading_days_per_year", 252)),
    )
    stats["avg_cost_drag"] = (
        float(periods["total_cost"].mean()) if "total_cost" in periods else np.nan
    )
    bench, _ = build_benchmark_series(
        None,
        "close",
        "close",
        info,
        benchmark_return_series=inputs.benchmark_returns,
    )
    active_stats, active = ({}, pd.Series(dtype=float, name="active_return"))
    if not bench.empty:
        active_stats, active = summarize_active_returns(
            net,
            bench,
            stats.get("periods_per_year", np.nan),
        )
    return {
        "status": "ok",
        "split": split,
        "eval_scored": _eval_scored_for_dates(inputs, split.test_dates),
        "net_series": net,
        "gross_series": gross,
        "turnover_series": turnover,
        "benchmark_series": bench,
        "active_series": active,
        "period_info": info,
        "bt_stats": stats,
        "active_stats": active_stats,
    }


def _build_splits(
    inputs: ArtifactCPCVInputs,
    *,
    n_groups: int,
    test_groups: int,
    embargo_days: int | None,
) -> list[cpcv_module.CPCVSplit]:
    dates = cpcv_module._as_date_tuple(inputs.periods["rebalance_ts"])
    _groups, splits = cpcv_module.build_cpcv_splits(
        dates,
        n_groups=n_groups,
        test_groups=test_groups,
        event_windows=_event_windows(inputs.periods),
        embargo_days=int(
            embargo_days if embargo_days is not None else inputs.cfg.get("embargo_days", 0)
        ),
        fallback_gap_steps=int(inputs.cfg.get("fallback_gap_steps", 0) or 0),
        min_train_dates=int(inputs.cfg.get("min_train_dates", 1)),
        min_test_dates=int(inputs.cfg.get("min_test_dates", 1)),
    )
    return splits


def _path_rows(
    inputs: ArtifactCPCVInputs,
    split_results: dict[int, dict[str, Any]],
    valid_splits: list[cpcv_module.CPCVSplit],
    *,
    n_groups: int,
    test_groups: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    returns: list[dict[str, Any]] = []
    paths = cpcv_module.build_cpcv_paths(valid_splits, n_groups=n_groups, test_groups=test_groups)
    for path_id, path_splits in enumerate(paths, start=1):
        result_rows = [split_results[split.split_id] for split in path_splits]
        row, return_rows = cpcv_module._path_metric_row(
            path_id,
            result_rows,
            target_col=str(inputs.cfg.get("target_col", "future_return")),
            n_quantiles=int(inputs.cfg.get("n_quantiles", 5)),
            trading_days_per_year=int(inputs.cfg.get("trading_days_per_year", 252)),
        )
        if row is not None:
            metrics.append(row)
            returns.extend(return_rows)
    return metrics, returns


def _summary(
    inputs: ArtifactCPCVInputs,
    *,
    n_groups: int,
    test_groups: int,
    splits: list[cpcv_module.CPCVSplit],
    valid_splits: list[cpcv_module.CPCVSplit],
    path_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    dates = cpcv_module._as_date_tuple(inputs.periods["rebalance_ts"])
    return {
        "schema": "artifact_cpcv_summary.v1",
        "artifact_mode": "frozen_materialized_periods",
        "path_metric_note": (
            "Path returns reuse fixed materialized period returns; this audits frozen artifact "
            "period coverage and path aggregation, not model retraining instability."
        ),
        "run_dir": str(inputs.run_dir) if inputs.run_dir else None,
        "positions_rows": inputs.positions_rows,
        "scored_available": not inputs.scored.empty,
        "n_groups": int(n_groups),
        "test_groups": int(test_groups),
        "split_count": len(splits),
        "valid_split_count": len(valid_splits),
        "path_count": cpcv_module.expected_cpcv_path_count(n_groups, test_groups),
        "eligible_date_count": len(dates),
        "eligible_start": cpcv_module._format_date(dates[0]) if dates else None,
        "eligible_end": cpcv_module._format_date(dates[-1]) if dates else None,
        "purge_mode": "event_window",
        "embargo_days": int(inputs.cfg.get("embargo_days", 0) or 0),
        **cpcv_module._summarize_cpcv(path_metrics),
    }


def run_artifact_cpcv(
    config_path: str | Path,
    *,
    n_groups: int,
    test_groups: int,
    embargo_days: int | None,
    out_dir: Path,
) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    inputs = _load_inputs(path)
    splits = _build_splits(
        inputs,
        n_groups=n_groups,
        test_groups=test_groups,
        embargo_days=embargo_days,
    )
    split_results = {split.split_id: _split_result(inputs, split) for split in splits}
    valid_splits = [
        split for split in splits if split_results.get(split.split_id, {}).get("status") == "ok"
    ]
    path_metrics, path_returns = _path_rows(
        inputs,
        split_results,
        valid_splits,
        n_groups=n_groups,
        test_groups=test_groups,
    )
    summary = _summary(
        inputs,
        n_groups=n_groups,
        test_groups=test_groups,
        splits=splits,
        valid_splits=valid_splits,
        path_metrics=path_metrics,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([cpcv_module._split_to_row(split) for split in splits]).to_csv(
        out_dir / "cpcv_splits.csv",
        index=False,
    )
    pd.DataFrame(path_returns).to_csv(out_dir / "cpcv_path_returns.csv", index=False)
    pd.DataFrame(path_metrics).to_csv(out_dir / "cpcv_path_metrics.csv", index=False)
    (out_dir / "cpcv_summary.json").write_text(
        json.dumps(cpcv_module._to_jsonable(summary), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary
