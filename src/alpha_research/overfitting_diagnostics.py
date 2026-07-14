from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import daily_ic_series, summarize_ic


def _resolve_path(path_text: str | Path | None) -> Path | None:
    if path_text is None:
        return None
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, default=str),
        encoding="utf-8",
    )


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _as_dates(values: Any) -> pd.Series:
    return pd.Series(pd.to_datetime(values, errors="coerce")).dt.normalize()


def uniqueness_report(
    frame: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    label_start_col: str | None = None,
    label_end_col: str | None = None,
    horizon_days: int = 1,
    bootstrap_samples: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    if date_col not in frame.columns:
        raise SystemExit(f"Missing date column: {date_col}")
    data = frame.copy()
    signal_dates = _as_dates(data[date_col])
    if label_start_col and label_start_col in data.columns:
        starts = _as_dates(data[label_start_col])
    else:
        starts = signal_dates
    if label_end_col and label_end_col in data.columns:
        ends = _as_dates(data[label_end_col])
    else:
        ends = signal_dates + pd.to_timedelta(max(0, int(horizon_days)), unit="D")
    valid = pd.DataFrame({"signal_date": signal_dates, "label_start": starts, "label_end": ends})
    valid = valid.dropna().reset_index(names="event_id")
    if valid.empty:
        raise SystemExit("No valid event windows for uniqueness diagnostics.")

    calendar = pd.date_range(valid["label_start"].min(), valid["label_end"].max(), freq="D")
    concurrency = pd.Series(0.0, index=calendar)
    for row in valid.itertuples(index=False):
        concurrency.loc[row.label_start : row.label_end] += 1.0

    rows: list[dict[str, Any]] = []
    for row in valid.itertuples(index=False):
        active = concurrency.loc[row.label_start : row.label_end]
        uniqueness = float((1.0 / active.replace(0.0, np.nan)).mean())
        rows.append(
            {
                "event_id": int(row.event_id),
                "signal_date": row.signal_date.date().isoformat(),
                "label_start": row.label_start.date().isoformat(),
                "label_end": row.label_end.date().isoformat(),
                "event_length": int(active.shape[0]),
                "avg_concurrency": float(active.mean()),
                "uniqueness": uniqueness,
                "sample_weight": uniqueness,
            }
        )

    bootstrap: list[int] = []
    if bootstrap_samples > 0:
        weights = np.asarray([row["uniqueness"] for row in rows], dtype=float)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(rows), dtype=float)
        weights = weights / weights.sum()
        rng = np.random.default_rng(seed)
        bootstrap = [
            int(idx)
            for idx in rng.choice(
                len(rows),
                size=bootstrap_samples,
                replace=True,
                p=weights,
            )
        ]

    summary = {
        "schema_version": 1,
        "event_count": len(rows),
        "average_uniqueness": float(np.mean([row["uniqueness"] for row in rows])),
        "mean_concurrency": float(concurrency.mean()),
        "max_concurrency": float(concurrency.max()),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_event_ids": bootstrap,
    }
    return {"summary": summary, "rows": rows}


def _ic_stats(frame: pd.DataFrame, *, score_col: str, target_col: str) -> dict[str, Any]:
    valid = frame.dropna(subset=["trade_date", score_col, target_col])
    if valid.empty:
        return {"n": 0, "ic_mean": None, "ic_ir": None}
    ic = daily_ic_series(valid, target_col, score_col)
    stats = summarize_ic(ic)
    return {
        "n": int(stats["n"]),
        "ic_mean": _to_float(stats["mean"]),
        "ic_ir": _to_float(stats["ir"]),
    }


def negative_control_report(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target_col: str,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    shifted_periods: int = 1,
    random_features: int = 3,
    random_universe_frac: float = 0.5,
    sentinel_features: list[str] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    missing = [
        column for column in [date_col, target_col, *features] if column not in frame.columns
    ]
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing))
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data[date_col])
    rows: list[dict[str, Any]] = []

    proxy = "__feature_proxy"
    data[proxy] = data[features].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    rows.append(
        {
            "control": "baseline_feature_proxy",
            **_ic_stats(data, score_col=proxy, target_col=target_col),
        }
    )

    shifted = data.copy()
    if symbol_col in shifted.columns:
        shifted["__shifted_target"] = shifted.groupby(symbol_col)[target_col].shift(
            -int(shifted_periods)
        )
    else:
        shifted["__shifted_target"] = shifted[target_col].shift(-int(shifted_periods))
    rows.append(
        {
            "control": "shifted_label",
            "shifted_periods": int(shifted_periods),
            **_ic_stats(shifted, score_col=proxy, target_col="__shifted_target"),
        }
    )

    rng = np.random.default_rng(seed)
    for idx in range(max(0, int(random_features))):
        control_col = f"__random_feature_{idx}"
        random_frame = data.copy()
        random_frame[control_col] = rng.normal(size=random_frame.shape[0])
        rows.append(
            {
                "control": "random_feature",
                "feature": control_col,
                **_ic_stats(random_frame, score_col=control_col, target_col=target_col),
            }
        )

    if 0.0 < random_universe_frac < 1.0 and symbol_col in data.columns:
        sampled_symbols = (
            data[[symbol_col]]
            .drop_duplicates()
            .sample(frac=random_universe_frac, random_state=seed)[symbol_col]
        )
        sampled = data[data[symbol_col].isin(sampled_symbols)].copy()
        rows.append(
            {
                "control": "random_universe",
                "sampled_symbol_count": int(sampled_symbols.shape[0]),
                "universe_frac": float(random_universe_frac),
                **_ic_stats(sampled, score_col=proxy, target_col=target_col),
            }
        )

    for feature in sentinel_features or []:
        if feature not in data.columns:
            rows.append(
                {
                    "control": "future_feature_sentinel",
                    "feature": feature,
                    "status": "missing",
                }
            )
            continue
        rows.append(
            {
                "control": "future_feature_sentinel",
                "feature": feature,
                "status": "present",
                **_ic_stats(data, score_col=feature, target_col=target_col),
            }
        )
    return rows


def _sharpe(series: pd.Series, periods_per_year: int | None) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.shape[0] < 2:
        return np.nan
    std = float(values.std(ddof=1))
    if std <= 0 or not np.isfinite(std):
        return np.nan
    scale = math.sqrt(periods_per_year) if periods_per_year and periods_per_year > 0 else 1.0
    return float(values.mean() / std * scale)


def _max_drawdown(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if values.empty:
        return np.nan
    equity = (1.0 + values).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def scenario_backtest_report(
    frame: pd.DataFrame,
    *,
    return_col: str,
    n_scenarios: int = 100,
    block_size: int = 5,
    periods_per_year: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    if return_col not in frame.columns:
        raise SystemExit(f"Missing return column: {return_col}")
    returns = pd.to_numeric(frame[return_col], errors="coerce").dropna().reset_index(drop=True)
    if returns.empty:
        raise SystemExit("No valid returns for scenario backtest.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    block = max(1, int(block_size))
    for scenario_id in range(max(1, int(n_scenarios))):
        sampled: list[float] = []
        while len(sampled) < len(returns):
            start = int(rng.integers(0, len(returns)))
            sampled.extend(returns.iloc[start : min(start + block, len(returns))].tolist())
        scenario = pd.Series(sampled[: len(returns)], dtype=float)
        rows.append(
            {
                "scenario_id": scenario_id,
                "total_return": float((1.0 + scenario).prod() - 1.0),
                "mean_return": float(scenario.mean()),
                "sharpe": _sharpe(scenario, periods_per_year),
                "max_drawdown": _max_drawdown(scenario),
            }
        )
    sharpe_values = [row["sharpe"] for row in rows if np.isfinite(row["sharpe"])]
    drawdowns = [row["max_drawdown"] for row in rows if np.isfinite(row["max_drawdown"])]
    summary = {
        "schema_version": 1,
        "scenario_count": len(rows),
        "block_size": block,
        "base_sharpe": _sharpe(returns, periods_per_year),
        "base_max_drawdown": _max_drawdown(returns),
        "scenario_sharpe_p05": float(np.quantile(sharpe_values, 0.05)) if sharpe_values else None,
        "scenario_sharpe_p50": float(np.quantile(sharpe_values, 0.50)) if sharpe_values else None,
        "scenario_max_drawdown_p05": float(np.quantile(drawdowns, 0.05)) if drawdowns else None,
    }
    return {"summary": summary, "rows": rows}


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_freeze_manifest(
    *,
    run_dir: Path,
    targets_file: Path | None = None,
    promotion_gate_report: Path | None = None,
    lifecycle_stage: str = "frozen_candidate",
    paper_start_date: str | None = None,
    paper_end_date: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    files = {
        "summary": run_dir / "summary.json",
        "config": run_dir / "config.used.yml",
        "inputs_lock": run_dir / "inputs.lock.json",
        "targets": targets_file,
        "promotion_gate_report": promotion_gate_report,
    }
    return {
        "schema_version": 1,
        "frozen_at": datetime.now(UTC).isoformat(),
        "lifecycle_stage": lifecycle_stage,
        "paper_start_date": paper_start_date,
        "paper_end_date": paper_end_date,
        "run_dir": str(run_dir),
        "note": note,
        "files": {
            name: {
                "path": str(path) if path else None,
                "exists": bool(path and path.exists()),
                "sha256": _file_sha256(path) if path else None,
            }
            for name, path in files.items()
        },
    }


def add_overfitting_diagnostics_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    subparsers = parser.add_subparsers(dest="mode", required=True)

    uniqueness = subparsers.add_parser("uniqueness", help="Compute label event uniqueness weights.")
    uniqueness.add_argument("--input", required=True, help="CSV/parquet event data.")
    uniqueness.add_argument("--date-col", default="trade_date")
    uniqueness.add_argument("--label-start-col", default=None)
    uniqueness.add_argument("--label-end-col", default=None)
    uniqueness.add_argument("--horizon-days", type=int, default=1)
    uniqueness.add_argument("--bootstrap-samples", type=int, default=0)
    uniqueness.add_argument("--seed", type=int, default=42)
    uniqueness.add_argument("--output", default="artifacts/reports/event_uniqueness.csv")
    uniqueness.add_argument("--output-json", default="artifacts/reports/event_uniqueness.json")

    controls = subparsers.add_parser(
        "negative-controls",
        help="Run shifted-label/random negative controls.",
    )
    controls.add_argument("--input", required=True, help="CSV/parquet feature data.")
    controls.add_argument(
        "--feature",
        action="append",
        required=True,
        help="Feature column. Can be repeated.",
    )
    controls.add_argument("--target-col", required=True)
    controls.add_argument("--date-col", default="trade_date")
    controls.add_argument("--symbol-col", default="symbol")
    controls.add_argument("--shifted-periods", type=int, default=1)
    controls.add_argument("--random-features", type=int, default=3)
    controls.add_argument("--random-universe-frac", type=float, default=0.5)
    controls.add_argument("--sentinel-feature", action="append", default=None)
    controls.add_argument("--seed", type=int, default=42)
    controls.add_argument("--output", default="artifacts/reports/negative_controls.csv")

    scenario = subparsers.add_parser("scenario-backtest", help="Bootstrap scenario return paths.")
    scenario.add_argument("--returns", required=True, help="CSV/parquet returns file.")
    scenario.add_argument("--return-col", required=True)
    scenario.add_argument("--n-scenarios", type=int, default=100)
    scenario.add_argument("--block-size", type=int, default=5)
    scenario.add_argument("--periods-per-year", type=int, default=None)
    scenario.add_argument("--seed", type=int, default=42)
    scenario.add_argument("--out", default="artifacts/reports/scenario_backtest")

    freeze = subparsers.add_parser(
        "candidate-freeze",
        help="Freeze a candidate/paper trading manifest.",
    )
    freeze.add_argument("--run-dir", required=True)
    freeze.add_argument("--targets-file", default=None)
    freeze.add_argument("--promotion-gate-report", default=None)
    freeze.add_argument("--lifecycle-stage", default="frozen_candidate")
    freeze.add_argument("--paper-start-date", default=None)
    freeze.add_argument("--paper-end-date", default=None)
    freeze.add_argument("--note", default=None)
    freeze.add_argument("--output-json", default="artifacts/reports/candidate_freeze_manifest.json")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.mode == "uniqueness":
        input_path = _resolve_path(args.input)
        if input_path is None or not input_path.exists():
            raise SystemExit(f"Input file not found: {args.input}")
        report = uniqueness_report(
            _read_frame(input_path),
            date_col=args.date_col,
            label_start_col=args.label_start_col,
            label_end_col=args.label_end_col,
            horizon_days=args.horizon_days,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        output = _resolve_path(args.output)
        output_json = _resolve_path(args.output_json)
        assert output is not None and output_json is not None
        _write_rows(report["rows"], output)
        _write_json(report["summary"], output_json)
        return 0

    if args.mode == "negative-controls":
        input_path = _resolve_path(args.input)
        if input_path is None or not input_path.exists():
            raise SystemExit(f"Input file not found: {args.input}")
        rows = negative_control_report(
            _read_frame(input_path),
            features=args.feature,
            target_col=args.target_col,
            date_col=args.date_col,
            symbol_col=args.symbol_col,
            shifted_periods=args.shifted_periods,
            random_features=args.random_features,
            random_universe_frac=args.random_universe_frac,
            sentinel_features=args.sentinel_feature,
            seed=args.seed,
        )
        output = _resolve_path(args.output)
        assert output is not None
        _write_rows(rows, output)
        return 0

    if args.mode == "scenario-backtest":
        returns_path = _resolve_path(args.returns)
        if returns_path is None or not returns_path.exists():
            raise SystemExit(f"Returns file not found: {args.returns}")
        report = scenario_backtest_report(
            _read_frame(returns_path),
            return_col=args.return_col,
            n_scenarios=args.n_scenarios,
            block_size=args.block_size,
            periods_per_year=args.periods_per_year,
            seed=args.seed,
        )
        out_dir = _resolve_path(args.out)
        assert out_dir is not None
        _write_rows(report["rows"], out_dir / "scenario_paths.csv")
        _write_json(report["summary"], out_dir / "scenario_summary.json")
        return 0

    run_dir = _resolve_path(args.run_dir)
    if run_dir is None or not run_dir.exists():
        raise SystemExit(f"Run directory not found: {args.run_dir}")
    manifest = candidate_freeze_manifest(
        run_dir=run_dir,
        targets_file=_resolve_path(args.targets_file),
        promotion_gate_report=_resolve_path(args.promotion_gate_report),
        lifecycle_stage=args.lifecycle_stage,
        paper_start_date=args.paper_start_date,
        paper_end_date=args.paper_end_date,
        note=args.note,
    )
    output_json = _resolve_path(args.output_json)
    assert output_json is not None
    _write_json(manifest, output_json)
    return 0
