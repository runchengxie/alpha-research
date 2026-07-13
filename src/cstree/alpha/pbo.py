from __future__ import annotations

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from .research_artifacts import write_strict_json


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


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _sharpe(values: pd.Series, *, periods_per_year: int | None = None) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.shape[0] < 2:
        return np.nan
    std = float(series.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        return np.nan
    scale = math.sqrt(periods_per_year) if periods_per_year and periods_per_year > 0 else 1.0
    return float(series.mean() / std * scale)


def _max_drawdown(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if series.empty:
        return np.nan
    equity = (1.0 + series).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _candidate_columns(
    frame: pd.DataFrame,
    *,
    date_col: str,
    columns: list[str] | None,
) -> list[str]:
    if columns:
        return [column for column in columns if column in frame.columns]
    out: list[str] = []
    for column in frame.columns:
        if column == date_col:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            out.append(str(column))
    return out


def _date_groups(dates: pd.Series, n_groups: int) -> list[pd.Index]:
    unique_dates = pd.Index(pd.to_datetime(dates).dropna().unique()).sort_values()
    if n_groups < 2:
        raise SystemExit("--n-groups must be >= 2.")
    if unique_dates.shape[0] < n_groups:
        raise SystemExit("--n-groups cannot exceed the number of available dates.")
    groups: list[pd.Index] = []
    base, extra = divmod(unique_dates.shape[0], n_groups)
    cursor = 0
    for group_id in range(n_groups):
        size = base + (1 if group_id < extra else 0)
        groups.append(unique_dates[cursor : cursor + size])
        cursor += size
    return groups


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-12), 1.0 - 1e-12)
    return float(math.log(clipped / (1.0 - clipped)))


def _deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    selected_sharpe: float,
    n_trials: int,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    series = pd.to_numeric(returns, errors="coerce").dropna()
    n_obs = int(series.shape[0])
    if n_obs < 3 or not np.isfinite(selected_sharpe):
        return {
            "dsr": None,
            "dsr_z": None,
            "n_obs": n_obs,
            "n_trials": int(n_trials),
            "expected_max_sharpe": None,
        }
    scale = math.sqrt(periods_per_year) if periods_per_year and periods_per_year > 0 else 1.0
    periodic_sr = selected_sharpe / scale
    skew = float(series.skew()) if n_obs >= 3 else 0.0
    kurtosis = float(series.kurt()) + 3.0 if n_obs >= 4 else 3.0
    variance = (
        1.0 - skew * periodic_sr + ((kurtosis - 1.0) / 4.0) * periodic_sr * periodic_sr
    ) / max(1, n_obs - 1)
    if not np.isfinite(variance) or variance <= 0:
        return {
            "dsr": None,
            "dsr_z": None,
            "n_obs": n_obs,
            "n_trials": int(n_trials),
            "expected_max_sharpe": None,
        }
    n_eff = max(1, int(n_trials))
    if n_eff <= 1:
        expected_max_periodic = 0.0
    else:
        normal = NormalDist()
        gamma = 0.5772156649015329
        z_1 = normal.inv_cdf(1.0 - 1.0 / n_eff)
        z_2 = normal.inv_cdf(1.0 - 1.0 / (n_eff * math.e))
        expected_max_periodic = math.sqrt(variance) * ((1.0 - gamma) * z_1 + gamma * z_2)
    z_value = (periodic_sr - expected_max_periodic) / math.sqrt(variance)
    return {
        "dsr": float(NormalDist().cdf(z_value)),
        "dsr_z": float(z_value),
        "n_obs": n_obs,
        "n_trials": int(n_eff),
        "expected_max_sharpe": float(expected_max_periodic * scale),
        "return_skew": skew,
        "return_kurtosis": kurtosis,
    }


def _validated_pbo_inputs(
    frame: pd.DataFrame,
    *,
    date_col: str,
    candidate_cols: list[str] | None,
    n_groups: int,
    test_groups: int | None,
) -> tuple[pd.DataFrame, list[str], list[pd.Index], int]:
    if date_col not in frame.columns:
        raise SystemExit(f"Missing date column: {date_col}")
    data = frame.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    columns = _candidate_columns(data, date_col=date_col, columns=candidate_cols)
    if len(columns) < 2:
        raise SystemExit("PBO requires at least two candidate return columns.")
    groups = _date_groups(data[date_col], n_groups)
    resolved_test_groups = n_groups // 2 if test_groups is None else test_groups
    if resolved_test_groups <= 0 or resolved_test_groups >= n_groups:
        raise SystemExit("--test-groups must satisfy 1 <= test_groups < n_groups.")
    return data, columns, groups, resolved_test_groups


def _select_best_sharpe_column(
    columns: list[str],
    sharpes: dict[str, float],
) -> str:
    return max(
        columns,
        key=lambda column: (
            sharpes[column] if np.isfinite(sharpes[column]) else -np.inf,
            column,
        ),
    )


def _build_pbo_split_row(
    *,
    split_id: int,
    train_group_ids: tuple[int, ...],
    test_group_ids: tuple[int, ...],
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    periods_per_year: int | None,
) -> dict[str, Any]:
    train_sharpes = {
        column: _sharpe(train[column], periods_per_year=periods_per_year) for column in columns
    }
    test_sharpes = {
        column: _sharpe(test[column], periods_per_year=periods_per_year) for column in columns
    }
    selected = _select_best_sharpe_column(columns, train_sharpes)
    selected_oos = test_sharpes[selected]
    finite_oos = [value for value in test_sharpes.values() if np.isfinite(value)]
    if finite_oos and np.isfinite(selected_oos):
        relative_rank = float(sum(value <= selected_oos for value in finite_oos) / len(finite_oos))
        logit_rank = _logit(relative_rank)
    else:
        relative_rank = np.nan
        logit_rank = np.nan
    return {
        "split_id": split_id,
        "train_groups": "|".join(str(idx) for idx in train_group_ids),
        "test_groups": "|".join(str(idx) for idx in test_group_ids),
        "selected_candidate": selected,
        "selected_train_sharpe": train_sharpes[selected],
        "selected_oos_sharpe": selected_oos,
        "selected_oos_relative_rank": relative_rank,
        "logit_oos_rank": logit_rank,
        "is_overfit": bool(np.isfinite(logit_rank) and logit_rank < 0.0),
    }


def _build_pbo_rows(
    data: pd.DataFrame,
    *,
    date_col: str,
    columns: list[str],
    groups: list[pd.Index],
    n_groups: int,
    test_groups: int,
    periods_per_year: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_id, train_group_ids in enumerate(
        combinations(range(n_groups), n_groups - test_groups)
    ):
        train_group_set = set(train_group_ids)
        test_group_ids = tuple(idx for idx in range(n_groups) if idx not in train_group_set)
        train_dates = pd.Index([]).append([groups[idx] for idx in train_group_ids])
        test_dates = pd.Index([]).append([groups[idx] for idx in test_group_ids])
        rows.append(
            _build_pbo_split_row(
                split_id=split_id,
                train_group_ids=train_group_ids,
                test_group_ids=test_group_ids,
                train=data[data[date_col].isin(train_dates)],
                test=data[data[date_col].isin(test_dates)],
                columns=columns,
                periods_per_year=periods_per_year,
            )
        )
    return rows


def _finite_row_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [row[field] for row in rows if np.isfinite(row[field])]


def _build_pbo_summary(
    data: pd.DataFrame,
    *,
    rows: list[dict[str, Any]],
    columns: list[str],
    n_groups: int,
    test_groups: int,
    periods_per_year: int | None,
) -> dict[str, Any]:
    full_sharpes = {
        column: _sharpe(data[column], periods_per_year=periods_per_year) for column in columns
    }
    selected_full = _select_best_sharpe_column(columns, full_sharpes)
    dsr = _deflated_sharpe_ratio(
        data[selected_full],
        selected_sharpe=full_sharpes[selected_full],
        n_trials=len(columns),
        periods_per_year=periods_per_year,
    )
    finite_logits = _finite_row_values(rows, "logit_oos_rank")
    finite_oos = _finite_row_values(rows, "selected_oos_sharpe")
    return {
        "schema_version": 1,
        "n_groups": int(n_groups),
        "test_groups": int(test_groups),
        "split_count": len(rows),
        "candidate_count": len(columns),
        "n_trials": len(columns),
        "candidate_columns": columns,
        "pbo": float(np.mean([value < 0.0 for value in finite_logits])) if finite_logits else None,
        "logit_oos_rank_mean": float(np.mean(finite_logits)) if finite_logits else None,
        "selected_oos_sharpe_mean": float(np.mean(finite_oos)) if finite_oos else None,
        "selected_oos_sharpe_p25": float(np.quantile(finite_oos, 0.25)) if finite_oos else None,
        "selected_candidate": selected_full,
        "selected_sharpe": _to_float(full_sharpes[selected_full]),
        "selected_max_drawdown": _to_float(_max_drawdown(data[selected_full])),
        "dsr": _to_float(dsr["dsr"]),
        "dsr_z": _to_float(dsr["dsr_z"]),
        "dsr_n_trials": dsr["n_trials"],
        "dsr_n_obs": dsr["n_obs"],
        "dsr_expected_max_sharpe": _to_float(dsr["expected_max_sharpe"]),
    }


def compute_pbo_report(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    candidate_cols: list[str] | None = None,
    n_groups: int = 8,
    test_groups: int | None = None,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    data, columns, groups, resolved_test_groups = _validated_pbo_inputs(
        frame,
        date_col=date_col,
        candidate_cols=candidate_cols,
        n_groups=n_groups,
        test_groups=test_groups,
    )
    rows = _build_pbo_rows(
        data,
        date_col=date_col,
        columns=columns,
        groups=groups,
        n_groups=n_groups,
        test_groups=resolved_test_groups,
        periods_per_year=periods_per_year,
    )
    summary = _build_pbo_summary(
        data,
        rows=rows,
        columns=columns,
        n_groups=n_groups,
        test_groups=resolved_test_groups,
        periods_per_year=periods_per_year,
    )
    return {"summary": summary, "rows": rows}


def _write_report(report: dict[str, Any], *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = report["rows"]
    with (out_dir / "pbo_splits.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_strict_json(out_dir / "pbo_summary.json", report["summary"])


def add_pbo_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--returns", required=True, help="CSV/parquet matrix of candidate returns.")
    parser.add_argument("--date-col", default="date", help="Date column in the returns matrix.")
    parser.add_argument(
        "--candidate-col",
        action="append",
        default=None,
        help="Candidate return column. Can be repeated; default uses all numeric non-date columns.",
    )
    parser.add_argument("--n-groups", type=int, default=8, help="Number of CSCV groups.")
    parser.add_argument("--test-groups", type=int, default=None, help="Number of held-out groups.")
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=None,
        help="Annualization periods for Sharpe. Omit to use per-period Sharpe.",
    )
    parser.add_argument("--out", default="artifacts/reports/pbo", help="Output report directory.")
    return parser


def run(args: argparse.Namespace) -> int:
    returns_path = _resolve_path(args.returns)
    if returns_path is None or not returns_path.exists():
        raise SystemExit(f"Returns file not found: {args.returns}")
    frame = _read_frame(returns_path)
    report = compute_pbo_report(
        frame,
        date_col=args.date_col,
        candidate_cols=args.candidate_col,
        n_groups=args.n_groups,
        test_groups=args.test_groups,
        periods_per_year=args.periods_per_year,
    )
    out_dir = _resolve_path(args.out)
    assert out_dir is not None
    _write_report(report, out_dir=out_dir)
    return 0
