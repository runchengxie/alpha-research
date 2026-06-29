from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

_RECENCY_WINDOW_PATTERN = re.compile(r"^\s*(\d+)\s*([dwm])\s*$", re.IGNORECASE)
DEFAULT_RECENCY_WINDOWS = ["6m", "1m", "1w"]
RECENCY_DIAGNOSTIC_COLUMNS = [
    "window",
    "role",
    "status",
    "start",
    "end",
    "ic_count",
    "ic_mean",
    "ic_ir",
    "ic_positive_ratio",
    "return_count",
    "total_return",
    "ann_return",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "active_count",
    "active_total_return",
    "active_information_ratio",
    "turnover_count",
    "avg_turnover",
]


def normalize_recency_windows(
    value: object | None,
    default: list[str] | None = None,
) -> list[str]:
    if value is None:
        raw_items: object = default or DEFAULT_RECENCY_WINDOWS
    elif isinstance(value, bool):
        raw_items = (default or DEFAULT_RECENCY_WINDOWS) if value else []
    elif isinstance(value, (str, int, float)):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raise SystemExit(
            "eval.recency.windows must contain positive duration labels like 6m, 1m, 1w, or 5d."
        )

    windows: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        parsed = parse_recency_window(item)
        if parsed is None:
            raise SystemExit(
                "eval.recency.windows must contain positive duration labels like 6m, 1m, 1w, or 5d."
            )
        label, _, _ = parsed
        if label not in seen:
            windows.append(label)
            seen.add(label)
    return windows


def parse_recency_window(value: object) -> tuple[str, int, str] | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        amount = int(value)
        return (f"{amount}m", amount, "m") if amount > 0 else None

    text = str(value).strip().lower()
    match = _RECENCY_WINDOW_PATTERN.fullmatch(text)
    if match is None:
        return None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if amount <= 0:
        return None
    return f"{amount}{unit}", amount, unit


def build_recency_diagnostics(
    *,
    window_labels: list[str],
    ic_series: pd.Series | None = None,
    returns: pd.Series | None = None,
    active_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    periods_per_year: float | None = None,
) -> pd.DataFrame:
    windows = normalize_recency_windows(window_labels, default=[])
    if not windows:
        return pd.DataFrame(columns=RECENCY_DIAGNOSTIC_COLUMNS)

    prepared = {
        "ic": _prepare_recency_series(ic_series),
        "returns": _prepare_recency_series(returns),
        "active_returns": _prepare_recency_series(active_returns),
        "turnover": _prepare_recency_series(turnover),
    }
    end = _latest_recency_end(prepared.values())
    if end is None:
        return pd.DataFrame(columns=RECENCY_DIAGNOSTIC_COLUMNS)

    rows = []
    for label in windows:
        parsed = parse_recency_window(label)
        if parsed is None:
            continue
        start = _recency_window_start(end, amount=parsed[1], unit=parsed[2])
        row = _recency_window_row(
            label=parsed[0],
            start=start,
            end=end,
            ic=_slice_recency_window(prepared["ic"], start, end),
            returns=_slice_recency_window(prepared["returns"], start, end),
            active_returns=_slice_recency_window(prepared["active_returns"], start, end),
            turnover=_slice_recency_window(prepared["turnover"], start, end),
            periods_per_year=periods_per_year,
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=RECENCY_DIAGNOSTIC_COLUMNS)


def _prepare_recency_series(series: pd.Series | None) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    work = series.copy()
    work.index = pd.to_datetime(work.index, errors="coerce")
    work = work[work.index.notna()]
    work = pd.to_numeric(work, errors="coerce")
    work = work[work.notna()].astype(float).sort_index()
    return work


def _latest_recency_end(series_list: Iterable[pd.Series]) -> pd.Timestamp | None:
    ends = [
        series.index.max()
        for series in series_list
        if isinstance(series, pd.Series) and not series.empty
    ]
    if not ends:
        return None
    return pd.to_datetime(max(ends))


def _recency_window_start(end: pd.Timestamp, *, amount: int, unit: str) -> pd.Timestamp:
    if unit == "m":
        return end - pd.DateOffset(months=amount)
    if unit == "w":
        return end - pd.Timedelta(weeks=amount)
    return end - pd.Timedelta(days=amount)


def _slice_recency_window(
    series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    if series.empty:
        return series
    return series[(series.index >= start) & (series.index <= end)]


def _recency_window_row(
    *,
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ic: pd.Series,
    returns: pd.Series,
    active_returns: pd.Series,
    turnover: pd.Series,
    periods_per_year: float | None,
) -> dict[str, object]:
    ic_summary = _mean_ir_summary(ic, annualize=False)
    return_summary = _return_summary(returns, periods_per_year=periods_per_year)
    active_summary = _mean_ir_summary(
        active_returns,
        annualize=True,
        periods_per_year=periods_per_year,
    )
    turnover_summary = _mean_ir_summary(turnover, annualize=False)

    return {
        "window": label,
        "role": _recency_role(label),
        "status": _recency_status(
            ic_count=ic_summary["count"],
            return_count=return_summary["count"],
            active_count=active_summary["count"],
            turnover_count=turnover_summary["count"],
        ),
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "ic_count": ic_summary["count"],
        "ic_mean": ic_summary["mean"],
        "ic_ir": ic_summary["ir"],
        "ic_positive_ratio": ic_summary["positive_ratio"],
        "return_count": return_summary["count"],
        "total_return": return_summary["total_return"],
        "ann_return": return_summary["ann_return"],
        "ann_vol": return_summary["ann_vol"],
        "sharpe": return_summary["sharpe"],
        "max_drawdown": return_summary["max_drawdown"],
        "active_count": active_summary["count"],
        "active_total_return": _compound_return(active_returns),
        "active_information_ratio": active_summary["ir"],
        "turnover_count": turnover_summary["count"],
        "avg_turnover": turnover_summary["mean"],
    }


def _recency_role(label: str) -> str:
    if label == "6m":
        return "current_effectiveness"
    if label == "1m":
        return "watch_signal"
    if label == "1w":
        return "monitoring_only"
    return "diagnostic"


def _recency_status(
    *,
    ic_count: int,
    return_count: int,
    active_count: int,
    turnover_count: int,
) -> str:
    if max(ic_count, return_count, active_count, turnover_count) <= 0:
        return "empty"
    if max(ic_count, return_count, active_count) < 2:
        return "limited_sample"
    return "ok"


def _mean_ir_summary(
    series: pd.Series,
    *,
    annualize: bool,
    periods_per_year: float | None = None,
) -> dict[str, float | int]:
    count = int(series.shape[0])
    if count <= 0:
        return {"count": 0, "mean": np.nan, "ir": np.nan, "positive_ratio": np.nan}
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if count > 1 else np.nan
    scale = _annualization_scale(periods_per_year) if annualize else 1.0
    ir = mean / std * scale if np.isfinite(std) and std > 0 else np.nan
    positive_ratio = float(series.gt(0).mean())
    return {"count": count, "mean": mean, "ir": ir, "positive_ratio": positive_ratio}


def _return_summary(
    returns: pd.Series,
    *,
    periods_per_year: float | None,
) -> dict[str, float | int]:
    count = int(returns.shape[0])
    total_return = _compound_return(returns)
    if count <= 0:
        return {
            "count": 0,
            "total_return": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    ppy = _valid_periods_per_year(periods_per_year)
    ann_return = np.nan
    ann_vol = np.nan
    sharpe = np.nan
    if np.isfinite(ppy):
        if np.isfinite(total_return) and total_return > -1.0:
            ann_return = float((1.0 + total_return) ** (ppy / count) - 1.0)
        if count > 1:
            std = float(returns.std(ddof=1))
            ann_vol = float(std * np.sqrt(ppy))
            if std > 0:
                sharpe = float(returns.mean() / std * np.sqrt(ppy))

    return {
        "count": count,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown_from_returns(returns),
    }


def _compound_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() - 1.0)


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    nav = np.concatenate([[1.0], np.cumprod(1.0 + returns.to_numpy(dtype=float))])
    running_max = np.maximum.accumulate(nav)
    drawdown = nav / running_max - 1.0
    return float(np.min(drawdown))


def _annualization_scale(periods_per_year: float | None) -> float:
    ppy = _valid_periods_per_year(periods_per_year)
    return float(np.sqrt(ppy)) if np.isfinite(ppy) else 1.0


def _valid_periods_per_year(periods_per_year: float | None) -> float:
    try:
        value = float(periods_per_year)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) and value > 0 else np.nan


def _estimate_obs_per_year(series: pd.Series) -> float:
    if series is None or series.empty:
        return np.nan
    if not isinstance(series.index, pd.DatetimeIndex):
        return np.nan
    start = series.index.min()
    end = series.index.max()
    if start is pd.NaT or end is pd.NaT:
        return np.nan
    days = float((end - start).days)
    if days <= 0:
        return np.nan
    return float(series.shape[0] / (days / 365.25))


def latest_rolling_stats(frame: pd.DataFrame, columns: list[str]) -> dict[str, float] | None:
    if frame is None or frame.empty:
        return None
    valid = frame.dropna(subset=columns, how="any")
    if valid.empty:
        return None
    last = valid.iloc[-1]
    return {col: float(last[col]) for col in columns}


def compute_rolling_ic(
    ic_series: pd.Series, window_months: list[int]
) -> tuple[dict[str, pd.DataFrame], float]:
    results: dict[str, pd.DataFrame] = {}
    if ic_series is None or ic_series.empty:
        return results, np.nan
    obs_per_year = _estimate_obs_per_year(ic_series)
    if not np.isfinite(obs_per_year) or obs_per_year <= 0:
        return results, np.nan
    for months in window_months:
        window_obs = int(round(obs_per_year * months / 12))
        if window_obs < 2:
            continue
        rolling = ic_series.rolling(window_obs, min_periods=window_obs)
        mean = rolling.mean()
        std = rolling.std(ddof=0)
        ir = mean / std
        frame = pd.DataFrame({"ic_mean": mean, "ic_std": std, "ic_ir": ir})
        results[f"{months}m"] = frame
    return results, float(obs_per_year)


def compute_rolling_sharpe(
    returns: pd.Series, window_months: list[int], periods_per_year: float
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    if returns is None or returns.empty:
        return results
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return results
    for months in window_months:
        window_obs = int(round(periods_per_year * months / 12))
        if window_obs < 2:
            continue
        rolling = returns.rolling(window_obs, min_periods=window_obs)
        mean = rolling.mean()
        std = rolling.std(ddof=1)
        sharpe = mean / std * np.sqrt(periods_per_year)
        frame = pd.DataFrame({"mean": mean, "std": std, "sharpe": sharpe})
        results[f"{months}m"] = frame
    return results


__all__ = [
    "DEFAULT_RECENCY_WINDOWS",
    "RECENCY_DIAGNOSTIC_COLUMNS",
    "build_recency_diagnostics",
    "compute_rolling_ic",
    "compute_rolling_sharpe",
    "latest_rolling_stats",
    "normalize_recency_windows",
    "parse_recency_window",
]
