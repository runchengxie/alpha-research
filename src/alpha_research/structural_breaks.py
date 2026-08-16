"""Point-in-time structural-break diagnostics for financial research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from .fracdiff import augmented_dickey_fuller_t_stat


@dataclass(frozen=True)
class CusumBreak:
    timestamp: pd.Timestamp
    direction: int
    statistic: float


def symmetric_cusum_filter(
    series: pd.Series,
    *,
    threshold: float | pd.Series,
    drift: float = 0.0,
) -> pd.DatetimeIndex:
    """Return timestamps where cumulative positive or negative moves cross a threshold."""

    values = pd.to_numeric(series, errors="coerce")
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("series must use a DatetimeIndex")
    if drift < 0:
        raise ValueError("drift must be >= 0")
    if np.isscalar(threshold):
        threshold_values = pd.Series(float(cast(float, threshold)), index=values.index)
    else:
        threshold_values = pd.to_numeric(threshold, errors="coerce").reindex(values.index)
    if bool((threshold_values <= 0).dropna().any()):
        raise ValueError("threshold must be positive")

    positive = 0.0
    negative = 0.0
    events: list[pd.Timestamp] = []
    for timestamp, value in values.items():
        limit = threshold_values.loc[timestamp]
        if not np.isfinite(value) or not np.isfinite(limit):
            continue
        positive = max(0.0, positive + float(value) - drift)
        negative = min(0.0, negative + float(value) + drift)
        if positive > float(limit):
            positive = 0.0
            events.append(cast(pd.Timestamp, pd.Timestamp(timestamp)))
        elif negative < -float(limit):
            negative = 0.0
            events.append(cast(pd.Timestamp, pd.Timestamp(timestamp)))
    return pd.DatetimeIndex(events, name=values.index.name)


def rolling_standardized_cusum(
    series: pd.Series,
    *,
    window: int = 60,
    z_threshold: float = 4.0,
) -> pd.DataFrame:
    """Compute rolling standardized CUSUM statistics and break flags."""

    if window < 10:
        raise ValueError("window must be >= 10")
    if z_threshold <= 0:
        raise ValueError("z_threshold must be > 0")
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std(ddof=1)
    zscore = (values - mean).div(std.replace(0.0, np.nan))
    positive = zscore.clip(lower=0.0).rolling(window, min_periods=window).sum()
    negative = zscore.clip(upper=0.0).rolling(window, min_periods=window).sum()
    statistic = pd.concat([positive.abs(), negative.abs()], axis=1).max(axis=1)
    return pd.DataFrame(
        {
            "rolling_mean": mean,
            "rolling_std": std,
            "zscore": zscore,
            "cusum_stat": statistic,
            "break_flag": statistic >= z_threshold * np.sqrt(window),
        },
        index=values.index,
    )


def recursive_residual_cusum(
    y: pd.Series,
    x: pd.DataFrame,
    *,
    min_train_size: int | None = None,
) -> pd.DataFrame:
    """Compute one-step recursive residuals and their standardized cumulative sum."""

    response = pd.to_numeric(y, errors="coerce")
    features = x.apply(pd.to_numeric, errors="coerce")
    aligned = pd.concat([response.rename("y"), features], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame(columns=pd.Index(["recursive_residual", "cusum", "standardized_cusum"]))
    x_values = np.column_stack(
        [np.ones(len(aligned), dtype=float), aligned[features.columns].to_numpy(dtype=float)]
    )
    y_values = aligned["y"].to_numpy(dtype=float)
    parameter_count = x_values.shape[1]
    initial = int(min_train_size or max(parameter_count + 2, 20))
    if initial >= len(aligned):
        raise ValueError("min_train_size leaves no observations for recursive residuals")

    residuals = pd.Series(np.nan, index=aligned.index, dtype=float)
    for end in range(initial, len(aligned)):
        train_x = x_values[:end]
        train_y = y_values[:end]
        beta, *_ = np.linalg.lstsq(train_x, train_y, rcond=None)
        prediction = float(x_values[end] @ beta)
        leverage = float(x_values[end] @ np.linalg.pinv(train_x.T @ train_x) @ x_values[end])
        denominator = np.sqrt(max(1.0 + leverage, 1e-12))
        residuals.iloc[end] = (y_values[end] - prediction) / denominator
    valid = residuals.dropna()
    scale = float(valid.std(ddof=1))
    cumulative = valid.cumsum()
    standardized = cumulative / scale if np.isfinite(scale) and scale > 0 else cumulative * np.nan
    result = pd.DataFrame(index=aligned.index)
    result["recursive_residual"] = residuals
    result["cusum"] = cumulative.reindex(result.index)
    result["standardized_cusum"] = standardized.reindex(result.index)
    return result


def sadf_series(
    series: pd.Series,
    *,
    min_window: int = 40,
    lags: int = 1,
    step: int = 1,
) -> pd.Series:
    """Compute a Supremum ADF statistic using backward-expanding start points."""

    if min_window < max(10, lags + 5):
        raise ValueError("min_window is too small for the requested lag count")
    if step <= 0:
        raise ValueError("step must be > 0")
    values = pd.to_numeric(series, errors="coerce")
    output = pd.Series(np.nan, index=values.index, dtype=float, name="sadf")
    for end in range(min_window, len(values) + 1):
        statistics: list[float] = []
        latest_start = end - min_window
        for start in range(0, latest_start + 1, step):
            statistic = augmented_dickey_fuller_t_stat(values.iloc[start:end], lags=lags)
            if np.isfinite(statistic):
                statistics.append(float(statistic))
        if statistics:
            output.iloc[end - 1] = max(statistics)
    return output


def structural_break_receipt(
    frame: pd.DataFrame,
    *,
    statistic_col: str,
    threshold: float,
) -> dict[str, object]:
    """Summarize a structural-break artifact without embedding market data."""

    if statistic_col not in frame.columns:
        raise ValueError(f"statistic column not found: {statistic_col}")
    statistic = pd.to_numeric(frame[statistic_col], errors="coerce")
    return {
        "schema_version": 1,
        "statistic": statistic_col,
        "threshold": float(threshold),
        "observations": int(statistic.notna().sum()),
        "break_count": int((statistic >= threshold).sum()),
        "maximum": float(statistic.max()) if statistic.notna().any() else float("nan"),
        "latest": float(statistic.dropna().iloc[-1]) if statistic.notna().any() else float("nan"),
    }


__all__ = [
    "CusumBreak",
    "recursive_residual_cusum",
    "rolling_standardized_cusum",
    "sadf_series",
    "structural_break_receipt",
    "symmetric_cusum_filter",
]
