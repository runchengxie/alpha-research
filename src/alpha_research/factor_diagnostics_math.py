from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from .factor_diagnostics_config import DEFAULT_SIZE_BUCKET_LABELS


def bucket_labels(count: int, labels: Sequence[str]) -> list[str]:
    count = max(int(count), 2)
    if len(labels) == count:
        return [str(label) for label in labels]
    if count == 3:
        return list(DEFAULT_SIZE_BUCKET_LABELS)
    return [f"q{idx + 1}" for idx in range(count)]


def size_buckets(values: pd.Series, *, bucket_count: int, labels: Sequence[str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(index=values.index, dtype=object)
    valid = numeric.dropna()
    if valid.nunique() < 2:
        return out
    count = min(bucket_count, int(valid.nunique()))
    use_labels = list(labels[:count])
    ranks = valid.rank(method="first")
    bucketed = pd.qcut(ranks, q=count, labels=use_labels)
    out.loc[bucketed.index] = bucketed.astype(object)
    return out


def long_short_return(values: pd.Series, target: pd.Series, *, min_obs: int) -> float:
    valid = values.notna() & target.notna()
    if int(valid.sum()) < min_obs:
        return np.nan
    frame = pd.DataFrame({"value": values.loc[valid], "target": target.loc[valid]})
    frame = frame.sort_values("value")
    count = max(1, int(np.floor(len(frame) * 0.2)))
    return float(frame.tail(count)["target"].mean() - frame.head(count)["target"].mean())


def factor_correlation_rows(correlation: pd.DataFrame, factor: str) -> pd.DataFrame:
    if correlation.empty:
        return pd.DataFrame()
    return correlation.loc[
        (correlation["factor_a"] == factor) | (correlation["factor_b"] == factor)
    ]


def dominant_style(exposure: pd.DataFrame) -> str | None:
    if exposure.empty or "rank_corr" not in exposure:
        return None
    grouped = exposure.assign(abs_corr=exposure["rank_corr"].abs())
    grouped = grouped.groupby("style")["abs_corr"].mean().dropna()
    if grouped.empty:
        return None
    return str(grouped.sort_values(ascending=False).index[0])


def max_abs_group_mean(frame: pd.DataFrame, *, group_col: str, value_col: str) -> float:
    if frame.empty or value_col not in frame or group_col not in frame:
        return np.nan
    grouped = frame.assign(abs_value=frame[value_col].abs())
    grouped = grouped.groupby(group_col)["abs_value"].mean().dropna()
    return float(grouped.max()) if not grouped.empty else np.nan


def size_bucket_ic_spread(frame: pd.DataFrame) -> float:
    if frame.empty or "rank_ic" not in frame:
        return np.nan
    grouped = frame.groupby("size_bucket")["rank_ic"].mean().dropna()
    if grouped.empty:
        return np.nan
    return float(grouped.max() - grouped.min())


def date_text(value: Any) -> str:
    return cast(pd.Timestamp, pd.Timestamp(value)).strftime("%Y%m%d")


def zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = numeric.mean()
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return numeric - mean
    return (numeric - mean) / std


def spearman(left: pd.Series, right: pd.Series) -> float:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    valid = left_num.notna() & right_num.notna()
    if int(valid.sum()) < 2:
        return np.nan
    return float(left_num.loc[valid].corr(right_num.loc[valid], method="spearman"))


def r2_score(y: np.ndarray, fitted: np.ndarray) -> float:
    total = float(np.sum((y - np.mean(y)) ** 2))
    if not np.isfinite(total) or total == 0:
        return np.nan
    resid = float(np.sum((y - fitted) ** 2))
    return float(1.0 - resid / total)


def safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


def safe_std(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.std(ddof=0)) if not numeric.empty else np.nan


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def column_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return safe_mean(frame[column])


def column_min(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(numeric.min()) if not numeric.empty else np.nan


def ir(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    std = numeric.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return np.nan
    return float(numeric.mean() / std)
