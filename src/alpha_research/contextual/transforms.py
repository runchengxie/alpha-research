from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

_TRANSFORMS = {
    "level",
    "change_1p",
    "change_np",
    "yoy",
    "rolling_zscore",
    "acceleration",
    "rolling_percentile",
}
_WINDOWED = {"change_np", "rolling_zscore", "rolling_percentile"}
_METADATA_COLUMNS = (
    "series_id",
    "period_end",
    "available_at",
    "source_retrieved_at",
)


@dataclass(frozen=True)
class ContextTransformSpec:
    series_id: str
    transform: str
    feature_name: str
    window: int | None = None
    minimum_history: int = 1
    staleness_limit_days: int | None = None

    def __post_init__(self) -> None:
        if not str(self.series_id).strip():
            raise ValueError("ContextTransformSpec.series_id must be non-empty")
        if self.transform not in _TRANSFORMS:
            raise ValueError(f"unknown contextual transform: {self.transform}")
        if not str(self.feature_name).strip():
            raise ValueError("ContextTransformSpec.feature_name must be non-empty")
        if self.minimum_history < 1:
            raise ValueError("ContextTransformSpec.minimum_history must be positive")
        if self.staleness_limit_days is not None and self.staleness_limit_days < 0:
            raise ValueError("ContextTransformSpec.staleness_limit_days must be non-negative")
        if self.transform in _WINDOWED and (self.window is None or self.window < 1):
            raise ValueError(f"transform {self.transform} requires a positive window")
        if self.transform not in _WINDOWED and self.window is not None and self.window < 1:
            raise ValueError("ContextTransformSpec.window must be positive when supplied")


def _timestamps(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if values.isna().any():
        raise ValueError(f"context observations contain invalid {column}")
    return values


def _validate_input(observations: pd.DataFrame) -> pd.DataFrame:
    required = {*_METADATA_COLUMNS, "value"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"context observations are missing columns: {', '.join(missing)}")
    frame = observations.copy()
    frame["series_id"] = frame["series_id"].astype(str)
    for column in ("period_end", "available_at", "source_retrieved_at"):
        frame[column] = _timestamps(frame, column)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if frame["value"].isna().any():
        raise ValueError("context observations value must be numeric")
    duplicates = frame.duplicated(["series_id", "period_end"], keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, ["series_id", "period_end"]].head(3).to_dict("records")
        raise ValueError(
            "build_context_features expects a revision-selected PIT panel with one row per "
            f"series_id/period_end; duplicates include {examples}"
        )
    return frame.sort_values(
        ["series_id", "period_end", "available_at", "source_retrieved_at"],
        kind="stable",
    ).reset_index(drop=True)


def _with_minimum_history(values: pd.Series, result: pd.Series, minimum: int) -> pd.Series:
    positions = pd.Series(np.arange(len(values)), index=values.index)
    return result.where(positions >= minimum - 1)


def _yoy_delta(period_end: pd.Series, values: pd.Series) -> pd.Series:
    lookup = {
        pd.Timestamp(period): float(value) for period, value in zip(period_end, values, strict=True)
    }
    result: list[float] = []
    for period, value in zip(period_end, values, strict=True):
        timestamp = pd.Timestamp(period)
        if not isinstance(timestamp, pd.Timestamp):
            result.append(np.nan)
            continue
        prior_period = timestamp - pd.DateOffset(years=1)
        prior = lookup.get(prior_period)
        result.append(float(value) - prior if prior is not None else np.nan)
    return pd.Series(result, index=values.index, dtype=float)


def _rolling_zscore(values: pd.Series, window: int) -> pd.Series:
    rolling = values.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    return ((values - mean) / std).where(std > 0)


def _last_percentile(values: pd.Series, window: int) -> pd.Series:
    def percentile(array: np.ndarray) -> float:
        if len(array) < window or np.isnan(array).any():
            return np.nan
        current = array[-1]
        less = float(np.sum(array < current))
        equal = float(np.sum(array == current))
        # Mid-rank percentile is stable when ties occur and stays in [0, 1].
        rank = less + (equal + 1.0) / 2.0
        return (rank - 1.0) / max(len(array) - 1.0, 1.0)

    return values.rolling(window=window, min_periods=window).apply(percentile, raw=True)


def _apply_transform(frame: pd.DataFrame, spec: ContextTransformSpec) -> pd.Series:
    values = frame["value"].astype(float)
    if spec.transform == "level":
        result = values.copy()
    elif spec.transform == "change_1p":
        result = values.diff(1)
    elif spec.transform == "change_np":
        assert spec.window is not None
        result = values.diff(spec.window)
    elif spec.transform == "yoy":
        result = _yoy_delta(frame["period_end"], values)
    elif spec.transform == "rolling_zscore":
        assert spec.window is not None
        result = _rolling_zscore(values, spec.window)
    elif spec.transform == "acceleration":
        result = values.diff(1).diff(1)
    elif spec.transform == "rolling_percentile":
        assert spec.window is not None
        result = _last_percentile(values, spec.window)
    else:  # pragma: no cover - dataclass validation makes this unreachable
        raise AssertionError(spec.transform)
    return _with_minimum_history(values, result.astype(float), spec.minimum_history)


def build_context_features(
    observations: pd.DataFrame,
    specs: Sequence[ContextTransformSpec],
) -> pd.DataFrame:
    """Build deterministic context-state columns from a revision-selected PIT panel.

    ``observations`` must contain at most one visible vintage per ``series_id`` and
    ``period_end``. This deliberately keeps revision selection outside the transform
    layer: callers must first construct the PIT state appropriate for their as-of.

    ``yoy`` means an arithmetic delta from the exact corresponding period one year
    earlier. It does not silently convert level units to percentage growth.
    """

    frame = _validate_input(observations)
    names = [spec.feature_name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate contextual feature_name values are not allowed")

    outputs: list[pd.DataFrame] = []
    specs_by_series: dict[str, list[ContextTransformSpec]] = {}
    for spec in specs:
        specs_by_series.setdefault(spec.series_id, []).append(spec)

    for series_id, series_frame in frame.groupby("series_id", sort=False):
        requested = specs_by_series.get(str(series_id), [])
        if not requested:
            continue
        output = series_frame.loc[:, list(_METADATA_COLUMNS)].copy()
        for name in names:
            output[name] = np.nan
        for spec in requested:
            output[spec.feature_name] = _apply_transform(series_frame, spec).to_numpy()
        outputs.append(output)

    if not outputs:
        columns = [*_METADATA_COLUMNS, *names]
        return pd.DataFrame(columns=pd.Index(columns))
    return (
        pd.concat(outputs, ignore_index=True)
        .sort_values(["series_id", "period_end"], kind="stable")
        .reset_index(drop=True)
    )
