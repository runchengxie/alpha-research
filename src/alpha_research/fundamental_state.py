"""PIT-safe building blocks for fundamental-state forecasting research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd

FUNDAMENTAL_STATE_SCHEMA = "fundamental_state_forecasting.v1"
TargetTransform = Literal["level", "delta", "pct_change"]


@dataclass(frozen=True)
class FundamentalTargetSpec:
    """Define one future fundamental target from a canonical annual observation."""

    name: str
    source_col: str
    transform: TargetTransform = "level"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("FundamentalTargetSpec.name must be non-empty")
        if not self.source_col.strip():
            raise ValueError("FundamentalTargetSpec.source_col must be non-empty")
        if self.transform not in {"level", "delta", "pct_change"}:
            raise ValueError("FundamentalTargetSpec.transform must be level, delta, or pct_change")


@dataclass(frozen=True)
class FundamentalTargetPanel:
    frame: pd.DataFrame
    audit: dict[str, object]


@dataclass(frozen=True)
class FundamentalScoreSpec:
    """Describe one forecast or valuation input to the cross-sectional score."""

    column: str
    weight: float = 1.0
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not self.column.strip():
            raise ValueError("FundamentalScoreSpec.column must be non-empty")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("FundamentalScoreSpec.weight must be finite and positive")


@dataclass(frozen=True)
class FundamentalPurgeResult:
    frame: pd.DataFrame
    audit: dict[str, object]


def _normalized_dates(series: pd.Series, *, column: str) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"fundamental state requires valid dates in {column}")
    if values.dt.tz is not None:
        values = values.dt.tz_localize(None)
    return values.dt.normalize()


def _numeric(series: pd.Series) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(series, errors="coerce")).replace(
        [np.inf, -np.inf], np.nan
    )


def _validate_target_specs(specs: tuple[FundamentalTargetSpec, ...]) -> None:
    if not specs:
        raise ValueError("fundamental target specs must be non-empty")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("fundamental target names must be unique")


def _target_values(
    current: pd.Series,
    future: pd.Series,
    *,
    transform: TargetTransform,
) -> pd.Series:
    if transform == "level":
        return future
    if transform == "delta":
        return future - current
    valid_base = current.where(current.notna() & np.isfinite(current) & (current > 0))
    return ((future / valid_base) - 1.0).replace([np.inf, -np.inf], np.nan)


def build_annual_fundamental_target_panel(
    frame: pd.DataFrame,
    target_specs: tuple[FundamentalTargetSpec, ...],
    *,
    horizon_years: int = 1,
    symbol_col: str = "symbol",
    report_period_col: str = "report_period",
    available_date_col: str = "available_date",
) -> FundamentalTargetPanel:
    """Attach exact-horizon annual targets without hiding their future availability date.

    The input contract is deliberately strict: one canonical, PIT-audited observation per
    ``(symbol, report_period)``. Revision selection belongs to the data platform, not here.
    """

    specs = tuple(target_specs)
    _validate_target_specs(specs)
    if int(horizon_years) <= 0:
        raise ValueError("horizon_years must be positive")
    required = {symbol_col, report_period_col, available_date_col}
    required.update(spec.source_col for spec in specs)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fundamental state frame missing columns: {missing}")

    out = frame.copy()
    out[symbol_col] = out[symbol_col].astype("string")
    if out[symbol_col].isna().any() or out[symbol_col].str.strip().eq("").any():
        raise ValueError("fundamental state requires non-empty symbols")
    out[symbol_col] = out[symbol_col].astype(str)
    out[report_period_col] = _normalized_dates(out[report_period_col], column=report_period_col)
    out[available_date_col] = _normalized_dates(out[available_date_col], column=available_date_col)
    if out.duplicated([symbol_col, report_period_col]).any():
        raise ValueError("fundamental state input contains duplicate symbol/report_period rows")
    if (out[available_date_col] <= out[report_period_col]).any():
        raise ValueError("annual observations must become available after their report period")

    out["feature_as_of_date"] = out[available_date_col]
    out["target_report_period"] = out[report_period_col] + pd.DateOffset(years=int(horizon_years))

    future_columns = [symbol_col, report_period_col, available_date_col]
    future_columns.extend(sorted({spec.source_col for spec in specs}))
    future = out[future_columns].copy()
    rename = {
        report_period_col: "target_report_period",
        available_date_col: "target_available_date",
    }
    rename.update({spec.source_col: f"__future__{spec.source_col}" for spec in specs})
    future.rename(columns=rename, inplace=True)

    merged = out.merge(
        future,
        how="left",
        on=[symbol_col, "target_report_period"],
        validate="many_to_one",
        sort=False,
    )
    merged["target_available_date"] = pd.to_datetime(
        merged["target_available_date"], errors="coerce"
    ).dt.normalize()
    invalid_availability = merged["target_available_date"].notna() & (
        merged["target_available_date"] <= merged["feature_as_of_date"]
    )
    if invalid_availability.any():
        raise ValueError("future fundamental labels must become available after feature_as_of_date")
    merged["fundamental_label_end_date"] = merged["target_available_date"]

    for spec in specs:
        current = _numeric(merged[spec.source_col])
        future_values = _numeric(merged[f"__future__{spec.source_col}"])
        merged[spec.name] = _target_values(current, future_values, transform=spec.transform)
        merged.drop(columns=[f"__future__{spec.source_col}"], inplace=True)

    complete = merged["target_available_date"].notna()
    complete &= merged[[spec.name for spec in specs]].notna().all(axis=1)
    audit: dict[str, object] = {
        "schema_version": FUNDAMENTAL_STATE_SCHEMA,
        "input_contract": "one canonical PIT-audited row per symbol/report_period",
        "horizon_years": int(horizon_years),
        "rows": int(len(merged)),
        "complete_label_rows": int(complete.sum()),
        "target_names": [spec.name for spec in specs],
        "label_end_semantics": "target_available_date",
    }
    return FundamentalTargetPanel(merged, audit)


def build_persistence_baseline(
    frame: pd.DataFrame,
    target_spec: FundamentalTargetSpec,
) -> pd.Series:
    """Return the naive no-change benchmark for one target definition."""

    if target_spec.source_col not in frame.columns:
        raise ValueError(f"persistence baseline missing column: {target_spec.source_col}")
    current = _numeric(frame[target_spec.source_col])
    if target_spec.transform == "level":
        return current
    return pd.Series(0.0, index=frame.index, dtype=float)


def _rank_ic(actual: pd.Series, predicted: pd.Series) -> float:
    actual_rank = actual.rank(method="average")
    predicted_rank = predicted.rank(method="average")
    value = actual_rank.corr(predicted_rank)
    return float(value) if pd.notna(value) else float("nan")


def evaluate_fundamental_forecast(
    frame: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    *,
    directional: bool = False,
) -> dict[str, float | int | None]:
    """Evaluate one OOS fundamental forecast with scale and rank metrics."""

    missing = sorted({actual_col, predicted_col} - set(frame.columns))
    if missing:
        raise ValueError(f"forecast evaluation missing columns: {missing}")
    actual = _numeric(frame[actual_col])
    predicted = _numeric(frame[predicted_col])
    valid = actual.notna() & predicted.notna()
    actual = actual.loc[valid]
    predicted = predicted.loc[valid]
    if actual.empty:
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "rank_ic": float("nan"),
            "direction_accuracy": None if not directional else float("nan"),
        }
    error = predicted - actual
    direction_accuracy: float | None = None
    if directional:
        direction_accuracy = float((np.sign(predicted) == np.sign(actual)).mean())
    return {
        "count": int(len(actual)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error.to_numpy(dtype=float))))),
        "rank_ic": _rank_ic(actual, predicted),
        "direction_accuracy": direction_accuracy,
    }


def build_fundamental_forecast_score(
    frame: pd.DataFrame,
    score_specs: tuple[FundamentalScoreSpec, ...],
    *,
    date_col: str = "signal_date",
    score_col: str = "fundamental_score",
) -> pd.DataFrame:
    """Combine forecast and valuation columns into a transparent cross-sectional score."""

    specs = tuple(score_specs)
    if not specs:
        raise ValueError("fundamental score specs must be non-empty")
    required = {date_col}
    required.update(spec.column for spec in specs)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fundamental score frame missing columns: {missing}")

    out = frame.copy()
    out[date_col] = _normalized_dates(out[date_col], column=date_col)
    weighted_sum = pd.Series(0.0, index=out.index, dtype=float)
    observed_weight = pd.Series(0.0, index=out.index, dtype=float)
    for spec in specs:
        values = _numeric(out[spec.column])
        ranked = values.groupby(out[date_col], sort=False).rank(
            method="average",
            pct=True,
            ascending=spec.higher_is_better,
        )
        component_col = f"{score_col}__{spec.column}_pct"
        out[component_col] = ranked
        observed = ranked.notna()
        weighted_sum.loc[observed] += ranked.loc[observed] * spec.weight
        observed_weight.loc[observed] += spec.weight
    out[score_col] = weighted_sum / observed_weight.replace(0.0, np.nan)
    out[f"{score_col}_coverage_weight"] = observed_weight
    out["fundamental_rank"] = out.groupby(date_col, sort=False)[score_col].rank(
        method="min", ascending=False
    )
    out["fundamental_percentile"] = out.groupby(date_col, sort=False)[score_col].rank(
        method="average", pct=True
    )
    return out


def purge_and_embargo_fundamental_rows(
    frame: pd.DataFrame,
    *,
    test_start: object,
    test_end: object,
    embargo_days: int = 0,
    feature_date_col: str = "feature_as_of_date",
    label_end_col: str = "fundamental_label_end_date",
) -> FundamentalPurgeResult:
    """Remove training rows whose label windows touch the test interval or embargo buffer."""

    missing = sorted({feature_date_col, label_end_col} - set(frame.columns))
    if missing:
        raise ValueError(f"fundamental purge frame missing columns: {missing}")
    if int(embargo_days) < 0:
        raise ValueError("embargo_days must be non-negative")
    start = cast(pd.Timestamp, pd.Timestamp(test_start)).normalize()
    end = cast(pd.Timestamp, pd.Timestamp(test_end)).normalize()
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("test_start and test_end must define a valid interval")

    feature_dates = _normalized_dates(frame[feature_date_col], column=feature_date_col)
    label_ends = _normalized_dates(frame[label_end_col], column=label_end_col)
    overlaps = (feature_dates <= end) & (label_ends >= start)
    embargo_end = end + pd.Timedelta(days=int(embargo_days))
    embargoed = (~overlaps) & (feature_dates > end) & (feature_dates <= embargo_end)
    keep = ~(overlaps | embargoed)
    audit: dict[str, object] = {
        "schema_version": FUNDAMENTAL_STATE_SCHEMA,
        "test_start": start.strftime("%Y-%m-%d"),
        "test_end": end.strftime("%Y-%m-%d"),
        "embargo_days": int(embargo_days),
        "input_rows": int(len(frame)),
        "purged_overlap_rows": int(overlaps.sum()),
        "embargoed_rows": int(embargoed.sum()),
        "kept_rows": int(keep.sum()),
    }
    return FundamentalPurgeResult(frame.loc[keep].copy(), audit)


__all__ = [
    "FUNDAMENTAL_STATE_SCHEMA",
    "FundamentalPurgeResult",
    "FundamentalScoreSpec",
    "FundamentalTargetPanel",
    "FundamentalTargetSpec",
    "build_annual_fundamental_target_panel",
    "build_fundamental_forecast_score",
    "build_persistence_baseline",
    "evaluate_fundamental_forecast",
    "purge_and_embargo_fundamental_rows",
]
