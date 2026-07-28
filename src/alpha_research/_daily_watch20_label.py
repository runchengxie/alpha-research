"""DailyWatch20 configuration and label construction (no model-fit dependencies)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from .daily_watch20_features import (
    DEFAULT_LABEL_HORIZON_WEIGHTS,
    label_columns_for_horizon_weights,
    normalize_label_horizon_weights,
)

RELATIVE_PERCENTILE_COL = "relative_percentile"
PREPARED_FEATURE_POLICY_ID = "prepared_features.v1"
PRICE_ONLY_LABEL_POLICY_ID = "forward_price_only.v1"


def _stable_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _normalize_policy_id(value: object, *, field: str) -> str:
    policy_id = str(value).strip()
    if not policy_id:
        raise ValueError(f"DailyWatch20Config.{field} must be non-empty.")
    return policy_id


@dataclass(frozen=True)
class DailyWatch20Config:
    """Configuration for a DailyWatch20 cross-sectional ranker."""

    features: Sequence[str]
    date_col: str = "trade_date"
    symbol_col: str = "symbol"
    price_col: str = "adj_close"
    forward_days: int = 5
    label_horizon_weights: Mapping[int, float] | Sequence[tuple[int, float]] = (
        DEFAULT_LABEL_HORIZON_WEIGHTS
    )
    label_col: str | None = None
    forward_return_col: str | None = None
    label_end_col: str = "forward_label_end_date"
    feature_policy_id: str = PREPARED_FEATURE_POLICY_ID
    label_policy_id: str = PRICE_ONLY_LABEL_POLICY_ID
    train_window_dates: int | None = 252
    sample_weight_mode: str | None = "date_equal"
    sample_weight_params: Mapping[str, object] = field(default_factory=dict)
    model_params: Mapping[str, Any] = field(default_factory=dict)
    min_query_size: int = 2
    eligible_for_backtest: bool = True
    eligible_for_live: bool = False

    def __post_init__(self) -> None:
        features = tuple(str(feature).strip() for feature in self.features)
        if not features or any(not feature for feature in features):
            raise ValueError("DailyWatch20Config.features must contain non-empty names.")
        if len(set(features)) != len(features):
            raise ValueError("DailyWatch20Config.features must be unique.")
        horizon_weights = normalize_label_horizon_weights(self.label_horizon_weights)
        if int(self.forward_days) != max(horizon for horizon, _weight in horizon_weights):
            raise ValueError(
                "DailyWatch20Config.forward_days must equal the longest configured label horizon."
            )
        if self.train_window_dates is not None and int(self.train_window_dates) <= 0:
            raise ValueError("DailyWatch20Config.train_window_dates must be positive or None.")
        if int(self.min_query_size) < 2:
            raise ValueError("DailyWatch20Config.min_query_size must be at least 2.")
        feature_policy_id = _normalize_policy_id(self.feature_policy_id, field="feature_policy_id")
        label_policy_id = _normalize_policy_id(self.label_policy_id, field="label_policy_id")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "forward_days", int(self.forward_days))
        object.__setattr__(self, "label_horizon_weights", horizon_weights)
        default_label, default_return = label_columns_for_horizon_weights(horizon_weights)
        object.__setattr__(self, "label_col", self.label_col or default_label)
        object.__setattr__(self, "forward_return_col", self.forward_return_col or default_return)
        object.__setattr__(self, "min_query_size", int(self.min_query_size))
        object.__setattr__(self, "feature_policy_id", feature_policy_id)
        object.__setattr__(self, "label_policy_id", label_policy_id)
        object.__setattr__(self, "sample_weight_params", dict(self.sample_weight_params))
        object.__setattr__(self, "model_params", dict(self.model_params))


@dataclass(frozen=True)
class DailyWatch20TrainingSummary:
    as_of_date: pd.Timestamp
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    rows: int
    query_groups: int
    sample_weight_mode: str | None


def _coerce_training_summary(
    value: DailyWatch20TrainingSummary | Mapping[str, Any],
) -> DailyWatch20TrainingSummary:
    if isinstance(value, DailyWatch20TrainingSummary):
        summary = value
    else:
        required = {
            "as_of_date",
            "train_start_date",
            "train_end_date",
            "rows",
            "query_groups",
            "sample_weight_mode",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError("DailyWatch20 training summary is missing: " + ", ".join(missing))
        summary = DailyWatch20TrainingSummary(
            as_of_date=_as_timestamp(value["as_of_date"], label="as_of_date"),
            train_start_date=_as_timestamp(value["train_start_date"], label="train_start_date"),
            train_end_date=_as_timestamp(value["train_end_date"], label="train_end_date"),
            rows=int(value["rows"]),
            query_groups=int(value["query_groups"]),
            sample_weight_mode=cast(str | None, value["sample_weight_mode"]),
        )
    if summary.train_start_date > summary.train_end_date:
        raise ValueError("DailyWatch20 training summary starts after it ends.")
    if summary.train_end_date > summary.as_of_date:
        raise ValueError("DailyWatch20 training summary ends after its as-of date.")
    if summary.rows <= 0 or summary.query_groups <= 0:
        raise ValueError("DailyWatch20 training summary counts must be positive.")
    return summary


@dataclass(frozen=True)
class DailyWatch20Explanation:
    source: str
    local_contributions: pd.DataFrame
    feature_importance: pd.DataFrame


def _normalized_dates(values: pd.Series, *, column: str) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.isna().any():
        raise ValueError(f"DailyWatch20 requires valid dates in column: {column}")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def _optional_normalized_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def _as_timestamp(value: Any, *, label: str) -> pd.Timestamp:
    timestamp = cast(pd.Timestamp, pd.Timestamp(value))
    if pd.isna(timestamp):
        raise ValueError(f"DailyWatch20 requires a valid {label}.")
    return timestamp.normalize()


def _validate_keys(
    frame: pd.DataFrame,
    *,
    date_col: str,
    symbol_col: str,
) -> pd.DataFrame:
    missing = [column for column in (date_col, symbol_col) if column not in frame.columns]
    if missing:
        raise ValueError("DailyWatch20 missing key columns: " + ", ".join(missing))
    out = frame.copy()
    out[date_col] = _normalized_dates(out[date_col], column=date_col)
    symbols = out[symbol_col].astype("string")
    if symbols.isna().any() or symbols.str.strip().eq("").any():
        raise ValueError(f"DailyWatch20 requires non-empty symbols in column: {symbol_col}")
    out[symbol_col] = symbols.astype(str)
    if out.duplicated([date_col, symbol_col]).any():
        raise ValueError(f"DailyWatch20 requires unique ({date_col}, {symbol_col}) rows.")
    return out


def _label_end_dates(
    frame: pd.DataFrame,
    *,
    date_col: str,
    symbol_col: str,
    forward_days: int,
) -> pd.Series:
    ordered = frame.sort_values([symbol_col, date_col], kind="mergesort")
    shifted = ordered.groupby(symbol_col, sort=False)[date_col].shift(-forward_days)
    return shifted.reindex(frame.index)


def build_forward_rank_label(
    frame: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    forward_days: int = 5,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    label_col: str = "forward_rank_5d",
    forward_return_col: str = "forward_return_5d",
    label_end_col: str = "forward_label_end_date",
) -> pd.DataFrame:
    """Add a forward-return percentile label ranked within each trade date."""

    if int(forward_days) <= 0:
        raise ValueError("forward_days must be positive.")
    out = _validate_keys(frame, date_col=date_col, symbol_col=symbol_col)
    if price_col not in out.columns:
        raise ValueError(f"DailyWatch20 missing price column: {price_col}")
    price = pd.to_numeric(out[price_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    price = price.where(price > 0)
    ordered = out.assign(_daily_watch20_price=price).sort_values(
        [symbol_col, date_col], kind="mergesort"
    )
    future_price = ordered.groupby(symbol_col, sort=False)["_daily_watch20_price"].shift(
        -int(forward_days)
    )
    ordered[forward_return_col] = future_price / ordered["_daily_watch20_price"] - 1.0
    ordered[label_end_col] = ordered.groupby(symbol_col, sort=False)[date_col].shift(
        -int(forward_days)
    )
    ordered[label_col] = ordered.groupby(date_col, sort=False)[forward_return_col].rank(
        method="average", pct=True
    )
    return ordered.drop(columns="_daily_watch20_price").sort_index()


def build_multi_horizon_forward_rank_label(
    frame: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    horizon_weights: Mapping[int, float] | Sequence[tuple[int, float]] = (
        DEFAULT_LABEL_HORIZON_WEIGHTS
    ),
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    label_col: str | None = None,
    forward_return_col: str | None = None,
    label_end_col: str = "forward_label_end_date",
) -> pd.DataFrame:
    """Build complete-case weighted forward ranks for one or more horizons."""

    normalized = normalize_label_horizon_weights(horizon_weights)
    target_label, target_return = label_columns_for_horizon_weights(normalized)
    target_label = label_col or target_label
    target_return = forward_return_col or target_return
    out = _validate_keys(frame, date_col=date_col, symbol_col=symbol_col)
    if price_col not in out.columns:
        raise ValueError(f"DailyWatch20 missing price column: {price_col}")
    price = pd.to_numeric(out[price_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    price = price.where(price > 0)
    ordered = out.assign(_daily_watch20_price=price).sort_values(
        [symbol_col, date_col], kind="mergesort"
    )
    grouped = ordered.groupby(symbol_col, sort=False)
    return_parts: list[pd.Series] = []
    rank_parts: list[pd.Series] = []
    for horizon, weight in normalized:
        return_name = f"forward_return_{horizon}d"
        rank_name = f"forward_rank_{horizon}d"
        future_price = grouped["_daily_watch20_price"].shift(-horizon)
        ordered[return_name] = future_price / ordered["_daily_watch20_price"] - 1.0
        ordered[rank_name] = ordered.groupby(date_col, sort=False)[return_name].rank(
            method="average", pct=True
        )
        return_parts.append(ordered[return_name].mul(weight))
        rank_parts.append(ordered[rank_name].mul(weight))
    if len(normalized) > 1 or target_return not in ordered.columns:
        ordered[target_return] = pd.concat(return_parts, axis=1).sum(
            axis=1, min_count=len(normalized)
        )
    if len(normalized) > 1 or target_label not in ordered.columns:
        ordered[target_label] = pd.concat(rank_parts, axis=1).sum(axis=1, min_count=len(normalized))
    longest_horizon = normalized[-1][0]
    ordered[label_end_col] = grouped[date_col].shift(-longest_horizon)
    return ordered.drop(columns="_daily_watch20_price").sort_index()


def _coerce_features(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError("DailyWatch20 missing feature columns: " + ", ".join(missing))
    out = frame.copy()
    for feature in features:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return out
