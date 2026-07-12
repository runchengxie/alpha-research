"""Daily cross-sectional XGBRanker research for prepared stock-date features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, cast

import numpy as np
import pandas as pd
from xgboost import DMatrix
from xgboost.core import XGBoostError

from .daily_watch20_features import (
    DEFAULT_LABEL_HORIZON_WEIGHTS,
    label_columns_for_horizon_weights,
    normalize_label_horizon_weights,
)
from .modeling import (
    build_model,
    feature_importance_frame,
    fit_model,
    resolve_model_spec,
)
from .signal_artifact import CANONICAL_SIGNAL_COLUMNS, build_signal_artifact_frame
from .split import build_sample_weight, select_train_window_dates

RELATIVE_PERCENTILE_COL = "relative_percentile"


def _stable_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


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
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "forward_days", int(self.forward_days))
        object.__setattr__(self, "label_horizon_weights", horizon_weights)
        default_label, default_return = label_columns_for_horizon_weights(horizon_weights)
        object.__setattr__(self, "label_col", self.label_col or default_label)
        object.__setattr__(self, "forward_return_col", self.forward_return_col or default_return)
        object.__setattr__(self, "min_query_size", int(self.min_query_size))
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


class DailyWatch20Ranker:
    """Train and score a date-grouped XGBRanker without data-lake dependencies."""

    def __init__(self, config: DailyWatch20Config) -> None:
        self.config = config
        model_type, model_params = resolve_model_spec(
            {"type": "xgb_ranker", "params": dict(config.model_params)}
        )
        self.model_type = model_type
        self.model_params = model_params
        self.model: Any | None = None
        self.training_summary: DailyWatch20TrainingSummary | None = None

    @property
    def model_version(self) -> str:
        payload = {"type": self.model_type, "params": self.model_params}
        return f"{self.model_type}:{_stable_id(payload)}"

    @property
    def feature_set_id(self) -> str:
        payload = {
            "features": list(self.config.features),
            "label": self.config.label_col,
            "forward_days": self.config.forward_days,
            "label_horizon_weights": list(self.config.label_horizon_weights),
        }
        return _stable_id(payload)

    @property
    def persistence_metadata(self) -> dict[str, str]:
        """Compatibility identifiers that must accompany a persisted model."""

        return {
            "model_version": self.model_version,
            "feature_set_id": self.feature_set_id,
        }

    def restore(
        self,
        model: Any,
        training_summary: DailyWatch20TrainingSummary | Mapping[str, Any],
        *,
        metadata: Mapping[str, Any],
    ) -> DailyWatch20Ranker:
        """Attach a trained model only when caller-supplied compatibility IDs match."""

        required = {"model_version", "feature_set_id"}
        missing = sorted(required - set(metadata))
        if missing:
            raise ValueError("DailyWatch20 restore metadata is missing: " + ", ".join(missing))
        expected = self.persistence_metadata
        mismatches = [key for key in sorted(required) if str(metadata[key]) != expected[key]]
        if mismatches:
            raise ValueError(
                "DailyWatch20 restore compatibility mismatch: " + ", ".join(mismatches)
            )
        if model is None or not callable(getattr(model, "predict", None)):
            raise ValueError("DailyWatch20 restored model must provide predict().")
        summary = _coerce_training_summary(training_summary)
        self.model = model
        self.training_summary = summary
        return self

    def restore_from_path(
        self,
        path: str | PathLike[str],
        training_summary: DailyWatch20TrainingSummary | Mapping[str, Any],
        *,
        metadata: Mapping[str, Any],
    ) -> DailyWatch20Ranker:
        """Load a native model file and apply the standard compatibility checks."""

        model = build_model(self.model_type, self.model_params)
        load_model = getattr(model, "load_model", None)
        if not callable(load_model):
            raise ValueError("DailyWatch20 model type does not support load_model().")
        load_model(path)
        return self.restore(model, training_summary, metadata=metadata)

    def _with_label_metadata(self, frame: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        out = _validate_keys(frame, date_col=cfg.date_col, symbol_col=cfg.symbol_col)
        if cfg.label_col not in out.columns:
            return build_multi_horizon_forward_rank_label(
                out,
                price_col=cfg.price_col,
                horizon_weights=cfg.label_horizon_weights,
                date_col=cfg.date_col,
                symbol_col=cfg.symbol_col,
                label_col=cfg.label_col,
                forward_return_col=cfg.forward_return_col,
                label_end_col=cfg.label_end_col,
            )
        if cfg.label_end_col not in out.columns:
            out[cfg.label_end_col] = _label_end_dates(
                out,
                date_col=cfg.date_col,
                symbol_col=cfg.symbol_col,
                forward_days=cfg.forward_days,
            )
        else:
            out[cfg.label_end_col] = _optional_normalized_dates(out[cfg.label_end_col])
        return out

    def _training_data(
        self, frame: pd.DataFrame, *, as_of_date: object | None
    ) -> tuple[pd.DataFrame, np.ndarray | None, pd.Timestamp]:
        cfg = self.config
        data = _coerce_features(self._with_label_metadata(frame), cfg.features)
        data[cfg.label_col] = pd.to_numeric(data[cfg.label_col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        raw_as_of = data[cfg.date_col].max() if as_of_date is None else as_of_date
        as_of = _as_timestamp(raw_as_of, label="as_of_date")
        invalid_end = data[cfg.label_end_col].notna() & (
            data[cfg.label_end_col] <= data[cfg.date_col]
        )
        if invalid_end.any():
            raise ValueError("DailyWatch20 label end dates must follow their feature dates.")
        known = data[cfg.label_col].notna() & data[cfg.label_end_col].notna()
        data = data.loc[known & (data[cfg.label_end_col] <= as_of)].copy()
        if data.empty:
            raise ValueError(f"DailyWatch20 has no known labels as of {as_of.date()}.")
        window_mode = "rolling" if cfg.train_window_dates is not None else "full"
        selected_dates = select_train_window_dates(
            data[cfg.date_col].to_numpy(),
            mode=window_mode,
            size=cfg.train_window_dates,
            unit="dates",
        )
        data = data[data[cfg.date_col].isin(selected_dates)].copy()
        query_size = data.groupby(cfg.date_col, sort=False)[cfg.date_col].transform("size")
        data = data.loc[query_size >= cfg.min_query_size].copy()
        if data.empty:
            raise ValueError("DailyWatch20 has no query groups meeting min_query_size.")
        sample_weight = build_sample_weight(
            data,
            cfg.sample_weight_mode,
            date_col=cfg.date_col,
            params=cfg.sample_weight_params,
        )
        return data, sample_weight, as_of

    def fit(self, frame: pd.DataFrame, *, as_of_date: object | None = None) -> DailyWatch20Ranker:
        data, sample_weight, as_of = self._training_data(frame, as_of_date=as_of_date)
        self.model = build_model(self.model_type, self.model_params)
        fit_model(
            self.model,
            self.model_type,
            data,
            features=self.config.features,
            target_col=self.config.label_col,
            sample_weight=sample_weight,
            date_col=self.config.date_col,
        )
        dates = pd.Index(data[self.config.date_col].unique()).sort_values()
        self.training_summary = DailyWatch20TrainingSummary(
            as_of_date=as_of,
            train_start_date=_as_timestamp(dates[0], label="train_start_date"),
            train_end_date=_as_timestamp(dates[-1], label="train_end_date"),
            rows=len(data),
            query_groups=len(dates),
            sample_weight_mode=self.config.sample_weight_mode,
        )
        return self

    def _prediction_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_model()
        cfg = self.config
        data = _validate_keys(frame, date_col=cfg.date_col, symbol_col=cfg.symbol_col)
        return _coerce_features(data, cfg.features)

    def _require_model(self) -> Any:
        if self.model is None:
            raise ValueError("DailyWatch20Ranker requires fit() before prediction.")
        return self.model

    def predict_relative(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return only cross-sectional percentiles and ranks, never model margins."""

        cfg = self.config
        data = self._prediction_frame(frame)
        model = self._require_model()
        margin = np.asarray(model.predict(data[list(cfg.features)]), dtype=float)
        scored = data[[cfg.date_col, cfg.symbol_col]].copy()
        scored["_model_margin"] = margin
        scored[RELATIVE_PERCENTILE_COL] = scored.groupby(cfg.date_col, sort=False)[
            "_model_margin"
        ].rank(method="average", pct=True)
        scored = scored.sort_values(
            [cfg.date_col, "_model_margin", cfg.symbol_col],
            ascending=[True, False, True],
            kind="mergesort",
        )
        scored["rank"] = scored.groupby(cfg.date_col, sort=False).cumcount().add(1).astype("Int64")
        return scored.drop(columns="_model_margin").reset_index(drop=True)

    def predict_signals(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build canonical research signals whose score fields are percentiles."""

        relative = self.predict_relative(frame)
        relative["raw_pred"] = relative[RELATIVE_PERCENTILE_COL]
        relative["signal_eval"] = relative[RELATIVE_PERCENTILE_COL]
        relative["signal_backtest"] = relative[RELATIVE_PERCENTILE_COL]
        signals = build_signal_artifact_frame(
            relative,
            model_version=self.model_version,
            feature_set_id=self.feature_set_id,
            signal_direction=1.0,
            eligible_for_backtest=self.config.eligible_for_backtest,
            eligible_for_live=self.config.eligible_for_live,
        )
        return signals[list(CANONICAL_SIGNAL_COLUMNS)]

    def fit_predict(self, frame: pd.DataFrame, prediction_date: Any) -> pd.DataFrame:
        """Fit through one as-of date and score that date's cross-section."""

        date = _as_timestamp(prediction_date, label="prediction_date")
        self.fit(frame, as_of_date=date)
        normalized = _validate_keys(
            frame,
            date_col=self.config.date_col,
            symbol_col=self.config.symbol_col,
        )
        prediction = normalized.loc[normalized[self.config.date_col].eq(date)]
        if prediction.empty:
            raise ValueError(f"DailyWatch20 has no rows for prediction date {date.date()}.")
        return self.predict_relative(prediction)

    def explain(self, frame: pd.DataFrame) -> DailyWatch20Explanation:
        """Return XGBoost contributions, or global feature importance as a fallback."""

        data = self._prediction_frame(frame)
        importance, importance_source = feature_importance_frame(
            self.model,
            self.config.features,
            model_type=self.model_type,
        )
        try:
            local = self._native_contributions(data)
        except (AttributeError, TypeError, ValueError, XGBoostError):
            return DailyWatch20Explanation(
                source=importance_source,
                local_contributions=pd.DataFrame(
                    columns=[
                        self.config.date_col,
                        self.config.symbol_col,
                        "feature",
                        "contribution",
                    ]
                ),
                feature_importance=importance.reset_index(drop=True),
            )
        return DailyWatch20Explanation(
            source="xgb_pred_contribs",
            local_contributions=local,
            feature_importance=importance.reset_index(drop=True),
        )

    def _native_contributions(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        model = self._require_model()
        feature_names = list(cfg.features)
        matrix = DMatrix(data[feature_names], feature_names=feature_names)
        values = np.asarray(model.get_booster().predict(matrix, pred_contribs=True))
        names = [*feature_names, "__bias__"]
        if values.shape != (len(data), len(names)):
            raise ValueError("Unexpected XGBoost contribution matrix shape.")
        keys = data[[cfg.date_col, cfg.symbol_col]].reset_index(drop=True)
        contributions = pd.DataFrame(values, columns=names)
        wide = pd.concat([keys, contributions], axis=1)
        return wide.melt(
            id_vars=[cfg.date_col, cfg.symbol_col],
            var_name="feature",
            value_name="contribution",
        )


__all__ = [
    "RELATIVE_PERCENTILE_COL",
    "DailyWatch20Config",
    "DailyWatch20Explanation",
    "DailyWatch20Ranker",
    "DailyWatch20TrainingSummary",
    "build_forward_rank_label",
    "build_multi_horizon_forward_rank_label",
]
