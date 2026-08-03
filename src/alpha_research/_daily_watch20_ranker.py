"""DailyWatch20 ranker training/scoring (model-fit dependent)."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any, cast

import numpy as np
import pandas as pd
from xgboost import DMatrix
from xgboost.core import XGBoostError

from . import daily_watch20 as _stage
from ._daily_watch20_label import (
    RELATIVE_PERCENTILE_COL,
    DailyWatch20Config,
    DailyWatch20Explanation,
    DailyWatch20TrainingSummary,
    _as_timestamp,
    _coerce_features,
    _coerce_training_summary,
    _label_end_dates,
    _optional_normalized_dates,
    _stable_id,
    _validate_keys,
    build_multi_horizon_forward_rank_label,
)
from .modeling import feature_importance_frame, resolve_model_spec
from .signal_artifact import CANONICAL_SIGNAL_COLUMNS, build_signal_artifact_frame
from .split import build_sample_weight, select_train_window_dates


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
            "feature_policy_id": self.config.feature_policy_id,
            "label": self.config.label_col,
            "forward_days": self.config.forward_days,
            "label_horizon_weights": list(self.config.label_horizon_weights),
            "label_policy_id": self.config.label_policy_id,
        }
        return _stable_id(payload)

    @property
    def training_policy_id(self) -> str:
        """Stable identity for settings that determine the fitted training sample."""

        return _stable_id(self.training_policy)

    @property
    def training_policy(self) -> dict[str, Any]:
        """Return the auditable inputs used to derive ``training_policy_id``."""

        return {
            "train_window_dates": self.config.train_window_dates,
            "sample_weight_mode": self.config.sample_weight_mode,
            "sample_weight_params": dict(self.config.sample_weight_params),
            "min_query_size": self.config.min_query_size,
        }

    @property
    def persistence_metadata(self) -> dict[str, str]:
        """Compatibility identifiers that must accompany a persisted model."""

        return {
            "model_version": self.model_version,
            "feature_set_id": self.feature_set_id,
            "training_policy_id": self.training_policy_id,
        }

    def restore(
        self,
        model: Any,
        training_summary: DailyWatch20TrainingSummary | Mapping[str, Any],
        *,
        metadata: Mapping[str, Any],
    ) -> DailyWatch20Ranker:
        """Attach a trained model only when caller-supplied compatibility IDs match."""

        required = {"model_version", "feature_set_id", "training_policy_id"}
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

        model = _stage.build_model(self.model_type, self.model_params)
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
        self.model = _stage.build_model(self.model_type, self.model_params)
        _stage.fit_model(
            self.model,
            self.model_type,
            data,
            features=self.config.features,
            target_col=cast(str, self.config.label_col),
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
