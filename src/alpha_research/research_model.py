from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .modeling import (
    build_model,
    feature_importance_frame,
    fit_model,
    resolve_model_spec,
)
from .research_dataset import ResearchDataset


def _stable_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class ResearchModel:
    model_type: str
    model_params: dict[str, Any]
    features: list[str]
    target_col: str
    date_col: str = "trade_date"
    model: Any | None = None
    fitted_segment: str | None = None
    last_prediction: pd.DataFrame | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        model_cfg: dict[str, Any] | None,
        *,
        features: list[str],
        target_col: str,
        date_col: str = "trade_date",
    ) -> ResearchModel:
        model_type, model_params = resolve_model_spec(model_cfg)
        return cls(
            model_type=model_type,
            model_params=model_params,
            features=list(features),
            target_col=target_col,
            date_col=date_col,
        )

    @property
    def model_version(self) -> str:
        payload = {"type": self.model_type, "params": self.model_params}
        return f"{self.model_type}:{_stable_id(payload)}"

    @property
    def feature_set_id(self) -> str:
        return _stable_id({"features": self.features, "target": self.target_col})

    def fit(self, dataset: ResearchDataset, segment: str = "all") -> ResearchModel:
        train_frame = dataset.fetch_learn(segment)
        self.model = build_model(self.model_type, self.model_params)
        fit_model(
            self.model,
            self.model_type,
            train_frame,
            features=self.features,
            target_col=self.target_col,
            date_col=self.date_col,
        )
        self.fitted_segment = segment
        return self

    def predict(self, dataset: ResearchDataset, segment: str = "all") -> pd.DataFrame:
        if self.model is None:
            raise ValueError("ResearchModel.predict requires fit() first.")
        infer_frame = dataset.fetch_infer(segment, audit=True)
        pred = self.model.predict(infer_frame[self.features])
        out = infer_frame[[self.date_col, "symbol"]].copy()
        out.rename(columns={self.date_col: "signal_date"}, inplace=True)
        out["raw_pred"] = pred
        out["model_version"] = self.model_version
        out["feature_set_id"] = self.feature_set_id
        self.last_prediction = out
        return out

    def detail(self) -> dict[str, Any]:
        importance_source = None
        top_features: list[dict[str, Any]] = []
        zero_feature_importance: bool | None = None
        if self.model is not None:
            importance_df, importance_source = feature_importance_frame(
                self.model,
                self.features,
                model_type=self.model_type,
            )
            if not importance_df.empty:
                top_features = [
                    {
                        "feature": str(row["feature"]),
                        "importance": float(row["importance"]),
                    }
                    for _, row in importance_df.head(20).iterrows()
                ]
                values = pd.to_numeric(importance_df["importance"], errors="coerce").fillna(0.0)
                zero_feature_importance = bool((values.abs() > 0.0).sum() == 0)
        constant_prediction = None
        if self.last_prediction is not None and "raw_pred" in self.last_prediction.columns:
            constant_prediction = int(self.last_prediction["raw_pred"].nunique(dropna=True)) <= 1
        return {
            "model_type": self.model_type,
            "params": dict(self.model_params),
            "model_version": self.model_version,
            "feature_set_id": self.feature_set_id,
            "features": list(self.features),
            "fitted": self.model is not None,
            "fitted_segment": self.fitted_segment,
            "feature_importance_source": importance_source or "unavailable",
            "top_features": top_features,
            "constant_prediction": constant_prediction,
            "zero_feature_importance": zero_feature_importance,
            "metrics": dict(self.metrics),
        }

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "feature_set_id": self.feature_set_id,
            "params": dict(self.model_params),
        }
