"""Owner-native feature, label, and model policy for DailyWatch20."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .daily_watch20_features import (
    DAILY_WATCH20_FEATURES,
    DEFAULT_LABEL_HORIZON_WEIGHTS,
    LIMIT_AWARE_NEXT_OPEN_LABEL_POLICY_ID,
    normalize_label_horizon_weights,
)

ALPHA_POLICY_SCHEMA = "daily_watch20.alpha_policy.v1"


@dataclass(frozen=True, slots=True)
class DailyWatch20AlphaPolicy:
    features: tuple[str, ...] = DAILY_WATCH20_FEATURES
    label_horizon_weights: tuple[tuple[int, float], ...] = DEFAULT_LABEL_HORIZON_WEIGHTS
    label_policy_id: str = LIMIT_AWARE_NEXT_OPEN_LABEL_POLICY_ID
    train_window_dates: int = 504
    retrain_weekdays: tuple[int, ...] = (0,)
    max_model_age_trade_days: int = 10
    model_family: str = "xgboost_rank_pairwise"
    sample_weight_mode: str = "exp_decay"
    decay_halflife_dates: float = 126.0

    def __post_init__(self) -> None:
        if not self.features or len(self.features) != len(set(self.features)):
            raise ValueError("features must be non-empty and unique")
        normalized = normalize_label_horizon_weights(self.label_horizon_weights)
        object.__setattr__(self, "label_horizon_weights", normalized)
        if self.train_window_dates <= 0:
            raise ValueError("train_window_dates must be positive")
        if not self.retrain_weekdays or any(day not in range(7) for day in self.retrain_weekdays):
            raise ValueError("retrain_weekdays must contain weekdays in [0, 6]")
        if len(self.retrain_weekdays) != len(set(self.retrain_weekdays)):
            raise ValueError("retrain_weekdays must be unique")
        if self.max_model_age_trade_days <= 0:
            raise ValueError("max_model_age_trade_days must be positive")
        if self.decay_halflife_dates <= 0:
            raise ValueError("decay_halflife_dates must be positive")
        if not self.label_policy_id or not self.model_family or not self.sample_weight_mode:
            raise ValueError("policy identity fields must be non-empty")

    @property
    def policy_id(self) -> str:
        payload = {"schema_version": ALPHA_POLICY_SCHEMA, **asdict(self)}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return f"{ALPHA_POLICY_SCHEMA}:{digest}"

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": ALPHA_POLICY_SCHEMA, "policy_id": self.policy_id, **asdict(self)}


__all__ = ["ALPHA_POLICY_SCHEMA", "DailyWatch20AlphaPolicy"]
