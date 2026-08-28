from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from .exposures import ExposureSpec
from .interactions import ContextInteractionSpec
from .transforms import ContextTransformSpec

_CONTEXTUAL_FEATURE_SCHEMA = "alpha_research.contextual_feature_set.v1"


@dataclass(frozen=True)
class ContextualFeatureEvidence:
    feature_set_id: str
    transform_features: tuple[str, ...]
    exposure_versions: Mapping[str, str]
    interaction_features: tuple[str, ...]
    context_series_ids: tuple[str, ...]
    missing_by_reason: Mapping[str, int]
    max_context_age_days: float | None

    def __post_init__(self) -> None:
        if len(str(self.feature_set_id)) != 64:
            raise ValueError("ContextualFeatureEvidence.feature_set_id must be SHA-256 hex")
        if self.max_context_age_days is not None:
            value = float(self.max_context_age_days)
            if not math.isfinite(value) or value < 0:
                raise ValueError("max_context_age_days must be finite and non-negative")


def _canonical(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, tuple | list):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("contextual feature specs cannot contain non-finite floats")
        return value
    raise TypeError(f"unsupported contextual feature identity value: {type(value).__name__}")


def contextual_feature_set_payload(
    transform_specs: Sequence[ContextTransformSpec],
    exposure_specs: Sequence[ExposureSpec],
    interaction_specs: Sequence[ContextInteractionSpec],
) -> dict[str, Any]:
    return {
        "schema": _CONTEXTUAL_FEATURE_SCHEMA,
        "transforms": [_canonical(spec) for spec in transform_specs],
        "exposures": [_canonical(spec) for spec in exposure_specs],
        "interactions": [_canonical(spec) for spec in interaction_specs],
    }


def contextual_feature_set_id(
    transform_specs: Sequence[ContextTransformSpec],
    exposure_specs: Sequence[ExposureSpec],
    interaction_specs: Sequence[ContextInteractionSpec],
) -> str:
    payload = contextual_feature_set_payload(
        transform_specs,
        exposure_specs,
        interaction_specs,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_contextual_feature_evidence(
    transform_specs: Sequence[ContextTransformSpec],
    exposure_specs: Sequence[ExposureSpec],
    interaction_specs: Sequence[ContextInteractionSpec],
    *,
    missing_by_reason: Mapping[str, int] | None = None,
    max_context_age_days: float | None = None,
) -> ContextualFeatureEvidence:
    return ContextualFeatureEvidence(
        feature_set_id=contextual_feature_set_id(
            transform_specs,
            exposure_specs,
            interaction_specs,
        ),
        transform_features=tuple(spec.feature_name for spec in transform_specs),
        exposure_versions={spec.name: spec.version for spec in exposure_specs},
        interaction_features=tuple(spec.output_name for spec in interaction_specs),
        context_series_ids=tuple(dict.fromkeys(spec.series_id for spec in transform_specs)),
        missing_by_reason=dict(missing_by_reason or {}),
        max_context_age_days=max_context_age_days,
    )
