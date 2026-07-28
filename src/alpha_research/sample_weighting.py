"""Sample weighting and sequential bootstrap for overlapping financial labels.

Re-exports the public API implemented in :mod:`._sample_weighting_core` and
the private computation helpers from :mod:`._sample_weighting_helpers`.
"""

from __future__ import annotations

from ._sample_weighting_core import (
    SampleWeightConfig,
    SampleWeightReceipt,
    WeightMode,
    average_uniqueness,
    build_event_sample_weights,
    build_indicator_matrix,
    event_concurrency,
    return_attribution_weights,
    sequential_bootstrap,
    time_decay_weights,
    write_sample_weight_artifacts,
)
from ._sample_weighting_helpers import (
    _effective_sample_size,
    _events_hash,
    _group_bar_index,
    _group_returns,
    _grouped_interval_weights,
    _interval_weights,
    _normalize_events,
    _normalize_mean_one,
    _resolve_bars,
    _weight_hhi,
)

__all__ = [
    "SampleWeightConfig",
    "SampleWeightReceipt",
    "WeightMode",
    "_effective_sample_size",
    "_events_hash",
    "_group_bar_index",
    "_group_returns",
    "_grouped_interval_weights",
    "_interval_weights",
    "_normalize_events",
    "_normalize_mean_one",
    "_resolve_bars",
    "_weight_hhi",
    "average_uniqueness",
    "build_event_sample_weights",
    "build_indicator_matrix",
    "event_concurrency",
    "return_attribution_weights",
    "sequential_bootstrap",
    "time_decay_weights",
    "write_sample_weight_artifacts",
]
