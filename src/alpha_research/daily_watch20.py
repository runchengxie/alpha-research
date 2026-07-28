"""Daily cross-sectional XGBRanker research for prepared stock-date features."""

from __future__ import annotations

from ._daily_watch20_label import (
    PREPARED_FEATURE_POLICY_ID,
    PRICE_ONLY_LABEL_POLICY_ID,
    RELATIVE_PERCENTILE_COL,
    DailyWatch20Config,
    DailyWatch20Explanation,
    DailyWatch20TrainingSummary,
    build_forward_rank_label,
    build_multi_horizon_forward_rank_label,
)
from ._daily_watch20_ranker import DailyWatch20Ranker

# Monkeypatch-compatible module attributes: bind build_model/fit_model at the top
# of the shell so tests can patch ``daily_watch20.build_model`` / ``fit_model``
# and have the ranker resolve them dynamically via ``from . import daily_watch20``.
from .modeling import build_model, fit_model

__all__ = [
    "PREPARED_FEATURE_POLICY_ID",
    "PRICE_ONLY_LABEL_POLICY_ID",
    "RELATIVE_PERCENTILE_COL",
    "DailyWatch20Config",
    "DailyWatch20Explanation",
    "DailyWatch20Ranker",
    "DailyWatch20TrainingSummary",
    "build_forward_rank_label",
    "build_model",
    "build_multi_horizon_forward_rank_label",
    "fit_model",
]
