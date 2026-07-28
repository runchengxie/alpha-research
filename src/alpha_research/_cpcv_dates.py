"""Date, window, and JSON-helper utilities for CPCV (private helpers).

These helpers are part of the public-facing ``alpha_research.cpcv`` module's
surface (some are referenced by ``cpcv_audit`` and ``artifact_cpcv``) but are
kept here to keep the historical ``cpcv.py`` file smaller. They are re-exported
from ``alpha_research.cpcv`` so existing imports keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelEventWindow:
    signal_date: pd.Timestamp
    label_start: pd.Timestamp
    label_end: pd.Timestamp


def _date_key(date: Any) -> pd.Timestamp:
    return pd.Timestamp(date).normalize()


def _as_date_tuple(dates: Any) -> tuple[pd.Timestamp, ...]:
    values = pd.to_datetime(
        list(dates) if not isinstance(dates, pd.Series) else dates, errors="coerce"
    )
    cleaned = [pd.Timestamp(date).normalize() for date in values if not pd.isna(date)]
    return tuple(pd.Index(cleaned).drop_duplicates().sort_values())


def _format_date(date: Any) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def _format_dates(dates: tuple[pd.Timestamp, ...] | list[pd.Timestamp]) -> str:
    return "|".join(_format_date(date) for date in dates)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return _format_date(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _lookup_shifted_date(
    date: pd.Timestamp,
    all_dates: tuple[pd.Timestamp, ...],
    shift_days: int,
) -> pd.Timestamp | None:
    try:
        idx = all_dates.index(date)
    except ValueError:
        return None
    shifted_idx = idx + max(0, int(shift_days))
    if shifted_idx >= len(all_dates):
        return None
    return all_dates[shifted_idx]
