from __future__ import annotations

import pandas as pd
import pytest

from alpha_research import date_utils


def test_normalize_date_token_aliases() -> None:
    assert date_utils.normalize_date_token("TODAY", "today") == "today"
    assert date_utils.normalize_date_token("now", "today") == "today"
    assert date_utils.normalize_date_token("Yesterday", "today") == "t-1"
    assert date_utils.normalize_date_token(None, "today") == "today"
    assert date_utils.normalize_date_token("2026-08-10", "today") == "2026-08-10"


def test_is_relative_date_token() -> None:
    assert date_utils.is_relative_date_token("today")
    assert date_utils.is_relative_date_token("t-1")
    assert date_utils.is_relative_date_token("last_trading_day")
    assert not date_utils.is_relative_date_token("2026-08-10")
    assert not date_utils.is_relative_date_token("2026/08/10")


def test_resolve_date_token_relative() -> None:
    today = date_utils.resolve_date_token("today")
    assert isinstance(today, pd.Timestamp)
    assert date_utils.resolve_date_token("t-1") == today - pd.Timedelta(days=1)


def test_resolve_date_token_explicit() -> None:
    resolved = date_utils.resolve_date_token("2026-08-10")
    assert resolved == pd.Timestamp("2026-08-10").normalize()


def test_resolve_date_token_compact() -> None:
    resolved = date_utils.resolve_date_token("20260810")
    assert resolved == pd.Timestamp("2026-08-10").normalize()


def test_resolve_date_token_invalid() -> None:
    with pytest.raises(SystemExit):
        date_utils.resolve_date_token("not-a-date")


def test_normalized_timestamp_or_none() -> None:
    assert date_utils._normalized_timestamp_or_none("2026-08-10") == pd.Timestamp(
        "2026-08-10"
    ).normalize()
    assert date_utils._normalized_timestamp_or_none("garbage") is None
    assert date_utils._normalized_timestamp_or_none(None) is None
