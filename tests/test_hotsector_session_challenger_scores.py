from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
import pytest

from alpha_research.daily_watch20 import RELATIVE_PERCENTILE_COL
from alpha_research.hotsector_session_challenger_scores import build_session_challenger_scores


@dataclass(frozen=True)
class _Selection:
    trade_date: pd.Timestamp
    numeric_symbols: tuple[str, ...]
    selected_symbols: tuple[str, ...]
    confidence_scores: tuple[int, ...]
    candidate_sha256: str = "a" * 64
    plan_sha256: str = "b" * 64


@dataclass(frozen=True)
class _Manifest:
    selections: tuple[_Selection, ...]


def _selection() -> _Selection:
    numeric = tuple(f"{index:06d}.SZ" for index in range(1, 11))
    selected = (*numeric[:7], "000011.SZ", "000012.SZ", "000013.SZ")
    return _Selection(
        trade_date=cast(pd.Timestamp, pd.Timestamp("2026-05-06")),
        numeric_symbols=numeric,
        selected_symbols=selected,
        confidence_scores=tuple(range(91, 101)),
    )


def test_build_session_challenger_scores_preserves_frozen_row_contract() -> None:
    selection = _selection()
    scores = build_session_challenger_scores(
        _Manifest((selection,)),
        numeric_variant="NUMERIC",
        challenger_variant="GPT_CODEX_SESSION",
    )

    assert len(scores) == 20
    assert set(scores["variant"]) == {"NUMERIC", "GPT_CODEX_SESSION"}

    numeric = scores.loc[scores["variant"].eq("NUMERIC")]
    challenger = scores.loc[scores["variant"].eq("GPT_CODEX_SESSION")]
    assert numeric["symbol"].tolist() == list(selection.numeric_symbols)
    assert challenger["symbol"].tolist() == list(selection.selected_symbols)
    assert numeric["confidence_score"].isna().all()
    assert challenger["confidence_score"].tolist() == list(selection.confidence_scores)
    assert challenger[RELATIVE_PERCENTILE_COL].tolist() == [
        float(value) for value in range(10, 0, -1)
    ]
    assert challenger["available_at"].dt.hour.eq(15).all()
    assert challenger["available_at"].dt.minute.eq(1).all()
    assert str(challenger["available_at"].dt.tz) == "Asia/Shanghai"
    assert challenger["candidate_sha256"].eq("a" * 64).all()
    assert challenger["plan_sha256"].eq("b" * 64).all()


def test_build_session_challenger_scores_rejects_incomplete_arm() -> None:
    selection = _selection()
    incomplete = _Selection(
        trade_date=selection.trade_date,
        numeric_symbols=selection.numeric_symbols[:-1],
        selected_symbols=selection.selected_symbols,
        confidence_scores=selection.confidence_scores,
    )

    with pytest.raises(ValueError, match="do not match top_k"):
        build_session_challenger_scores(
            _Manifest((incomplete,)),
            numeric_variant="NUMERIC",
            challenger_variant="GPT_CODEX_SESSION",
        )
