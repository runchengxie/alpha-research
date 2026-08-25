"""Pure score-frame builder for bounded session challenger selections.

The strategy-specific manifest validation and variant identities remain with the
application layer.  This module owns only the deterministic transformation from
validated selection records to timestamped score rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pandas as pd

from .daily_watch20 import RELATIVE_PERCENTILE_COL


class SessionSelectionLike(Protocol):
    """Read-only fields required from one validated selection record."""

    @property
    def trade_date(self) -> pd.Timestamp: ...

    @property
    def numeric_symbols(self) -> Sequence[str]: ...

    @property
    def selected_symbols(self) -> Sequence[str]: ...

    @property
    def confidence_scores(self) -> Sequence[int]: ...

    @property
    def candidate_sha256(self) -> str: ...

    @property
    def plan_sha256(self) -> str: ...


class SessionChallengerManifestLike(Protocol):
    """Read-only manifest surface required by the score builder."""

    @property
    def selections(self) -> Sequence[SessionSelectionLike]: ...


def build_session_challenger_scores(
    manifest: SessionChallengerManifestLike,
    *,
    numeric_variant: str,
    challenger_variant: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """Materialize paired equal-weight score rows from validated selections.

    Variant identities are supplied by the caller so the compute kernel does not
    depend on a strategy-app contract.  ``top_k`` controls both the required
    symbol count per arm and the descending rank score ``top_k .. 1``.
    """

    if not numeric_variant.strip() or not challenger_variant.strip():
        raise ValueError("session challenger variants must be non-empty")
    if numeric_variant == challenger_variant:
        raise ValueError("session challenger variants must be distinct")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not manifest.selections:
        raise ValueError("session challenger manifest must contain selections")

    rows: list[dict[str, object]] = []
    for selection in manifest.selections:
        if len(selection.numeric_symbols) != top_k or len(selection.selected_symbols) != top_k:
            raise ValueError("session challenger selections do not match top_k")
        if len(selection.confidence_scores) != top_k:
            raise ValueError("session challenger confidence scores do not match top_k")

        confidence = dict(
            zip(selection.selected_symbols, selection.confidence_scores, strict=True)
        )
        available_at = selection.trade_date.tz_localize("Asia/Shanghai") + pd.Timedelta(
            hours=15, minutes=1
        )
        for variant, symbols in (
            (numeric_variant, selection.numeric_symbols),
            (challenger_variant, selection.selected_symbols),
        ):
            for rank, symbol in enumerate(symbols, start=1):
                rows.append(
                    {
                        "trade_date": selection.trade_date,
                        "symbol": symbol,
                        "variant": variant,
                        "selection_rank": rank,
                        RELATIVE_PERCENTILE_COL: float(top_k + 1 - rank),
                        "available_at": available_at,
                        "confidence_score": (
                            confidence.get(symbol) if variant == challenger_variant else None
                        ),
                        "candidate_sha256": selection.candidate_sha256,
                        "plan_sha256": selection.plan_sha256,
                    }
                )

    scores = pd.DataFrame(rows)
    expected_rows = len(manifest.selections) * 2 * top_k
    if len(scores) != expected_rows or scores.duplicated(
        ["trade_date", "variant", "symbol"]
    ).any():
        raise ValueError("session challenger scores are incomplete or duplicated")
    return scores.sort_values(
        ["trade_date", "variant", "selection_rank"], kind="mergesort"
    ).reset_index(drop=True)


__all__ = [
    "SessionChallengerManifestLike",
    "SessionSelectionLike",
    "build_session_challenger_scores",
]
