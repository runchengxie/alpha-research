from __future__ import annotations

import pandas as pd

from alpha_research.fundamental_candidate import (
    FundamentalCandidateSpec,
    apply_candidate_buffer,
    build_fundamental_candidate_score,
)
from alpha_research.fundamental_state import FundamentalScoreSpec


def test_candidate_score_and_buffer_distinguish_entries_from_incumbents() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": ["2026-01-01"] * 4,
            "symbol": ["A", "B", "C", "D"],
            "quality": [0.9, 0.8, 0.4, 0.1],
            "value": [0.8, 0.7, 0.3, 0.2],
        }
    )
    spec = FundamentalCandidateSpec(
        score_specs=(
            FundamentalScoreSpec("quality"),
            FundamentalScoreSpec("value"),
        ),
        top_quantile=0.50,
        buffer_quantile=0.75,
    )

    scored = build_fundamental_candidate_score(frame, spec)
    result = apply_candidate_buffer(
        scored,
        previous_holdings=pd.DataFrame({"symbol": ["C"]}),
        top_quantile=spec.top_quantile,
        buffer_quantile=spec.buffer_quantile,
    )

    assert result.loc[result["symbol"].eq("A"), "candidate_entry"].item()
    assert not result.loc[result["symbol"].eq("C"), "candidate_entry"].item()
    assert result.loc[result["symbol"].eq("C"), "requalified"].item()
    assert not result.loc[result["symbol"].eq("D"), "requalified"].item()
