from __future__ import annotations

import pandas as pd
import pytest

from alpha_research.hotsector_numeric_v2_ranking import (
    HotsectorNumericV2RankingPolicy,
    bounded_visible_control_picks,
    build_hotsector_numeric_v2_rankings,
    risk_veto_visible_control_symbol,
    visible_field_control_ranking,
)


@pytest.fixture
def policy() -> HotsectorNumericV2RankingPolicy:
    return HotsectorNumericV2RankingPolicy(
        numeric_variant="NUMERIC",
        numeric_v2_variant="NUMERIC_V2",
        buffer_variant="NUMERIC_V2_BUFFER15",
        pool_variant="CANDIDATE_POOL_EQW",
        candidate_pool_size=30,
        top_k=10,
        buffer_rank=15,
        component_weights=(
            ("candidate_relevance", 0.10),
            ("daily_confirm_score", 0.30),
            ("intraday_stability_score", 0.20),
            ("liquidity_score", 0.25),
            ("trend_score", 0.15),
        ),
        ret5_threshold=0.12,
        ret5_full_excess=0.18,
        ret5_weight=0.12,
        ret10_threshold=0.20,
        ret10_full_excess=0.30,
        ret10_weight=0.10,
        amount_threshold=3.0,
        amount_full_excess=4.0,
        amount_weight=0.08,
        near_high_threshold=0.95,
        near_high_full_excess=0.05,
        short_heat_threshold=0.08,
        short_heat_full_excess=0.12,
        near_high_weight=0.10,
    )


def _candidates() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(pd.to_datetime(["2026-01-05", "2026-01-06"])):
        for rank in range(1, 32):
            daily = 1.0 - rank / 40
            if date_index == 1 and rank == 11:
                daily = 0.755
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": f"{rank:06d}.SZ",
                    "candidate_relevance": 0.9 if rank <= 30 else 0.1,
                    "candidate_score": 100.0 - rank,
                    "daily_confirm_score": daily,
                    "intraday_stability_score": daily,
                    "liquidity_score": daily,
                    "trend_score": daily,
                    "ret_5d": 0.0,
                    "ret_10d": 0.0,
                    "amount_ratio_20d": 1.0,
                    "close_to_20d_high": 0.8,
                }
            )
    return pd.DataFrame(rows)


def test_policy_rejects_inconsistent_ranks() -> None:
    with pytest.raises(ValueError, match="candidate_pool_size"):
        HotsectorNumericV2RankingPolicy(
            numeric_variant="N",
            numeric_v2_variant="V2",
            buffer_variant="B",
            pool_variant="P",
            candidate_pool_size=10,
            top_k=10,
            buffer_rank=15,
            component_weights=(
                ("candidate_relevance", 0.10),
                ("daily_confirm_score", 0.30),
                ("intraday_stability_score", 0.20),
                ("liquidity_score", 0.25),
                ("trend_score", 0.15),
            ),
            ret5_threshold=0.12,
            ret5_full_excess=0.18,
            ret5_weight=0.12,
            ret10_threshold=0.20,
            ret10_full_excess=0.30,
            ret10_weight=0.10,
            amount_threshold=3.0,
            amount_full_excess=4.0,
            amount_weight=0.08,
            near_high_threshold=0.95,
            near_high_full_excess=0.05,
            short_heat_threshold=0.08,
            short_heat_full_excess=0.12,
            near_high_weight=0.10,
        )


def test_top30_penalties_and_buffer(
    policy: HotsectorNumericV2RankingPolicy,
) -> None:
    candidates = _candidates()
    candidates.loc[
        (candidates["trade_date"] == pd.Timestamp("2026-01-05"))
        & (candidates["symbol"] == "000001.SZ"),
        "ret_5d",
    ] = 0.30

    result = build_hotsector_numeric_v2_rankings(candidates, policy)
    rankings = result.rankings

    assert "000031.SZ" not in set(rankings["symbol"])
    pool_day1 = rankings.loc[
        (rankings["variant"] == policy.pool_variant)
        & (rankings["trade_date"] == pd.Timestamp("2026-01-05"))
    ].set_index("symbol")
    assert pool_day1.at["000001.SZ", "ret_5d_penalty"] > 0

    day2_v2 = rankings.loc[
        (rankings["variant"] == policy.numeric_v2_variant)
        & (rankings["trade_date"] == pd.Timestamp("2026-01-06"))
    ]
    day2_buffer = rankings.loc[
        (rankings["variant"] == policy.buffer_variant)
        & (rankings["trade_date"] == pd.Timestamp("2026-01-06"))
    ]
    assert "000011.SZ" in set(day2_v2["symbol"])
    assert "000010.SZ" in set(day2_buffer["symbol"])
    assert "000011.SZ" not in set(day2_buffer["symbol"])


def test_visible_controls_ignore_hidden_fields(
    policy: HotsectorNumericV2RankingPolicy,
) -> None:
    boundary = _candidates().loc[lambda frame: frame["trade_date"].eq("2026-01-05")].head(8)
    baseline = visible_field_control_ranking(boundary, policy)
    tampered = boundary.copy()
    tampered["candidate_relevance"] = list(reversed(range(len(tampered))))
    tampered["candidate_score"] = list(range(len(tampered)))
    reranked = visible_field_control_ranking(tampered, policy)

    assert list(baseline["symbol"]) == list(reranked["symbol"])
    assert bounded_visible_control_picks(boundary, policy) == tuple(baseline.head(3)["symbol"])


def test_risk_veto_uses_largest_positive_penalty(
    policy: HotsectorNumericV2RankingPolicy,
) -> None:
    top10 = _candidates().loc[lambda frame: frame["trade_date"].eq("2026-01-05")].head(10)
    top10.loc[top10["symbol"].eq("000007.SZ"), "ret_5d"] = 0.30

    assert risk_veto_visible_control_symbol(top10, policy) == "000007.SZ"
