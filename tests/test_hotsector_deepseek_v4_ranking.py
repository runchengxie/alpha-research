from __future__ import annotations

from alpha_research.hotsector_deepseek_v4_ranking import (
    DeepSeekV4RankingPolicy,
    analyze_deepseek_v4_phase1,
    build_intention_to_deploy_scores,
)


def _policy() -> DeepSeekV4RankingPolicy:
    return DeepSeekV4RankingPolicy(
        models=("model-a",),
        sample_dates=("2026-01-02",),
        screen_dates=("2026-01-02",),
        screen_arms=("canonical", "shuffle", "opaque"),
        canonical_arm="canonical",
        shuffle_arm="shuffle",
        opaque_arm="opaque",
        top_k=2,
        numeric_variant="NUMERIC",
        model_variants=(("model-a", "MODEL_A"),),
        relative_percentile_column="relative_percentile",
        max_invalid_model_responses_per_model=0,
        max_invalid_ranking_contracts_per_model=0,
        max_invalid_pair_dates_per_model=0,
        shuffle_minimum_pair_overlap=2,
        shuffle_minimum_mean_top_k_overlap=2.0,
        opaque_minimum_pair_overlap=2,
        opaque_minimum_mean_top_k_overlap=2.0,
        publication_contract_required=True,
    )


def _trial(arm: str, order: list[str] | None = None) -> dict[str, object]:
    return {
        "date": "2026-01-02",
        "model": "model-a",
        "arm": arm,
        "actual_model": "model-a",
        "ranking_contract_valid": True,
        "publication_contract_valid": True,
        "ranking_order": order or ["000001.SZ", "600000.SH"],
    }


def test_analyze_deepseek_v4_phase1_uses_explicit_policy_only() -> None:
    report = analyze_deepseek_v4_phase1(
        [_trial("canonical"), _trial("shuffle"), _trial("opaque")],
        _policy(),
    )

    assert report["proceed_to_execution"] is True
    assert all(report["gates"].values())
    assert report["model_metrics"]["model-a"]["shuffle_overlaps"] == [2]


def test_build_intention_to_deploy_scores_falls_back_to_numeric() -> None:
    invalid = _trial("canonical", ["000002.SZ", "600001.SH"])
    invalid["actual_model"] = "wrong-model"

    scores, contracts = build_intention_to_deploy_scores(
        [invalid],
        {"2026-01-02": ("000001.SZ", "600000.SH")},
        _policy(),
    )

    model_scores = scores.loc[scores["variant"].eq("MODEL_A")]
    assert model_scores["symbol"].tolist() == ["000001.SZ", "600000.SH"]
    assert model_scores["relative_percentile"].tolist() == [1.0, 0.5]
    assert contracts.loc[0, "fallback_used"]
    assert contracts.loc[0, "fallback_reason"] == "invalid_canonical_ranking_contract"
