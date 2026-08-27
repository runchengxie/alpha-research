from __future__ import annotations

from dataclasses import fields

from alpha_research.style_replica import StyleReplicaConfig


def test_style_replica_alpha_config_contains_no_portfolio_policy_fields() -> None:
    names = {item.name for item in fields(StyleReplicaConfig)}

    assert names == {
        "resvol_window",
        "beta_window",
        "liquidity_window",
        "mom_short_window",
        "mom_long_window",
        "model_version",
        "feature_set_id",
    }
    assert "a_slots" not in names
    assert "b_slots" not in names
    assert "theme_quotas" not in names
    assert "b_industry_cap" not in names
    assert "overlap_policy" not in names
    assert "normal_slot_weight" not in names
    assert "max_name_weight" not in names
    assert "max_daily_replacements" not in names
