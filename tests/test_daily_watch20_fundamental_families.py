from alpha_research.daily_watch20_pit_features import GROWTH_FEATURES, QUALITY_FEATURES


def test_family_registry_reuses_canonical_qg_and_has_no_overlap() -> None:
    from alpha_research.daily_watch20_fundamental_families import (
        FUNDAMENTAL_FAMILY_SCHEMA,
        FUND_CONTEXT_FEATURES,
        STYLE_CONTROL_FEATURES,
        VALUE_FEATURES,
        fundamental_family_registry,
    )

    registry = fundamental_family_registry()
    assert FUNDAMENTAL_FAMILY_SCHEMA == "daily_watch20.fundamental_families.research.v1"
    assert registry["value"] == VALUE_FEATURES
    assert registry["quality"] == QUALITY_FEATURES
    assert registry["growth"] == GROWTH_FEATURES
    assert registry["style_controls"] == STYLE_CONTROL_FEATURES
    assert registry["fund_context"] == FUND_CONTEXT_FEATURES

    primary_names = [
        name
        for family in ("value", "quality", "growth")
        for name in registry[family]
    ]
    assert len(primary_names) == len(set(primary_names))
