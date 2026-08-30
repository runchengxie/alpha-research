from __future__ import annotations

from alpha_research.daily_watch20_pit_features import (
    FUNDAMENTAL_FEATURE_SCHEMA,
    GROWTH_FEATURES,
    PIT_LINEAGE_SUFFIXES,
    PIT_MAX_OBSERVATION_AGE_DAYS,
    PIT_MAX_REPORT_AGE_DAYS,
    PIT_SOURCE_FIELDS,
    QUALITY_FEATURES,
    build_fundamental_feature_panel,
    build_fundamental_feature_panel_from_pit_panel,
)


def test_pit_feature_contract_is_frozen() -> None:
    assert PIT_MAX_OBSERVATION_AGE_DAYS == 3
    assert PIT_MAX_REPORT_AGE_DAYS == 250
    assert FUNDAMENTAL_FEATURE_SCHEMA == "daily_watch20.fundamental_features.research.v1"
    assert len(QUALITY_FEATURES) == 4
    assert len(GROWTH_FEATURES) == 2
    assert "roa" in PIT_SOURCE_FIELDS
    assert "report_period" in PIT_LINEAGE_SUFFIXES


def test_pit_feature_entrypoints_are_callable() -> None:
    assert callable(build_fundamental_feature_panel)
    assert callable(build_fundamental_feature_panel_from_pit_panel)


def test_family_registry_references_existing_quality_growth_contract() -> None:
    from alpha_research.daily_watch20_fundamental_families import (
        fundamental_family_registry,
    )

    registry = fundamental_family_registry()
    assert registry["quality"] == QUALITY_FEATURES
    assert registry["growth"] == GROWTH_FEATURES
