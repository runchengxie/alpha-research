from __future__ import annotations

from datetime import date

import pytest

from alpha_research.factor_catalog import (
    FactorCatalog,
    FactorEvidenceSummary,
    FactorSpec,
)

SHA = "a" * 64


def _spec(version: str = "v1") -> FactorSpec:
    return FactorSpec(
        factor_id="momentum_12_1",
        version=version,
        owner="alpha-research",
        frequency="daily",
        dependencies=("close", "adj_factor"),
        pit_semantics="formation date uses only prices available by close",
        universe_semantics="formation-date eligible A-share universe",
        preprocessing=("winsorize", "industry_neutralize", "zscore"),
        implementation_sha256=SHA,
    )


def test_catalog_round_trip_preserves_factor_identity_and_evidence() -> None:
    catalog = FactorCatalog()
    spec = _spec()
    catalog.register(spec)
    catalog.add_evidence(
        spec.key,
        FactorEvidenceSummary(
            as_of=date(2026, 9, 1),
            observations=1200,
            status="candidate",
            rank_ic_mean=0.041,
            icir=0.63,
            turnover=0.32,
            neutralized_rank_ic_mean=0.028,
            decay_horizon_days=5,
        ),
    )

    payload = catalog.to_mapping()
    restored = FactorCatalog.from_mapping(payload)

    assert restored.get(spec.key) == spec
    assert restored.evidence(spec.key)[0].rank_ic_mean == pytest.approx(0.041)
    assert payload["schema_version"] == "alpha_research.factor_catalog.v1"


def test_catalog_serialization_is_stable_across_registration_order() -> None:
    first = FactorCatalog()
    first.register(_spec("v2"))
    first.register(_spec("v1"))
    second = FactorCatalog()
    second.register(_spec("v1"))
    second.register(_spec("v2"))

    assert first.to_mapping() == second.to_mapping()


def test_catalog_rejects_duplicate_factor_version() -> None:
    catalog = FactorCatalog()
    catalog.register(_spec())

    with pytest.raises(ValueError, match="already registered"):
        catalog.register(_spec())


def test_factor_spec_rejects_duplicate_dependencies_and_bad_hash() -> None:
    with pytest.raises(ValueError, match="dependencies"):
        FactorSpec(
            factor_id="bad",
            version="v1",
            owner="alpha-research",
            frequency="daily",
            dependencies=("close", "close"),
            pit_semantics="pit",
            universe_semantics="universe",
            implementation_sha256=SHA,
        )

    with pytest.raises(ValueError, match="implementation_sha256"):
        FactorSpec(
            factor_id="bad",
            version="v1",
            owner="alpha-research",
            frequency="daily",
            dependencies=("close",),
            pit_semantics="pit",
            universe_semantics="universe",
            implementation_sha256="abc",
        )


def test_evidence_rejects_nonfinite_statistics() -> None:
    with pytest.raises(ValueError, match="finite"):
        FactorEvidenceSummary(
            as_of=date(2026, 9, 1),
            observations=100,
            status="research",
            rank_ic_mean=float("nan"),
        )


def test_evidence_rejects_invalid_ic_turnover_and_decay() -> None:
    with pytest.raises(ValueError, match="rank_ic_mean"):
        FactorEvidenceSummary(
            as_of=date(2026, 9, 1),
            observations=100,
            status="research",
            rank_ic_mean=1.01,
        )
    with pytest.raises(ValueError, match="turnover"):
        FactorEvidenceSummary(
            as_of=date(2026, 9, 1),
            observations=100,
            status="research",
            turnover=-0.01,
        )
    with pytest.raises(ValueError, match="decay_horizon_days"):
        FactorEvidenceSummary(
            as_of=date(2026, 9, 1),
            observations=100,
            status="research",
            decay_horizon_days=1.5,
        )
