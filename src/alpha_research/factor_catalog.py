"""Versioned factor identity and evidence catalog owned by alpha research."""

from __future__ import annotations

import re
from collections.abc import Buffer
from dataclasses import dataclass, field
from datetime import date
from math import isfinite
from typing import Any, SupportsFloat, SupportsIndex, cast

FACTOR_CATALOG_SCHEMA = "alpha_research.factor_catalog.v1"
FACTOR_LIFECYCLE_STATUSES = frozenset({"research", "candidate", "production", "retired"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FactorKey = tuple[str, str]


def _text(value: object, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _unique_text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list/tuple")
    values = tuple(_text(item, f"{field_name}[]") for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")
    return values


def _optional_finite(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    number = float(cast(str | Buffer | SupportsFloat | SupportsIndex, value))
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _optional_ic(value: object, field_name: str) -> float | None:
    number = _optional_finite(value, field_name)
    if number is not None and not -1.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [-1, 1]")
    return number


@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    version: str
    owner: str
    frequency: str
    dependencies: tuple[str, ...]
    pit_semantics: str
    universe_semantics: str
    implementation_sha256: str
    preprocessing: tuple[str, ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor_id", _text(self.factor_id, "factor_id"))
        object.__setattr__(self, "version", _text(self.version, "version"))
        object.__setattr__(self, "owner", _text(self.owner, "owner"))
        object.__setattr__(self, "frequency", _text(self.frequency, "frequency"))
        object.__setattr__(
            self,
            "dependencies",
            _unique_text_tuple(self.dependencies, "dependencies"),
        )
        object.__setattr__(
            self,
            "preprocessing",
            _unique_text_tuple(self.preprocessing, "preprocessing"),
        )
        object.__setattr__(self, "pit_semantics", _text(self.pit_semantics, "pit_semantics"))
        object.__setattr__(
            self,
            "universe_semantics",
            _text(self.universe_semantics, "universe_semantics"),
        )
        implementation_sha256 = _text(self.implementation_sha256, "implementation_sha256")
        if _SHA256.fullmatch(implementation_sha256) is None:
            raise ValueError("implementation_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "implementation_sha256", implementation_sha256)
        if self.description is not None:
            object.__setattr__(self, "description", _text(self.description, "description"))

    @property
    def key(self) -> FactorKey:
        return (self.factor_id, self.version)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "version": self.version,
            "owner": self.owner,
            "frequency": self.frequency,
            "dependencies": list(self.dependencies),
            "pit_semantics": self.pit_semantics,
            "universe_semantics": self.universe_semantics,
            "implementation_sha256": self.implementation_sha256,
            "preprocessing": list(self.preprocessing),
            "description": self.description,
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> FactorSpec:
        return cls(
            factor_id=payload.get("factor_id", ""),
            version=payload.get("version", ""),
            owner=payload.get("owner", ""),
            frequency=payload.get("frequency", ""),
            dependencies=_unique_text_tuple(payload.get("dependencies", ()), "dependencies"),
            pit_semantics=payload.get("pit_semantics", ""),
            universe_semantics=payload.get("universe_semantics", ""),
            implementation_sha256=payload.get("implementation_sha256", ""),
            preprocessing=_unique_text_tuple(payload.get("preprocessing", ()), "preprocessing"),
            description=payload.get("description"),
        )


@dataclass(frozen=True)
class FactorEvidenceSummary:
    as_of: date
    observations: int
    status: str
    ic_mean: float | None = None
    rank_ic_mean: float | None = None
    icir: float | None = None
    turnover: float | None = None
    neutralized_rank_ic_mean: float | None = None
    decay_horizon_days: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.as_of, date):
            raise TypeError("as_of must be a date")
        if isinstance(self.observations, bool) or not isinstance(self.observations, int):
            raise TypeError("observations must be an integer")
        if self.observations <= 0:
            raise ValueError("observations must be > 0")
        status = _text(self.status, "status")
        if status not in FACTOR_LIFECYCLE_STATUSES:
            raise ValueError(
                "status must be one of " + ", ".join(sorted(FACTOR_LIFECYCLE_STATUSES))
            )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "ic_mean", _optional_ic(self.ic_mean, "ic_mean"))
        object.__setattr__(
            self,
            "rank_ic_mean",
            _optional_ic(self.rank_ic_mean, "rank_ic_mean"),
        )
        object.__setattr__(self, "icir", _optional_finite(self.icir, "icir"))
        object.__setattr__(
            self,
            "neutralized_rank_ic_mean",
            _optional_ic(self.neutralized_rank_ic_mean, "neutralized_rank_ic_mean"),
        )
        turnover = _optional_finite(self.turnover, "turnover")
        if turnover is not None and turnover < 0:
            raise ValueError("turnover must be >= 0")
        object.__setattr__(self, "turnover", turnover)
        if self.decay_horizon_days is not None:
            if isinstance(self.decay_horizon_days, bool) or not isinstance(
                self.decay_horizon_days, int
            ):
                raise ValueError("decay_horizon_days must be a non-negative integer")
            if self.decay_horizon_days < 0:
                raise ValueError("decay_horizon_days must be >= 0")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "observations": self.observations,
            "status": self.status,
            "ic_mean": self.ic_mean,
            "rank_ic_mean": self.rank_ic_mean,
            "icir": self.icir,
            "turnover": self.turnover,
            "neutralized_rank_ic_mean": self.neutralized_rank_ic_mean,
            "decay_horizon_days": self.decay_horizon_days,
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> FactorEvidenceSummary:
        return cls(
            as_of=date.fromisoformat(_text(payload.get("as_of"), "as_of")),
            observations=payload.get("observations", 0),
            status=payload.get("status", ""),
            ic_mean=payload.get("ic_mean"),
            rank_ic_mean=payload.get("rank_ic_mean"),
            icir=payload.get("icir"),
            turnover=payload.get("turnover"),
            neutralized_rank_ic_mean=payload.get("neutralized_rank_ic_mean"),
            decay_horizon_days=payload.get("decay_horizon_days"),
        )


@dataclass
class FactorCatalog:
    _specs: dict[FactorKey, FactorSpec] = field(default_factory=dict)
    _evidence: dict[FactorKey, list[FactorEvidenceSummary]] = field(default_factory=dict)

    def register(self, spec: FactorSpec) -> None:
        if spec.key in self._specs:
            raise ValueError(f"factor version already registered: {spec.factor_id}@{spec.version}")
        self._specs[spec.key] = spec
        self._evidence[spec.key] = []

    def add_evidence(self, key: FactorKey, evidence: FactorEvidenceSummary) -> None:
        if key not in self._specs:
            raise KeyError(f"unknown factor version: {key[0]}@{key[1]}")
        existing = self._evidence[key]
        if any(item.as_of == evidence.as_of for item in existing):
            raise ValueError(f"evidence already exists for {key[0]}@{key[1]} on {evidence.as_of}")
        existing.append(evidence)
        existing.sort(key=lambda item: item.as_of)

    def get(self, key: FactorKey) -> FactorSpec:
        try:
            return self._specs[key]
        except KeyError as exc:
            raise KeyError(f"unknown factor version: {key[0]}@{key[1]}") from exc

    def evidence(self, key: FactorKey) -> tuple[FactorEvidenceSummary, ...]:
        self.get(key)
        return tuple(self._evidence[key])

    def versions(self, factor_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(version for current_id, version in self._specs if current_id == factor_id)
        )

    def to_mapping(self) -> dict[str, Any]:
        entries = []
        for key in sorted(self._specs):
            spec = self._specs[key]
            entries.append(
                {
                    "spec": spec.to_mapping(),
                    "evidence": [item.to_mapping() for item in self._evidence[key]],
                }
            )
        return {"schema_version": FACTOR_CATALOG_SCHEMA, "entries": entries}

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> FactorCatalog:
        if payload.get("schema_version") != FACTOR_CATALOG_SCHEMA:
            raise ValueError(
                f"unsupported factor catalog schema: {payload.get('schema_version')!r}"
            )
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("entries must be a list")
        catalog = cls()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"entries[{index}] must be an object")
            spec_payload = entry.get("spec")
            if not isinstance(spec_payload, dict):
                raise ValueError(f"entries[{index}].spec must be an object")
            spec = FactorSpec.from_mapping(cast(dict[str, Any], spec_payload))
            catalog.register(spec)
            evidence_payload = entry.get("evidence", [])
            if not isinstance(evidence_payload, list):
                raise ValueError(f"entries[{index}].evidence must be a list")
            for raw_evidence in evidence_payload:
                if not isinstance(raw_evidence, dict):
                    raise ValueError(f"entries[{index}].evidence items must be objects")
                catalog.add_evidence(
                    spec.key,
                    FactorEvidenceSummary.from_mapping(cast(dict[str, Any], raw_evidence)),
                )
        return catalog


__all__ = [
    "FACTOR_CATALOG_SCHEMA",
    "FACTOR_LIFECYCLE_STATUSES",
    "FactorCatalog",
    "FactorEvidenceSummary",
    "FactorKey",
    "FactorSpec",
]
