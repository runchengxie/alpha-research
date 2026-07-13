from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class ArtifactIntegrityError(ValueError):
    """Raised when a persistent research artifact cannot be trusted or replayed."""


def canonical_research_json_value(value: Any, *, field: str = "value") -> JsonValue:
    """Project standard scientific scalars to JSON while rejecting opaque objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} contains a non-string mapping key: {key!r}")
            result[key] = canonical_research_json_value(item, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            canonical_research_json_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{field} contains unsupported {type(value).__module__}.{type(value).__qualname__}; "
        "framework objects cannot enter research artifacts"
    )


def strict_json_value(value: Any, *, field: str = "value") -> JsonValue:
    """Return a detached JSON value and reject opaque/framework objects.

    Research governance artifacts must never rely on ``default=str`` or pickle-like
    fallback behaviour.  Callers must explicitly project third-party values into
    canonical scalars before reaching this boundary.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise TypeError(f"{field} must contain finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} contains a non-string mapping key: {key!r}")
            result[key] = strict_json_value(item, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            strict_json_value(item, field=f"{field}[{index}]") for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{field} contains unsupported {type(value).__module__}.{type(value).__qualname__}; "
        "project it to canonical JSON before persistence"
    )


def strict_json_mapping(value: Mapping[str, Any], *, field: str = "value") -> dict[str, JsonValue]:
    normalized = strict_json_value(value, field=field)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded by the input annotation
        raise TypeError(f"{field} must be a mapping")
    return normalized


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_strict_json_value(path: str | Path, payload: Any) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = strict_json_value(payload, field="artifact")
    output_path.write_text(
        json.dumps(normalized, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_strict_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return write_strict_json_value(path, payload)


def write_canonical_json_value(path: str | Path, payload: Any) -> Path:
    return write_strict_json_value(
        path,
        canonical_research_json_value(payload, field="artifact"),
    )


def load_json_mapping(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Cannot read JSON artifact {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError(f"JSON artifact must contain a mapping: {resolved}")
    return payload


@dataclass(frozen=True)
class ArtifactHandle:
    """Framework-neutral, integrity-addressed reference to a persistent artifact."""

    artifact_type: str
    schema_version: str
    path: str
    sha256: str
    media_type: str = "application/json"

    def __post_init__(self) -> None:
        if not self.artifact_type.strip():
            raise ValueError("artifact_type cannot be empty")
        if not self.schema_version.strip():
            raise ValueError("schema_version cannot be empty")
        if not self.path.strip():
            raise ValueError("path cannot be empty")
        if not self.media_type.strip():
            raise ValueError("media_type cannot be empty")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("sha256 must be a lowercase hexadecimal SHA-256 digest")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        artifact_type: str,
        schema_version: str,
        media_type: str = "application/json",
    ) -> ArtifactHandle:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise ArtifactIntegrityError(f"Artifact file does not exist: {resolved}")
        return cls(
            artifact_type=artifact_type,
            schema_version=schema_version,
            path=str(resolved),
            sha256=sha256_file(resolved),
            media_type=media_type,
        )

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> ArtifactHandle:
        normalized = strict_json_mapping(payload, field="artifact_handle")

        def required_string(key: str, default: str | None = None) -> str:
            value = normalized.get(key, default)
            if not isinstance(value, str):
                raise ArtifactIntegrityError(f"Artifact handle field {key!r} must be a string")
            return value

        try:
            return cls(
                artifact_type=required_string("artifact_type"),
                schema_version=required_string("schema_version"),
                path=required_string("path"),
                sha256=required_string("sha256"),
                media_type=required_string("media_type", "application/json"),
            )
        except ValueError as exc:
            raise ArtifactIntegrityError(f"Invalid artifact handle: {exc}") from exc

    def verify(self) -> Path:
        resolved = Path(self.path).expanduser().resolve()
        if not resolved.is_file():
            raise ArtifactIntegrityError(f"Artifact file does not exist: {resolved}")
        actual = sha256_file(resolved)
        if actual != self.sha256:
            raise ArtifactIntegrityError(
                f"Artifact digest mismatch for {resolved}: expected {self.sha256}, got {actual}"
            )
        return resolved

    def to_metadata(self) -> dict[str, JsonValue]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "path": self.path,
            "sha256": self.sha256,
            "media_type": self.media_type,
        }


__all__ = [
    "ArtifactHandle",
    "ArtifactIntegrityError",
    "JsonScalar",
    "JsonValue",
    "canonical_research_json_value",
    "load_json_mapping",
    "sha256_file",
    "strict_json_mapping",
    "strict_json_value",
    "write_canonical_json_value",
    "write_strict_json",
    "write_strict_json_value",
]
