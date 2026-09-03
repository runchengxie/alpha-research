"""Research protocol manifests and evidence gates for candidate promotion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, SupportsFloat, cast

import yaml

ProtocolLevel = Literal["exploratory", "candidate", "release"]


@dataclass(frozen=True)
class EvidenceRequirement:
    name: str
    description: str
    allowed_statuses: tuple[str, ...] = ("pass",)
    any_of: tuple[str, ...] = ()
    artifact_required: bool = True


@dataclass(frozen=True)
class ProtocolPolicy:
    level: ProtocolLevel
    requirements: tuple[EvidenceRequirement, ...]
    require_event_window_purge: bool
    minimum_event_window_coverage: float
    allow_purge_fallback: bool


@dataclass(frozen=True)
class ProtocolReport:
    schema_version: int
    level: ProtocolLevel
    status: Literal["pass", "fail"]
    required_count: int
    satisfied_count: int
    missing: tuple[str, ...]
    failed: tuple[str, ...]
    warnings: tuple[str, ...]
    manifest_sha256: str
    policy: dict[str, object]
    evidence: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def protocol_policy(level: ProtocolLevel) -> ProtocolPolicy:
    """Return the canonical evidence policy for a research level."""

    base = (
        EvidenceRequirement("data_contract", "Published PIT/current data contract evidence."),
        EvidenceRequirement(
            "reproducible_run",
            "summary.json, config.used.yml, and input lineage.",
        ),
        EvidenceRequirement(
            "trial_registry",
            "Experiment ledger including failed and rejected trials.",
        ),
    )
    candidate = (
        *base,
        EvidenceRequirement(
            "feature_evidence",
            "Feature ablation, IC, importance, and correlation evidence.",
        ),
        EvidenceRequirement("weak_model_baseline", "Ridge or another low-complexity baseline."),
        EvidenceRequirement("walk_forward", "Rolling or expanding walk-forward evidence."),
        EvidenceRequirement("final_oos", "Untouched final out-of-sample evidence."),
        EvidenceRequirement("cpcv", "Combinatorial purged CV path distribution."),
        EvidenceRequirement(
            "costs_turnover_capacity",
            "Cost, turnover, liquidity, and capacity evidence.",
        ),
        EvidenceRequirement(
            "exposure_screen",
            "Post-selection style and industry exposure screen.",
        ),
        EvidenceRequirement(
            "negative_controls",
            "Label shuffle, sentinel, random feature, or universe controls.",
        ),
    )
    release = (
        *candidate,
        EvidenceRequirement("dsr", "Deflated Sharpe with a complete comparable trial set."),
        EvidenceRequirement(
            "pbo",
            "PBO/CSCV evidence or an explicit insufficient-trials record.",
            allowed_statuses=("pass", "insufficient_evidence"),
        ),
        EvidenceRequirement(
            "scenario_backtest",
            "Block/bootstrap or regime scenario stress evidence.",
        ),
        EvidenceRequirement(
            "candidate_freeze",
            "Frozen candidate run, targets, and artifact hashes.",
        ),
        EvidenceRequirement("paper_shadow", "Paper or shadow observation evidence."),
        EvidenceRequirement(
            "sizing_receipt",
            "Calibrated sizing and portfolio constraint receipt.",
        ),
        EvidenceRequirement(
            "strategy_risk",
            "PSR, concentration, failure probability, and cost resilience.",
        ),
        EvidenceRequirement(
            "operator_approval",
            "Explicit human approval for release handoff.",
            artifact_required=False,
        ),
    )
    if level == "exploratory":
        return ProtocolPolicy(level, base, False, 0.0, True)
    if level == "candidate":
        return ProtocolPolicy(level, candidate, True, 0.95, False)
    if level == "release":
        return ProtocolPolicy(level, release, True, 0.99, False)
    raise ValueError(f"Unsupported protocol level: {level}")


def load_protocol_manifest(path: str | Path) -> dict[str, object]:
    """Load a JSON or YAML evidence manifest."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    payload = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("protocol manifest must contain a mapping")
    return payload


def evaluate_protocol_manifest(
    manifest: Mapping[str, object],
    *,
    level: ProtocolLevel,
    base_dir: str | Path | None = None,
) -> ProtocolReport:
    """Evaluate evidence completeness, integrity, and purging semantics."""

    policy = protocol_policy(level)
    evidence_raw = manifest.get("evidence")
    evidence = dict(evidence_raw) if isinstance(evidence_raw, Mapping) else {}
    missing: list[str] = []
    failed: list[str] = []
    warnings: list[str] = []
    satisfied = 0
    evidence_root = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()

    for requirement in policy.requirements:
        item = evidence.get(requirement.name)
        if not isinstance(item, Mapping):
            if requirement.any_of and any(
                isinstance(evidence.get(name), Mapping) for name in requirement.any_of
            ):
                satisfied += 1
                continue
            missing.append(requirement.name)
            continue
        item_failed = _evaluate_evidence_item(
            requirement,
            item,
            evidence_root=evidence_root,
            failed=failed,
            warnings=warnings,
        )
        if not item_failed:
            satisfied += 1

    _evaluate_purging(manifest, policy, missing, failed, warnings)
    manifest_text = json.dumps(dict(manifest), sort_keys=True, ensure_ascii=False, default=str)
    status: Literal["pass", "fail"] = "fail" if missing or failed else "pass"
    return ProtocolReport(
        schema_version=1,
        level=level,
        status=status,
        required_count=len(policy.requirements),
        satisfied_count=satisfied,
        missing=tuple(sorted(set(missing))),
        failed=tuple(sorted(set(failed))),
        warnings=tuple(warnings),
        manifest_sha256=hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        policy={
            "require_event_window_purge": policy.require_event_window_purge,
            "minimum_event_window_coverage": policy.minimum_event_window_coverage,
            "allow_purge_fallback": policy.allow_purge_fallback,
            "verify_artifact_sha256": True,
        },
        evidence=evidence,
    )


def write_protocol_report(report: ProtocolReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def enforce_release_protocol(
    run_dir: str | Path,
    *,
    required: bool = False,
) -> ProtocolReport | None:
    """Enforce a saved release report before execution handoff."""

    root = Path(run_dir)
    report_path = root / "research_protocol_report.json"
    if not report_path.exists():
        if required:
            raise SystemExit(f"Release protocol report is required but missing: {report_path}")
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid release protocol report: {report_path}")
    if payload.get("level") != "release":
        raise SystemExit(f"Execution handoff requires a release protocol report: {report_path}")
    if payload.get("status") != "pass":
        raise SystemExit(f"Execution handoff blocked by research protocol: {report_path}")
    return ProtocolReport(
        schema_version=int(payload.get("schema_version", 1)),
        level="release",
        status="pass",
        required_count=int(payload.get("required_count", 0)),
        satisfied_count=int(payload.get("satisfied_count", 0)),
        missing=tuple(payload.get("missing") or ()),
        failed=tuple(payload.get("failed") or ()),
        warnings=tuple(payload.get("warnings") or ()),
        manifest_sha256=str(payload.get("manifest_sha256") or ""),
        policy=dict(payload.get("policy") or {}),
        evidence=dict(payload.get("evidence") or {}),
    )


def example_manifest(level: ProtocolLevel) -> dict[str, object]:
    """Return a reviewable manifest template for a protocol level."""

    policy = protocol_policy(level)
    evidence: dict[str, object] = {}
    for requirement in policy.requirements:
        item: dict[str, object] = {
            "status": "missing",
            "path": None,
            "sha256": None,
            "notes": requirement.description,
        }
        if requirement.name == "operator_approval":
            item.update({"approved_by": None, "approved_at": None})
        evidence[requirement.name] = item
    return {
        "schema_version": 1,
        "level": level,
        "run_id": None,
        "purging": {
            "mode": "event_window" if policy.require_event_window_purge else "gap",
            "event_window_coverage": 1.0 if policy.require_event_window_purge else None,
            "fallback_used": False,
        },
        "evidence": evidence,
    }


def _evaluate_evidence_item(
    requirement: EvidenceRequirement,
    item: Mapping[str, object],
    *,
    evidence_root: Path,
    failed: list[str],
    warnings: list[str],
) -> bool:
    status = str(item.get("status") or "").strip().lower()
    if status not in requirement.allowed_statuses:
        failed.append(requirement.name)
        return True

    if status == "insufficient_evidence":
        if not str(item.get("notes") or item.get("reason") or "").strip():
            failed.append(f"{requirement.name}.reason")
            return True
        return False

    if requirement.name == "operator_approval":
        approved_by = str(item.get("approved_by") or "").strip()
        approved_at = str(item.get("approved_at") or "").strip()
        if not approved_by or not approved_at:
            failed.append("operator_approval.identity")
            return True
        return False

    path_raw = item.get("path")
    if requirement.artifact_required and not path_raw:
        failed.append(f"{requirement.name}.path")
        return True
    if not path_raw:
        return False

    artifact_path = _resolve_evidence_path(path_raw, evidence_root)
    if not artifact_path.is_file():
        failed.append(f"{requirement.name}.path")
        return True
    expected = str(item.get("sha256") or "").strip().lower()
    if requirement.artifact_required and not expected:
        failed.append(f"{requirement.name}.sha256")
        return True
    if expected:
        actual = _sha256_file(artifact_path)
        if actual != expected:
            failed.append(f"{requirement.name}.sha256")
            return True
    else:
        warnings.append(f"{requirement.name}: artifact exists but has no SHA-256")
    return False


def _resolve_evidence_path(value: object, evidence_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (evidence_root / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_purging(
    manifest: Mapping[str, object],
    policy: ProtocolPolicy,
    missing: list[str],
    failed: list[str],
    warnings: list[str],
) -> None:
    purge_raw = manifest.get("purging")
    purge = purge_raw if isinstance(purge_raw, Mapping) else {}
    if not policy.require_event_window_purge:
        return
    if not purge:
        missing.append("purging")
        return
    mode = str(purge.get("mode") or "").strip().lower()
    try:
        coverage = float(cast("SupportsFloat", purge.get("event_window_coverage")))
    except (TypeError, ValueError):
        coverage = float("nan")
    fallback = bool(purge.get("fallback_used"))
    if mode != "event_window":
        failed.append("purging.mode")
    if not (coverage >= policy.minimum_event_window_coverage):
        failed.append("purging.event_window_coverage")
    if fallback and not policy.allow_purge_fallback:
        failed.append("purging.fallback")
    if coverage < 1.0 and coverage >= policy.minimum_event_window_coverage:
        warnings.append(f"event-window purge coverage is {coverage:.2%}; review uncovered events")


__all__ = [
    "EvidenceRequirement",
    "ProtocolPolicy",
    "ProtocolReport",
    "enforce_release_protocol",
    "evaluate_protocol_manifest",
    "example_manifest",
    "load_protocol_manifest",
    "protocol_policy",
    "write_protocol_report",
]
