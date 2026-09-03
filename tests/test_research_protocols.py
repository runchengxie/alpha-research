from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from alpha_research.research_protocols import (
    ProtocolLevel,
    enforce_release_protocol,
    evaluate_protocol_manifest,
    example_manifest,
    write_protocol_report,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pass_manifest(level: str, root: Path) -> dict[str, Any]:
    manifest = example_manifest(cast(ProtocolLevel, level))
    evidence = cast(dict[str, Any], manifest["evidence"])
    for name, item in evidence.items():
        assert isinstance(item, dict)
        item["status"] = "pass"
        if name == "operator_approval":
            item["approved_by"] = "test-operator"
            item["approved_at"] = "2026-07-14T12:00:00-07:00"
            continue
        artifact = root / f"{name}.json"
        artifact.write_text(json.dumps({"evidence": name}), encoding="utf-8")
        item["path"] = str(artifact)
        item["sha256"] = _sha256(artifact)
    return manifest


def test_candidate_protocol_rejects_gap_or_fallback_purging(tmp_path: Path) -> None:
    manifest = _pass_manifest("candidate", tmp_path)
    purging = cast(dict[str, Any], manifest["purging"])
    purging.update({"mode": "gap", "event_window_coverage": 0.75, "fallback_used": True})

    report = evaluate_protocol_manifest(manifest, level="candidate")

    assert report.status == "fail"
    assert "purging.mode" in report.failed
    assert "purging.event_window_coverage" in report.failed
    assert "purging.fallback" in report.failed


def test_release_protocol_accepts_explicit_pbo_insufficient_evidence(
    tmp_path: Path,
) -> None:
    manifest = _pass_manifest("release", tmp_path)
    pbo = cast(dict[str, Any], cast(dict[str, Any], manifest["evidence"])["pbo"])
    pbo.update(
        {
            "status": "insufficient_evidence",
            "path": None,
            "sha256": None,
            "notes": "Only one comparable candidate is available.",
        }
    )

    report = evaluate_protocol_manifest(manifest, level="release")

    assert report.status == "pass"
    assert report.satisfied_count == report.required_count


def test_protocol_rejects_tampered_evidence(tmp_path: Path) -> None:
    manifest = _pass_manifest("candidate", tmp_path)
    feature = cast(dict[str, Any], cast(dict[str, Any], manifest["evidence"])["feature_evidence"])
    Path(str(feature["path"])).write_text("tampered", encoding="utf-8")

    report = evaluate_protocol_manifest(manifest, level="candidate")

    assert report.status == "fail"
    assert "feature_evidence.sha256" in report.failed


def test_saved_release_report_can_gate_execution_handoff(tmp_path: Path) -> None:
    manifest = _pass_manifest("release", tmp_path)
    report = evaluate_protocol_manifest(manifest, level="release")
    write_protocol_report(report, tmp_path / "research_protocol_report.json")

    loaded = enforce_release_protocol(tmp_path, required=True)
    assert loaded is not None
    assert loaded.status == "pass"

    payload = json.loads((tmp_path / "research_protocol_report.json").read_text())
    payload["status"] = "fail"
    (tmp_path / "research_protocol_report.json").write_text(json.dumps(payload))

    try:
        enforce_release_protocol(tmp_path, required=True)
    except SystemExit as exc:
        assert "blocked" in str(exc)
    else:
        raise AssertionError("failed release protocol must block handoff")
