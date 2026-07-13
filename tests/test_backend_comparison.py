import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cstree.alpha.backend_comparison import (
    BackendPromotionThresholds,
    compare_backend_evaluations,
    replay_backend_comparison,
    write_backend_comparison_replay_receipt,
    write_backend_evaluation,
)
from cstree.alpha.backends import FittedModelHandle
from cstree.alpha.promotion_gate import build_promotion_record, load_promotion_gate_config
from cstree.alpha.research_artifacts import ArtifactIntegrityError


def _predictions(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03"]),
            "symbol": ["A", "B", "A"],
            "pred": values,
            "target": [0.1, -0.2, 0.3],
        }
    )


def _evaluation(tmp_path: Path, name: str, backend_id: str, values: list[float]):
    return write_backend_evaluation(
        tmp_path / name,
        backend_id=backend_id,
        run_id=name,
        predictions=_predictions(values),
        metrics={"rank_ic": 0.25, "sharpe": 1.1},
        model_handle=FittedModelHandle(
            backend_id=backend_id,
            model_id=f"{name}-model",
            model_type="ridge",
            metadata={"features": ["f1"]},
            runtime_ref="in-process-only",
        ),
        target_col="target",
    )


def _write_run(path: Path) -> None:
    path.mkdir(parents=True)
    summary = {
        "eval": {
            "ic": {"mean": 0.02, "ir": 0.4},
            "long_short": 0.01,
            "constant_prediction": False,
            "zero_feature_importance": False,
        },
        "backtest": {
            "stats": {
                "sharpe": 1.0,
                "max_drawdown": -0.1,
                "avg_turnover": 0.2,
                "avg_cost_drag": 0.002,
            }
        },
        "walk_forward": {
            "enabled": True,
            "results": [{"status": "ok", "test_ic": {"mean": 0.02}}],
        },
        "final_oos": {
            "enabled": True,
            "ic": {"mean": 0.02},
            "long_short": 0.01,
            "backtest": {"stats": {"sharpe": 0.8}},
        },
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "config.used.yml").write_text(
        yaml.safe_dump({"label": {"horizon_days": 20}}), encoding="utf-8"
    )


def test_backend_comparison_is_machine_readable_and_replayable(tmp_path: Path) -> None:
    native = _evaluation(tmp_path, "native", "native", [0.2, -0.1, 0.4])
    qlib = _evaluation(tmp_path, "qlib", "qlib", [0.2, -0.1, 0.4])
    report = compare_backend_evaluations(
        native,
        qlib,
        thresholds=BackendPromotionThresholds(
            metric_abs_tolerances={"rank_ic": 0.0, "sharpe": 0.0}
        ),
        output_path=tmp_path / "comparison.json",
    )

    payload = replay_backend_comparison(report)

    assert payload["schema"] == "backend_comparison.v1"
    assert payload["decision"] == {"status": "promotable", "failures": []}
    assert payload["comparison"]["overlap_ratio"] == pytest.approx(1.0)
    assert payload["comparison"]["prediction_pearson"] == pytest.approx(1.0)
    assert payload["replay_verified"] is True
    serialized = json.loads((tmp_path / "native" / "evaluation.json").read_text())
    assert "runtime_ref" not in json.dumps(serialized)
    assert "in-process-only" not in json.dumps(serialized)


def test_backend_comparison_detects_source_tampering(tmp_path: Path) -> None:
    native = _evaluation(tmp_path, "native", "native", [0.2, -0.1, 0.4])
    qlib = _evaluation(tmp_path, "qlib", "qlib", [0.2, -0.1, 0.4])
    report = compare_backend_evaluations(
        native,
        qlib,
        thresholds=BackendPromotionThresholds(),
        output_path=tmp_path / "comparison.json",
    )
    prediction_path = tmp_path / "qlib" / "predictions.csv"
    prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8").replace("0.4", "0.9"),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="digest mismatch"):
        replay_backend_comparison(report)


def test_backend_comparison_replay_receipt_is_strict_and_deterministic(tmp_path: Path) -> None:
    native = _evaluation(tmp_path, "native", "native", [0.2, -0.1, 0.4])
    qlib = _evaluation(tmp_path, "qlib", "qlib", [0.2, -0.1, 0.4])
    report = compare_backend_evaluations(
        native,
        qlib,
        thresholds=BackendPromotionThresholds(),
        output_path=tmp_path / "comparison.json",
    )

    first = write_backend_comparison_replay_receipt(
        report,
        output_path=tmp_path / "replay-receipt.json",
    )
    second = write_backend_comparison_replay_receipt(
        report.path,
        output_path=tmp_path / "replay-receipt-copy.json",
    )

    payload = json.loads(first.verify().read_text(encoding="utf-8"))
    assert payload["schema"] == "backend_comparison_replay_receipt.v1"
    assert payload["source_report_sha256"] == report.sha256
    assert payload["source_report"]["sha256"] == report.sha256
    assert payload["verification_method"] == "artifact-digest-and-decision-replay"
    assert payload["replay_verified"] is True
    assert payload["decision"] == {"status": "promotable", "failures": []}
    assert payload["comparison"]["native_backend_id"] == "native"
    assert payload["comparison"]["candidate_backend_id"] == "qlib"
    assert payload["comparison"]["prediction_pearson"] == pytest.approx(1.0)
    assert payload["comparison"]["prediction_mae"] == pytest.approx(0.0)
    assert payload["thresholds"] == BackendPromotionThresholds().to_metadata()
    assert second.sha256 == first.sha256


def test_backend_comparison_rejects_non_native_qlib_pair(tmp_path: Path) -> None:
    first = _evaluation(tmp_path, "first", "custom-a", [0.2, -0.1, 0.4])
    second = _evaluation(tmp_path, "second", "custom-b", [0.2, -0.1, 0.4])
    report = compare_backend_evaluations(
        first,
        second,
        thresholds=BackendPromotionThresholds(),
        output_path=tmp_path / "comparison.json",
    )

    payload = replay_backend_comparison(report)

    assert payload["decision"]["status"] == "non-comparable"
    assert payload["decision"]["failures"] == [
        "expected_native_baseline",
        "expected_qlib_candidate",
    ]


def test_promotion_gate_enforces_replayed_backend_comparison(tmp_path: Path) -> None:
    baseline_run = tmp_path / "baseline-run"
    candidate_run = tmp_path / "candidate-run"
    _write_run(baseline_run)
    _write_run(candidate_run)
    native = _evaluation(tmp_path, "native", "native", [0.2, -0.1, 0.4])
    qlib = _evaluation(tmp_path, "qlib", "qlib", [-0.2, 0.1, -0.4])
    report = compare_backend_evaluations(
        native,
        qlib,
        thresholds=BackendPromotionThresholds(),
        output_path=tmp_path / "comparison.json",
    )
    config = load_promotion_gate_config(
        {
            "baseline_run": str(baseline_run),
            "candidate_run": str(candidate_run),
            "comparability_keys": ["label.horizon_days"],
            "required_evidence": ["main_eval", "backtest", "backend_comparison"],
            "backend_comparison": {"candidate_report": report.path},
        }
    )

    record = build_promotion_record(config)

    assert record["promotion_status"] == "rejected"
    assert record["hard_failures"] == ["backend_comparison_rejected"]
    assert record["candidate_evidence"]["backend_comparison"]["replay_verified"] is True


def test_model_handle_rejects_opaque_metadata() -> None:
    with pytest.raises(TypeError, match="unsupported"):
        FittedModelHandle(
            backend_id="qlib",
            model_id="model",
            model_type="ridge",
            metadata={"model": object()},
        )


def test_model_handle_metadata_serialization_is_detached() -> None:
    handle = FittedModelHandle(
        backend_id="native",
        model_id="model",
        model_type="ridge",
        metadata={"features": ["f1"]},
    )
    first = handle.to_metadata()
    assert isinstance(first["metadata"], dict)
    features = first["metadata"]["features"]
    assert isinstance(features, list)
    features.append("mutated")

    assert handle.to_metadata()["metadata"] == {"features": ["f1"]}
