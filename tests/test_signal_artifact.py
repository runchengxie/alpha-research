from __future__ import annotations

import pandas as pd
import pytest
from research_contracts import (
    ArtifactEnvelopeV2,
    canonical_json_sha256,
    file_sha256,
    read_artifact_envelope,
)

from alpha_research.signal_artifact import (
    CANONICAL_SIGNAL_COLUMNS,
    SIGNAL_CONTRACT,
    SIGNAL_CONTRACT_NAME,
    assert_signal_artifact_frame,
    build_signal_artifact_frame,
    load_signal_metadata,
    read_signal_artifact,
    validate_signal_artifact_frame,
    write_signal_artifact,
)


def _scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-01-05", "2026-01-05"],
            "symbol": ["600519.SH", "000858.SZ"],
            "pred": [0.2, 0.1],
            "signal_eval": [0.2, 0.1],
            "signal_backtest": [0.2, 0.1],
            "future_return": [0.03, -0.01],
        }
    )


def test_signal_artifact_normalizes_and_validates_frame() -> None:
    signals = build_signal_artifact_frame(
        _scored_frame(),
        model_version="ridge:demo",
        feature_set_id="features:demo",
        signal_direction=1.0,
        eligible_for_backtest=True,
        eligible_for_live=False,
    )

    assert SIGNAL_CONTRACT_NAME == "alpha_research.signals"
    assert SIGNAL_CONTRACT.name == SIGNAL_CONTRACT_NAME
    assert SIGNAL_CONTRACT.required_columns == CANONICAL_SIGNAL_COLUMNS
    assert validate_signal_artifact_frame(signals) == []
    assert signals["rank"].tolist() == [1, 2]
    assert signals["eligible_for_backtest"].dtype.name == "boolean"


def test_signal_artifact_round_trip_writes_metadata(tmp_path) -> None:
    path = tmp_path / "signals.parquet"

    _, summary = write_signal_artifact(
        _scored_frame(),
        path,
        metadata={"run_id": "demo"},
        model_version="ridge:demo",
        signal_direction=1.0,
        eligible_for_backtest=True,
        eligible_for_live=False,
    )

    loaded = read_signal_artifact(path)
    metadata = load_signal_metadata(path)

    assert loaded["signal_date"].tolist() == ["20260105", "20260105"]
    assert summary["contract"] == SIGNAL_CONTRACT_NAME
    assert metadata["artifact_type"] == SIGNAL_CONTRACT_NAME
    assert metadata["summary"]["required_columns"] == list(CANONICAL_SIGNAL_COLUMNS)
    assert metadata["metadata"]["run_id"] == "demo"


def test_signal_artifact_reports_invalid_frame() -> None:
    invalid = pd.DataFrame({"signal_date": ["2026-01-05"], "symbol": ["600519.SH"]})

    issues = validate_signal_artifact_frame(invalid)

    assert any("missing columns" in issue for issue in issues)
    with pytest.raises(ValueError, match="Invalid signal artifact frame"):
        assert_signal_artifact_frame(invalid)


def test_signal_artifact_meta_carries_readable_v2_envelope(tmp_path) -> None:
    path = tmp_path / "signals.parquet"

    _, _ = write_signal_artifact(
        _scored_frame(),
        path,
        run_id="run-demo",
        model_version="ridge:demo",
        feature_set_id="features:demo",
        signal_direction=1.0,
        eligible_for_backtest=True,
        eligible_for_live=False,
        lineage=[("research_features.parquet", "c" * 64)],
    )

    payload = load_signal_metadata(path)
    envelope = read_artifact_envelope(payload, allow_legacy=False)

    assert isinstance(envelope, ArtifactEnvelopeV2)
    assert envelope.run_id == "run-demo"
    assert envelope.artifact_id == "signals:run-demo"
    assert envelope.artifact_type == "signals.parquet"
    assert envelope.created_at.utcoffset() is not None
    assert envelope.producer.repository == "alpha-research"
    assert envelope.producer.backend == "signal_artifact"
    assert envelope.content_sha256 == file_sha256(path)
    assert len(envelope.lineage) == 1
    assert envelope.lineage[0].artifact_id == "research_features.parquet"
    assert envelope.lineage[0].sha256 == "c" * 64
    assert envelope.configuration_sha256 == canonical_json_sha256(
        {
            "model_version": "ridge:demo",
            "feature_set_id": "features:demo",
            "signal_direction": 1.0,
            "eligible_for_backtest": True,
            "eligible_for_live": False,
        }
    )
    assert payload["artifact_type"] == SIGNAL_CONTRACT_NAME
    assert payload["schema_version"] == 1
