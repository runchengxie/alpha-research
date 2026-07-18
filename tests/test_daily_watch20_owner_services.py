from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_research.daily_watch20_model_lifecycle import (
    DEFAULT_SELECTION_SCHEMA,
    prepare_ranker_lifecycle,
)
from alpha_research.daily_watch20_statistics import holm_adjust, newey_west_mean_inference


class FakeRanker:
    def __init__(self) -> None:
        self.restored: tuple[Path, dict[str, object], dict[str, object]] | None = None

    def restore_from_path(
        self,
        model_path: Path,
        training: dict[str, object],
        *,
        metadata: dict[str, object],
    ) -> None:
        self.restored = (model_path, training, metadata)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_statistics_are_deterministic() -> None:
    result = newey_west_mean_inference([1.0, 2.0, 3.0, 4.0], minimum_lag=1)
    assert result["observations"] == 4
    assert np.isfinite(result["standard_error"])
    adjusted = holm_adjust([0.03, 0.01, np.nan])
    assert adjusted[:2].tolist() == [0.03, 0.02]
    assert np.isnan(adjusted[2])


def test_lifecycle_restores_integrity_verified_prior_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "prior"
    run_dir.mkdir(parents=True)
    model_path = run_dir / "model.ubj"
    metadata_path = run_dir / "model_metadata.json"
    model_path.write_bytes(b"model")
    metadata = {
        "training": {"as_of_date": "2026-07-14"},
        "persistence": {"model_version": "m1"},
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    receipt = {
        "schema_version": DEFAULT_SELECTION_SCHEMA,
        "status": "passed",
        "quality_status": "passed",
        "source_date": "20260715",
        "generated_at": "2026-07-15T09:00:00+08:00",
        "artifacts": {
            "model.ubj": {"path": "model.ubj", "sha256": _sha(model_path)},
            "model_metadata.json": {
                "path": "model_metadata.json",
                "sha256": _sha(metadata_path),
            },
        },
    }
    (run_dir / "selection_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    ranker = FakeRanker()
    lifecycle = prepare_ranker_lifecycle(
        ranker,
        force_retrain=False,
        retrain_weekdays=(0,),
        max_age_trade_days=5,
        output_root=tmp_path,
        source_date="20260716",
        open_dates=pd.date_range("2026-07-13", "2026-07-17", freq="B"),
    )
    assert lifecycle.mode == "reused"
    assert lifecycle.origin_source_date == "20260715"
    assert ranker.restored is not None
