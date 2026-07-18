from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from alpha_research.daily_watch20_minute_features import (
    MINUTE_FEATURE_COLUMNS,
    MINUTE_TRANSFORM_CONTRACT,
    transform_daily_watch20_minute_catalog,
)
from alpha_research.daily_watch20_policy import DailyWatch20AlphaPolicy
from market_data_platform.research_views.daily_watch20_minute_source import (
    MINUTE_SOURCE_CONTRACT,
    MinutePartitionState,
    MinuteSourceCatalog,
)


def _catalog(tmp_path: Path) -> MinuteSourceCatalog:
    root = tmp_path / "minute"
    directory = root / "trade_date=20260717"
    directory.mkdir(parents=True)
    path = directory / "part-000.parquet"
    pd.DataFrame(
        {
            "trade_date": ["20260717"] * 4,
            "ts_code": ["000001.SZ"] * 4,
            "trade_time": ["09:35:00", "10:00:00", "14:30:00", "15:00:00"],
            "open": [10.0, 10.1, 10.2, 10.3],
            "close": [10.1, 10.0, 10.3, 10.4],
            "high": [10.2, 10.2, 10.4, 10.5],
            "low": [9.9, 9.95, 10.1, 10.2],
            "vol": [100.0, 200.0, 300.0, 400.0],
        }
    ).to_parquet(path, index=False)
    metadata = (
        {
            "root": str(root),
            "source_kind": "canonical",
            "relative_path": "trade_date=20260717/part-000.parquet",
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        },
    )
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    state = MinutePartitionState(
        trade_date="20260717",
        fingerprint=fingerprint,
        files=(path,),
        file_metadata=metadata,
    )
    return MinuteSourceCatalog(
        source_contract=MINUTE_SOURCE_CONTRACT,
        start_date="20260717",
        end_date="20260717",
        canonical_root=root,
        overlay_roots=(),
        minute_version="v1",
        coverage_path=None,
        coverage_sha256=None,
        partitions={"20260717": state},
    )


def test_transform_catalog_returns_bound_evidence(tmp_path: Path) -> None:
    result = transform_daily_watch20_minute_catalog(
        _catalog(tmp_path), threads=1, memory_limit="1GB"
    )
    assert tuple(result.frame.columns) == MINUTE_FEATURE_COLUMNS
    assert result.frame[["trade_date", "symbol"]].to_dict("records") == [
        {"trade_date": "20260717", "symbol": "000001.SZ"}
    ]
    assert result.evidence["transform_contract"] == MINUTE_TRANSFORM_CONTRACT
    assert result.evidence["source_partitions"]["20260717"]["feature_rows"] == 1
    assert len(result.evidence["frame_sha256"]) == 64


def test_alpha_policy_identity_changes_with_model_semantics() -> None:
    baseline = DailyWatch20AlphaPolicy()
    changed = DailyWatch20AlphaPolicy(max_model_age_trade_days=20)
    assert baseline.policy_id != changed.policy_id
    assert baseline.to_dict()["label_policy_id"] == baseline.label_policy_id
