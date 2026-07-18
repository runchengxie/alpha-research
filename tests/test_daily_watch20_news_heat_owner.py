from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from alpha_research.daily_watch20_news_heat import (
    NEWS_HEAT_SCHEMA,
    join_news_heat_neutral,
    load_daily_watch20_news_heat,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_news_heat_validates_hash_and_uses_neutral_missing_semantics(tmp_path: Path) -> None:
    data = tmp_path / "news_heat.csv"
    pd.DataFrame(
        {
            "source_date": ["20260717"],
            "data_as_of": ["20260717"],
            "symbol": ["000001.SZ"],
            "news_heat_score": [0.8],
            "source_kind": ["news"],
        }
    ).to_csv(data, index=False)
    receipt = {
        "schema_version": NEWS_HEAT_SCHEMA,
        "status": "passed",
        "quality_status": "passed",
        "coverage_mode": "sparse_positive_only",
        "source_date": "20260717",
        "data_file": {"sha256": _sha(data)},
    }
    (tmp_path / "news_heat_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    heat = load_daily_watch20_news_heat(tmp_path, source_date="20260717", min_rows=1)
    joined = join_news_heat_neutral(
        pd.DataFrame({"symbol": ["000001.SZ", "600000.SH"]}),
        heat,
    )
    assert joined["news_heat_available"].tolist() == [True, False]
    assert joined["news_heat_guard"].tolist() == [0.8, 0.8]
