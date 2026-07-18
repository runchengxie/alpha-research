"""Consume the point-in-time sparse news-heat contract for DailyWatch20."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NEWS_HEAT_SCHEMA = "daily_watch20.news_heat.v1"
NEWS_HEAT_COLUMNS = {
    "source_date",
    "data_as_of",
    "symbol",
    "news_heat_score",
    "source_kind",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DailyWatch20NewsHeat:
    enabled: bool
    source_date: str
    frame: pd.DataFrame
    root: Path
    reason: str | None
    receipt: dict[str, Any] | None

    def receipt_summary(self) -> dict[str, Any]:
        payload = {
            "enabled": self.enabled,
            "source_date": self.source_date,
            "root": str(self.root),
            "reason": self.reason,
            "rows": len(self.frame),
            "coverage_mode": "sparse_positive_only",
            "missing_symbol_semantics": "neutral_unknown",
        }
        if self.receipt is not None:
            receipt_path = self.root / "news_heat_receipt.json"
            data_path = self.root / "news_heat.csv"
            payload["schema_version"] = self.receipt.get("schema_version")
            payload["generated_at"] = self.receipt.get("generated_at")
            payload["receipt_path"] = str(receipt_path)
            payload["receipt_sha256"] = _sha256_file(receipt_path)
            payload["data_path"] = str(data_path)
            payload["data_sha256"] = _sha256_file(data_path)
        return payload


def _date_key(value: object, *, field: str) -> str:
    text = str(value or "").strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"{field} must be YYYYMMDD")
    try:
        pd.Timestamp(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid date") from exc
    return text


def _disabled(root: Path, source_date: str, reason: str) -> DailyWatch20NewsHeat:
    return DailyWatch20NewsHeat(
        enabled=False,
        source_date=source_date,
        frame=pd.DataFrame(columns=pd.Index(sorted(NEWS_HEAT_COLUMNS))),
        root=root,
        reason=reason,
        receipt=None,
    )


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid news heat receipt: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("news heat receipt must be a JSON object")
    return payload


def _validated_frame(data_path: Path, receipt: dict[str, Any], expected: str) -> pd.DataFrame:
    data_meta = receipt.get("data_file")
    if not isinstance(data_meta, dict) or not data_path.is_file():
        raise RuntimeError("passed news heat artifact has no data file")
    expected_hash = str(data_meta.get("sha256") or "")
    if not expected_hash or _sha256_file(data_path) != expected_hash:
        raise RuntimeError("news heat data hash mismatch")
    frame = pd.read_csv(
        data_path,
        dtype={"source_date": str, "data_as_of": str, "symbol": str},
    )
    missing = sorted(NEWS_HEAT_COLUMNS - set(frame.columns))
    if missing:
        raise RuntimeError(f"news heat data is missing columns: {missing}")
    if frame["symbol"].duplicated().any():
        raise RuntimeError("news heat contains duplicate symbols")
    if not frame["symbol"].str.fullmatch(r"\d{6}\.(SH|SZ)").all():
        raise RuntimeError("news heat contains out-of-scope symbols")
    if set(frame["source_date"].str.replace("-", "")) != {expected}:
        raise RuntimeError("news heat row source_date mismatch")
    if set(frame["data_as_of"].str.replace("-", "")) != {expected}:
        raise RuntimeError("news heat row data_as_of mismatch")
    scores = pd.to_numeric(frame["news_heat_score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise RuntimeError("news heat scores must be finite")
    if bool(((scores < 0) | (scores > 1)).any()):
        raise RuntimeError("news heat scores must be within [0, 1]")
    out = frame.loc[:, sorted(NEWS_HEAT_COLUMNS)].copy()
    out["news_heat_score"] = scores.astype(float)
    return out


def load_daily_watch20_news_heat(
    root: str | Path,
    *,
    source_date: str,
    min_rows: int = 20,
) -> DailyWatch20NewsHeat:
    """Load a passed exact-date artifact; unavailable input is an explicit no-op."""

    expected = _date_key(source_date, field="source_date")
    resolved = Path(root).expanduser().resolve()
    receipt_path = resolved / "news_heat_receipt.json"
    data_path = resolved / "news_heat.csv"
    if not receipt_path.is_file():
        return _disabled(resolved, expected, "news heat receipt is unavailable")
    receipt = _read_receipt(receipt_path)
    actual = str(receipt.get("source_date") or "").replace("-", "")
    if actual != expected:
        return _disabled(
            resolved,
            expected,
            f"news heat source_date {actual or 'missing'} does not match {expected}",
        )
    if receipt.get("schema_version") != NEWS_HEAT_SCHEMA:
        raise RuntimeError("news heat receipt schema mismatch")
    if str(receipt.get("status") or "").lower() != "passed":
        return _disabled(
            resolved,
            expected,
            str(receipt.get("reason") or "news heat unavailable"),
        )
    if str(receipt.get("quality_status") or "passed").lower() != "passed":
        raise RuntimeError("news heat receipt quality status is not passed")
    if receipt.get("coverage_mode") != "sparse_positive_only":
        raise RuntimeError("news heat coverage mode must be sparse_positive_only")
    frame = _validated_frame(data_path, receipt, expected)
    if len(frame) < max(1, int(min_rows)):
        return _disabled(resolved, expected, f"news heat has only {len(frame)} rows")
    return DailyWatch20NewsHeat(
        enabled=True,
        source_date=expected,
        frame=frame,
        root=resolved,
        reason=None,
        receipt=receipt,
    )


def join_news_heat_neutral(
    candidates: pd.DataFrame,
    news_heat: DailyWatch20NewsHeat,
) -> pd.DataFrame:
    """Join sparse heat while treating absent symbols as unknown/neutral, never zero."""

    out = candidates.copy()
    if not news_heat.enabled:
        out["news_heat_score"] = np.nan
        out["news_heat_guard"] = np.nan
        out["news_heat_available"] = False
        return out
    heat = news_heat.frame[["symbol", "news_heat_score"]]
    out = out.merge(heat, on="symbol", how="left", validate="many_to_one")
    known = out["news_heat_score"].notna()
    neutral = float(news_heat.frame["news_heat_score"].median())
    out["news_heat_guard"] = out["news_heat_score"].fillna(neutral)
    out["news_heat_available"] = known
    return out


__all__ = [
    "DailyWatch20NewsHeat",
    "NEWS_HEAT_COLUMNS",
    "NEWS_HEAT_SCHEMA",
    "join_news_heat_neutral",
    "load_daily_watch20_news_heat",
]
