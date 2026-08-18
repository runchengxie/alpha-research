from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype
from research_contracts import (
    ArtifactEnvelopeV2,
    LineageInput,
    ProducerIdentity,
    attach_artifact_envelope_v2,
    canonical_json_sha256,
    file_sha256,
)

SIGNAL_CONTRACT_NAME = "alpha_research.signals"
SIGNAL_SCHEMA_VERSION = 1
CANONICAL_SIGNAL_FILE = "signals.parquet"
CANONICAL_SIGNAL_META_FILE = "signals.meta.json"
CANONICAL_SIGNAL_COLUMNS = (
    "signal_date",
    "symbol",
    "raw_pred",
    "signal_eval",
    "signal_backtest",
    "signal_direction",
    "rank",
    "model_version",
    "feature_set_id",
    "eligible_for_backtest",
    "eligible_for_live",
)
SCORE_COLUMNS = ("raw_pred", "signal_eval", "signal_backtest")
BOOL_COLUMNS = ("eligible_for_backtest", "eligible_for_live")


@dataclass(frozen=True)
class SignalArtifactContract:
    """Stable alpha-to-backtest scored signal artifact contract."""

    name: str = SIGNAL_CONTRACT_NAME
    schema_version: int = SIGNAL_SCHEMA_VERSION
    file_name: str = CANONICAL_SIGNAL_FILE
    metadata_file_name: str = CANONICAL_SIGNAL_META_FILE
    required_columns: tuple[str, ...] = tuple(CANONICAL_SIGNAL_COLUMNS)
    score_columns: tuple[str, ...] = SCORE_COLUMNS
    rank_column: str = "rank"
    date_column: str = "signal_date"
    symbol_column: str = "symbol"


SIGNAL_CONTRACT = SignalArtifactContract()

PRODUCER_REPOSITORY = "alpha-research"
PRODUCER_BACKEND_SIGNAL_ARTIFACT = "signal_artifact"
PRODUCER_BACKEND_STYLE_REPLICA = "style_replica"


def _producer_version() -> str:
    try:
        return package_version(PRODUCER_REPOSITORY)
    except PackageNotFoundError:
        return "0.0.0"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def default_signal_run_id() -> str:
    """Return a timezone-stamped run id when the caller does not provide one."""
    return f"signals-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def build_signal_envelope_v2(
    *,
    artifact_id: str,
    artifact_type: str,
    run_id: str,
    content_sha256: str,
    configuration: Mapping[str, Any],
    producer_backend: str,
    lineage: Sequence[tuple[str, str]] = (),
) -> ArtifactEnvelopeV2:
    """Build a research.artifact-envelope.v2 for a signal artifact write."""
    return ArtifactEnvelopeV2(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        run_id=run_id,
        created_at=datetime.now(UTC),
        producer=ProducerIdentity(
            repository=PRODUCER_REPOSITORY,
            version=_producer_version(),
            commit=_git_commit(),
            backend=producer_backend,
        ),
        configuration_sha256=canonical_json_sha256(configuration),
        content_sha256=content_sha256,
        lineage=tuple(LineageInput(artifact_id=item[0], sha256=item[1]) for item in lineage),
    )


@dataclass(frozen=True)
class _SignalSourceColumns:
    date_col: str
    raw_pred_col: str
    signal_eval_col: str
    signal_backtest_col: str
    direction_col: str | None
    rank_col: str | None
    model_col: str | None
    feature_col: str | None
    backtest_col: str | None
    live_col: str | None


def _first_existing(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _optional_series(frame: pd.DataFrame, column: str | None, default: Any) -> pd.Series:
    if column and column in frame.columns:
        return cast(pd.Series, frame[column])
    return pd.Series([default] * len(frame), index=frame.index)


def _column_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _rank_within_date(frame: pd.DataFrame, score_col: str) -> pd.Series:
    work = pd.DataFrame(
        {
            "signal_date": _column_series(frame, "signal_date"),
            "score": pd.to_numeric(_column_series(frame, score_col), errors="coerce"),
        }
    )
    ranked = work.groupby("signal_date", sort=False)["score"].rank(
        ascending=False,
        method="first",
        na_option="bottom",
    )
    return cast(pd.Series, ranked.astype("Int64"))


def _resolve_signal_source_columns(frame_columns: Sequence[str]) -> _SignalSourceColumns:
    date_col = _first_existing(
        frame_columns,
        ["signal_date", "trade_date", "rebalance_date", "date"],
    )
    if date_col is None:
        raise ValueError("Signal artifact requires a signal_date or trade_date column.")
    if "symbol" not in frame_columns:
        raise ValueError("Signal artifact requires a symbol column.")

    raw_pred_col = _first_existing(frame_columns, ["raw_pred", "pred", "score", "signal"])
    if raw_pred_col is None:
        raise ValueError("Signal artifact requires a prediction column such as pred.")
    signal_eval_col = (
        _first_existing(frame_columns, ["signal_eval", "signal", raw_pred_col]) or raw_pred_col
    )
    signal_backtest_col = (
        _first_existing(
            frame_columns,
            ["signal_backtest", "signal_bt", signal_eval_col, raw_pred_col],
        )
        or signal_eval_col
    )
    return _SignalSourceColumns(
        date_col=date_col,
        raw_pred_col=raw_pred_col,
        signal_eval_col=signal_eval_col,
        signal_backtest_col=signal_backtest_col,
        direction_col=_first_existing(frame_columns, ["signal_direction"]),
        rank_col=_first_existing(frame_columns, ["rank", "signal_rank"]),
        model_col=_first_existing(frame_columns, ["model_version"]),
        feature_col=_first_existing(frame_columns, ["feature_set_id"]),
        backtest_col=_first_existing(frame_columns, ["eligible_for_backtest"]),
        live_col=_first_existing(frame_columns, ["eligible_for_live"]),
    )


def validate_signal_artifact_frame(signals: pd.DataFrame) -> list[str]:
    """Return contract violations for a canonical scored signal frame."""

    issues: list[str] = []
    missing = [column for column in CANONICAL_SIGNAL_COLUMNS if column not in signals.columns]
    if missing:
        issues.append("missing columns: " + ", ".join(missing))
        return issues

    if signals.empty:
        return issues

    signal_dates = _column_series(signals, "signal_date").astype("string")
    invalid_dates = signal_dates.isna() | ~signal_dates.str.fullmatch(r"\d{8}").fillna(False)
    if bool(invalid_dates.any()):
        issues.append("signal_date must use YYYYMMDD strings")

    symbols = _column_series(signals, "symbol").astype("string")
    if bool(symbols.isna().any()) or bool(symbols.str.strip().eq("").any()):
        issues.append("symbol must be non-empty")

    for column in SCORE_COLUMNS:
        if not is_numeric_dtype(_column_series(signals, column)):
            issues.append(f"{column} must be numeric")

    if not is_numeric_dtype(_column_series(signals, "signal_direction")):
        issues.append("signal_direction must be numeric")
    if not is_integer_dtype(_column_series(signals, "rank")):
        issues.append("rank must be integer typed")
    for column in BOOL_COLUMNS:
        if not is_bool_dtype(_column_series(signals, column)):
            issues.append(f"{column} must be boolean typed")
    return issues


def assert_signal_artifact_frame(signals: pd.DataFrame) -> None:
    issues = validate_signal_artifact_frame(signals)
    if issues:
        raise ValueError("Invalid signal artifact frame: " + "; ".join(issues))


def build_signal_artifact_frame(
    scored: pd.DataFrame,
    *,
    model_version: str | None = None,
    feature_set_id: str | None = None,
    signal_direction: float | int | None = None,
    eligible_for_backtest: bool | None = None,
    eligible_for_live: bool | None = None,
) -> pd.DataFrame:
    """Normalize scored data into the canonical signal artifact schema."""

    if scored is None or scored.empty:
        return pd.DataFrame(columns=pd.Index(CANONICAL_SIGNAL_COLUMNS))
    frame = scored.copy()
    frame_columns = [str(column) for column in frame.columns]
    source_cols = _resolve_signal_source_columns(frame_columns)

    out = pd.DataFrame(index=frame.index)
    dates = pd.to_datetime(_column_series(frame, source_cols.date_col), errors="coerce")
    out["signal_date"] = dates.dt.strftime("%Y%m%d")
    out["symbol"] = _column_series(frame, "symbol").astype(str)
    out["raw_pred"] = pd.to_numeric(
        _column_series(frame, source_cols.raw_pred_col),
        errors="coerce",
    )
    out["signal_eval"] = pd.to_numeric(
        _column_series(frame, source_cols.signal_eval_col),
        errors="coerce",
    )
    out["signal_backtest"] = pd.to_numeric(
        _column_series(frame, source_cols.signal_backtest_col),
        errors="coerce",
    )
    out["signal_direction"] = pd.to_numeric(
        _optional_series(frame, source_cols.direction_col, signal_direction),
        errors="coerce",
    )
    if source_cols.rank_col is not None:
        rank = pd.to_numeric(_column_series(frame, source_cols.rank_col), errors="coerce")
        out["rank"] = cast(pd.Series, rank).astype("Int64")
    else:
        out["rank"] = _rank_within_date(out, "signal_backtest")
    out["model_version"] = _optional_series(frame, source_cols.model_col, model_version).astype(
        "string"
    )
    out["feature_set_id"] = _optional_series(frame, source_cols.feature_col, feature_set_id).astype(
        "string"
    )
    out["eligible_for_backtest"] = _optional_series(
        frame, source_cols.backtest_col, eligible_for_backtest
    ).astype("boolean")
    out["eligible_for_live"] = _optional_series(
        frame, source_cols.live_col, eligible_for_live
    ).astype("boolean")
    for column in frame.columns:
        if column not in out.columns and column not in {
            source_cols.date_col,
            source_cols.raw_pred_col,
        }:
            out[column] = frame[column]
    signals = out.reset_index(drop=True)
    assert_signal_artifact_frame(signals)
    return signals


def signal_artifact_summary(
    signals: pd.DataFrame,
    *,
    path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    score_columns = [column for column in SCORE_COLUMNS if column in signals.columns]
    return {
        "contract": SIGNAL_CONTRACT_NAME,
        "schema_version": SIGNAL_SCHEMA_VERSION,
        "file": str(path) if path else None,
        "metadata_file": str(metadata_path) if metadata_path else None,
        "rows": len(signals),
        "required_columns": list(CANONICAL_SIGNAL_COLUMNS),
        "score_columns": score_columns,
        "rank_col": "rank" if "rank" in signals.columns else None,
        "date_col": "signal_date" if "signal_date" in signals.columns else None,
        "symbol_col": "symbol" if "symbol" in signals.columns else None,
    }


def write_signal_artifact(
    scored: pd.DataFrame,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    model_version: str | None = None,
    feature_set_id: str | None = None,
    signal_direction: float | int | None = None,
    eligible_for_backtest: bool | None = None,
    eligible_for_live: bool | None = None,
    run_id: str | None = None,
    configuration: Mapping[str, Any] | None = None,
    lineage: Sequence[tuple[str, str]] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    signals = build_signal_artifact_frame(
        scored,
        model_version=model_version,
        feature_set_id=feature_set_id,
        signal_direction=signal_direction,
        eligible_for_backtest=eligible_for_backtest,
        eligible_for_live=eligible_for_live,
    )
    signals.to_parquet(out_path, index=False)
    meta_path = out_path.with_name(CANONICAL_SIGNAL_META_FILE)
    summary = signal_artifact_summary(signals, path=out_path, metadata_path=meta_path)
    payload = {
        "artifact_type": SIGNAL_CONTRACT_NAME,
        "schema_version": SIGNAL_SCHEMA_VERSION,
        "summary": summary,
        "metadata": dict(metadata or {}),
    }
    resolved_run_id = run_id or default_signal_run_id()
    config_payload = {
        **dict(configuration or {}),
        "model_version": model_version,
        "feature_set_id": feature_set_id,
        "signal_direction": signal_direction,
        "eligible_for_backtest": eligible_for_backtest,
        "eligible_for_live": eligible_for_live,
    }
    envelope = build_signal_envelope_v2(
        artifact_id=f"signals:{resolved_run_id}",
        artifact_type=CANONICAL_SIGNAL_FILE,
        run_id=resolved_run_id,
        content_sha256=file_sha256(out_path),
        configuration=config_payload,
        producer_backend=PRODUCER_BACKEND_SIGNAL_ARTIFACT,
        lineage=lineage,
    )
    payload = attach_artifact_envelope_v2(payload, envelope)
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return signals, summary


def read_signal_artifact(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    signals = pd.read_parquet(Path(path))
    if validate:
        assert_signal_artifact_frame(signals)
    return signals


def load_signal_metadata(path: str | Path) -> dict[str, Any]:
    meta_path = Path(path).with_name(CANONICAL_SIGNAL_META_FILE)
    if not meta_path.exists():
        return {}
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


__all__ = [
    "CANONICAL_SIGNAL_COLUMNS",
    "CANONICAL_SIGNAL_FILE",
    "CANONICAL_SIGNAL_META_FILE",
    "PRODUCER_BACKEND_SIGNAL_ARTIFACT",
    "PRODUCER_BACKEND_STYLE_REPLICA",
    "PRODUCER_REPOSITORY",
    "SIGNAL_CONTRACT",
    "SIGNAL_CONTRACT_NAME",
    "SIGNAL_SCHEMA_VERSION",
    "SignalArtifactContract",
    "assert_signal_artifact_frame",
    "build_signal_artifact_frame",
    "build_signal_envelope_v2",
    "default_signal_run_id",
    "load_signal_metadata",
    "read_signal_artifact",
    "signal_artifact_summary",
    "validate_signal_artifact_frame",
    "write_signal_artifact",
]
