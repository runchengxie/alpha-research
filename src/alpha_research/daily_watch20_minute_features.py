"""Owner-native DailyWatch20 minute transforms and feature evidence."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from market_data_platform.research_views.daily_watch20_minute_source import MinuteSourceCatalog

MINUTE_FEATURE_SCHEMA = "daily_watch20.minute_features.v3"
MINUTE_TRANSFORM_CONTRACT = "daily_watch20.minute_features.close_open.v3"
MINUTE_FEATURE_COLUMNS = (
    "trade_date",
    "symbol",
    "minute_realized_vol",
    "minute_downside_vol",
    "minute_range_pct",
    "minute_close_location",
    "minute_last_30m_return",
    "minute_open_30m_volume_share",
    "minute_last_30m_volume_share",
    "minute_volume_concentration",
    "minute_active_ratio",
    "minute_price_volume_corr",
    "minute_volume_activity",
    "minute_bar_count",
)


@dataclass(frozen=True, slots=True)
class MinuteFeatureTransformResult:
    frame: pd.DataFrame
    evidence: dict[str, Any]


def _duckdb() -> Any:
    try:
        return importlib.import_module("duckdb")
    except ImportError as exc:
        raise RuntimeError("DailyWatch20 minute transforms require DuckDB") from exc


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_file_list(paths: tuple[Path, ...]) -> str:
    if not paths:
        raise ValueError("At least one minute parquet file is required")
    return "[" + ", ".join(_sql_literal(path) for path in paths) + "]"


def daily_watch20_minute_feature_sql(paths: tuple[Path, ...]) -> str:
    """Return the frozen close/open minute aggregation contract."""

    return f"""
        WITH returns AS (
            SELECT
                CAST(trade_date AS VARCHAR) AS trade_date,
                ts_code AS symbol,
                trade_time,
                CAST(trade_time AS TIME) AS bar_time,
                open,
                close,
                high,
                low,
                vol,
                CASE
                    WHEN CAST(trade_time AS TIME) >= TIME '09:35:00'
                     AND open > 0 AND close > 0
                    THEN close / open - 1
                END AS minute_ret
            FROM read_parquet(
                {_sql_file_list(paths)},
                hive_partitioning = true,
                union_by_name = true
            )
            WHERE ts_code LIKE '%.SH' OR ts_code LIKE '%.SZ'
        )
        SELECT
            trade_date,
            symbol,
            sqrt(sum(minute_ret * minute_ret)) AS minute_realized_vol,
            sqrt(sum(CASE WHEN minute_ret < 0 THEN minute_ret * minute_ret ELSE 0 END))
                AS minute_downside_vol,
            (max(high) - min(low)) / nullif(arg_min(open, trade_time), 0)
                AS minute_range_pct,
            (arg_max(close, trade_time) - min(low)) / nullif(max(high) - min(low), 0)
                AS minute_close_location,
            arg_max(close, trade_time) FILTER (WHERE bar_time >= TIME '14:30:00') /
                nullif(
                    arg_min(open, trade_time) FILTER (WHERE bar_time >= TIME '14:30:00'),
                    0
                ) - 1 AS minute_last_30m_return,
            sum(vol) FILTER (
                WHERE bar_time >= TIME '09:35:00' AND bar_time < TIME '10:05:00'
            ) / nullif(sum(vol), 0) AS minute_open_30m_volume_share,
            sum(vol) FILTER (WHERE bar_time >= TIME '14:30:00') /
                nullif(sum(vol), 0) AS minute_last_30m_volume_share,
            sum(vol * vol) / nullif(sum(vol) * sum(vol), 0)
                AS minute_volume_concentration,
            count(*) FILTER (WHERE vol > 0)::DOUBLE / count(*) AS minute_active_ratio,
            corr(vol, minute_ret) AS minute_price_volume_corr,
            stddev_samp(vol) / nullif(avg(vol), 0) AS minute_volume_activity,
            count(*) AS minute_bar_count
        FROM returns
        GROUP BY trade_date, symbol
        ORDER BY trade_date, symbol
    """


def validate_daily_watch20_minute_feature_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate schema, keys, market scope, and finite volume-activity semantics."""

    columns = tuple(frame.columns)
    if columns != MINUTE_FEATURE_COLUMNS:
        raise ValueError(
            f"invalid minute feature columns: expected={MINUTE_FEATURE_COLUMNS}, actual={columns}"
        )
    if frame.empty:
        raise ValueError("minute feature frame is empty")
    if frame.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("minute feature frame contains duplicate stock-date keys")
    symbols = cast(pd.Series, frame["symbol"]).astype(str)
    if not symbols.str.fullmatch(r"\d{6}\.(?:SH|SZ)").all():
        raise ValueError("minute feature frame contains out-of-scope symbols")
    volume_activity = cast(
        pd.Series, pd.to_numeric(frame["minute_volume_activity"], errors="coerce")
    )
    observed = volume_activity.dropna().to_numpy(dtype=float)
    if not len(observed) or not np.isfinite(observed).all() or (observed < 0).any():
        raise ValueError("minute_volume_activity must contain finite non-negative observations")
    dates = cast(pd.Series, frame["trade_date"]).astype(str)
    date_rows = dates.value_counts().sort_index().astype(int).to_dict()
    return {
        "rows": len(frame),
        "symbols": int(symbols.nunique()),
        "date_min": min(date_rows),
        "date_max": max(date_rows),
        "date_rows": date_rows,
        "volume_activity_rows": int(volume_activity.notna().sum()),
    }


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    normalized = frame.loc[:, list(MINUTE_FEATURE_COLUMNS)].copy()
    normalized["trade_date"] = normalized["trade_date"].astype(str)
    normalized["symbol"] = normalized["symbol"].astype(str)
    hashed = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def transform_daily_watch20_minute_catalog(
    catalog: MinuteSourceCatalog,
    *,
    memory_limit: str = "12GB",
    threads: int = 3,
) -> MinuteFeatureTransformResult:
    """Transform an owner-resolved source catalog into validated stock-day features."""

    conn = _duckdb().connect()
    try:
        conn.execute(f"SET threads = {max(1, int(threads))}")
        conn.execute(f"SET memory_limit = {_sql_literal(memory_limit)}")
        frame = conn.execute(daily_watch20_minute_feature_sql(catalog.files)).fetch_df()
    finally:
        conn.close()
    frame = frame.loc[:, list(MINUTE_FEATURE_COLUMNS)]
    summary = validate_daily_watch20_minute_feature_frame(frame)
    partition_records = catalog.partition_records()
    for date, rows in summary["date_rows"].items():
        if date not in partition_records:
            raise ValueError(f"minute transform produced an unbound source date: {date}")
        partition_records[date]["feature_rows"] = int(rows)
    if set(summary["date_rows"]) != set(partition_records):
        raise ValueError("minute transform/source catalog dates do not match")
    evidence = {
        "schema_version": MINUTE_FEATURE_SCHEMA,
        "source_contract": catalog.source_contract,
        "transform_contract": MINUTE_TRANSFORM_CONTRACT,
        "source": catalog.source_record(),
        "source_partitions": partition_records,
        **{key: value for key, value in summary.items() if key != "date_rows"},
        "frame_sha256": _frame_fingerprint(frame),
        "policy": {
            "market_scope": "sh-sz",
            "exclude_exact_open_minutes_before": "09:35:00",
            "minute_return_definition": "close/open-1 within each minute bar",
            "minute_volume_activity_definition": "stddev_samp(volume)/avg(volume)",
            "absolute_vwap_features": False,
        },
    }
    return MinuteFeatureTransformResult(frame=frame, evidence=evidence)


__all__ = [
    "MINUTE_FEATURE_COLUMNS",
    "MINUTE_FEATURE_SCHEMA",
    "MINUTE_TRANSFORM_CONTRACT",
    "MinuteFeatureTransformResult",
    "daily_watch20_minute_feature_sql",
    "transform_daily_watch20_minute_catalog",
    "validate_daily_watch20_minute_feature_frame",
]
