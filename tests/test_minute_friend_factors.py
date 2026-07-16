from __future__ import annotations

import math
from collections.abc import Iterable

import duckdb
import pytest

from alpha_research.minute_friend_factors import (
    FRIEND_HARDENED_RV_COLUMNS,
    FRIEND_MINUTE_OUTPUT_COLUMNS,
    FRIEND_RV_COMPONENT_COLUMNS,
    FRIEND_VOLUME_ACTIVITY5_COLUMNS,
    friend_minute_feature_query,
)

BAR_COLUMNS = (
    "trade_date",
    "symbol",
    "bar_index",
    "session_id",
    "open",
    "close",
    "volume",
)


def _query_rows(rows: Iterable[tuple[object, ...]]) -> tuple[list[str], list[tuple[object, ...]]]:
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE minute_bars (
                trade_date VARCHAR,
                symbol VARCHAR,
                bar_index BIGINT,
                session_id VARCHAR,
                open DOUBLE,
                close DOUBLE,
                volume DOUBLE
            )
            """
        )
        connection.executemany(
            "INSERT INTO minute_bars VALUES (?, ?, ?, ?, ?, ?, ?)",
            list(rows),
        )
        result = connection.execute(friend_minute_feature_query())
        names = [str(item[0]) for item in result.description]
        return names, result.fetchall()
    finally:
        connection.close()


def _record(rows: Iterable[tuple[object, ...]]) -> dict[str, object]:
    columns, output = _query_rows(rows)
    assert columns == list(FRIEND_MINUTE_OUTPUT_COLUMNS)
    assert len(output) == 1
    return dict(zip(columns, output[0], strict=True))


def test_constant_volume_matches_replication_semantics() -> None:
    rows = [("20260105", "000001.SZ", index, "AM", 100.0, 100.0, 10.0) for index in range(1, 5)]

    record = _record(rows)

    assert record["friend_volume_cv_replication"] == pytest.approx(0.0)
    assert record["friend_log_volume_sd_replication"] == pytest.approx(0.0)
    assert record["friend_volume_absdiff_ratio_replication"] == pytest.approx(0.0)
    assert record["friend_peak_runs_1std_replication"] == 0
    assert record["friend_peak_runs_2std_replication"] == 0
    assert record["rv_oc_cc_1m"] == pytest.approx(0.0)
    assert record["bpv_oc_cc_1m"] == pytest.approx(0.0)
    assert record["minute_bar_count"] == 4
    assert record["minute_positive_volume_bar_count"] == 4


def test_replication_peak_run_does_not_reset_at_lunch() -> None:
    volumes = (1.0, 1.0, 10.0, 10.0, 1.0, 1.0)
    rows = [
        (
            "20260105",
            "000001.SZ",
            index,
            "AM" if index <= 3 else "PM",
            100.0,
            100.0,
            volume,
        )
        for index, volume in enumerate(volumes, start=1)
    ]

    record = _record(rows)

    assert record["friend_peak_runs_1std_replication"] == 1
    assert record["friend_peak_runs_2std_replication"] == 0
    assert record["minute_internal_missing_bar_count"] == 0


def test_first_bar_open_close_return_is_included() -> None:
    rows = [
        ("20260105", "000001.SZ", 1, "AM", 100.0, 101.0, 10.0),
        ("20260105", "000001.SZ", 2, "AM", 101.0, 101.0, 10.0),
    ]
    first_return = math.log(101.0 / 100.0)

    record = _record(rows)

    assert record["rv_oc_cc_1m"] == pytest.approx(first_return**2)
    assert record["log_rv_oc_cc_1m"] == pytest.approx(math.log(first_return**2))
    assert record["downside_rv_share_oc_cc_1m"] == pytest.approx(0.0)
    assert record["bpv_oc_cc_1m"] == pytest.approx(0.0)
    assert record["jump_share_bpv_oc_cc_1m"] == pytest.approx(1.0)


def test_lunch_boundary_keeps_consecutive_close_to_close_return() -> None:
    rows = [
        ("20260105", "000001.SZ", 120, "AM", 100.0, 101.0, 10.0),
        ("20260105", "000001.SZ", 121, "PM", 102.0, 103.0, 10.0),
    ]
    expected = math.log(101.0 / 100.0) ** 2 + math.log(103.0 / 101.0) ** 2
    reset_at_lunch = math.log(101.0 / 100.0) ** 2 + math.log(103.0 / 102.0) ** 2

    record = _record(rows)

    assert record["rv_oc_cc_1m"] == pytest.approx(expected)
    assert record["rv_oc_cc_1m"] != pytest.approx(reset_at_lunch)
    assert record["minute_internal_missing_bar_count"] == 0


def test_missing_bar_breaks_return_and_bpv_adjacency() -> None:
    rows = [
        ("20260105", "000001.SZ", 1, "AM", 100.0, 101.0, 10.0),
        ("20260105", "000001.SZ", 3, "AM", 103.0, 104.0, 10.0),
        ("20260105", "000001.SZ", 4, "AM", 104.0, 105.0, 10.0),
    ]
    expected_rv = math.log(101.0 / 100.0) ** 2 + math.log(105.0 / 104.0) ** 2
    bridged_rv = expected_rv + math.log(104.0 / 101.0) ** 2

    record = _record(rows)

    assert record["minute_internal_missing_bar_count"] == 1
    assert record["minute_valid_return_count"] == 2
    assert record["rv_oc_cc_1m"] == pytest.approx(expected_rv)
    assert record["rv_oc_cc_1m"] != pytest.approx(bridged_rv)
    assert record["bpv_oc_cc_1m"] is None
    assert record["jump_share_bpv_oc_cc_1m"] is None


def test_duplicate_bar_is_diagnosed_and_factor_values_fail_closed() -> None:
    rows = [
        ("20260105", "000001.SZ", 1, "AM", 100.0, 101.0, 10.0),
        ("20260105", "000001.SZ", 1, "AM", 100.0, 101.5, 12.0),
        ("20260105", "000001.SZ", 2, "AM", 101.0, 102.0, 10.0),
    ]

    record = _record(rows)

    assert record["minute_bar_count"] == 3
    assert record["minute_distinct_bar_count"] == 2
    assert record["minute_duplicate_bar_count"] == 1
    assert record["minute_bar_coverage"] == pytest.approx(2 / 240)
    for column in (
        *FRIEND_VOLUME_ACTIVITY5_COLUMNS,
        *FRIEND_HARDENED_RV_COLUMNS,
        *FRIEND_RV_COMPONENT_COLUMNS,
    ):
        assert record[column] is None


def test_optional_universe_relation_filters_before_aggregation() -> None:
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE minute_bars AS
            SELECT * FROM (VALUES
                ('20260105', '000001.SZ', 1, 'AM', 100.0, 100.0, 10.0),
                ('20260105', '000002.SZ', 1, 'AM', 100.0, 100.0, 10.0)
            ) AS bars(trade_date, symbol, bar_index, session_id, open, close, volume)
            """
        )
        connection.execute(
            "CREATE TABLE eligible AS SELECT '20260105' trade_date, '000002.SZ' symbol"
        )

        result = connection.execute(
            friend_minute_feature_query(universe_relation_sql="eligible")
        ).fetchall()

        assert len(result) == 1
        assert result[0][:2] == ("20260105", "000002.SZ")
    finally:
        connection.close()


def test_query_configuration_rejects_ambiguous_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="expected_bar_count"):
        friend_minute_feature_query(expected_bar_count=0)
    with pytest.raises(ValueError, match="disagree"):
        friend_minute_feature_query("bars_a", raw_relation_sql="bars_b")
    with pytest.raises(ValueError, match="semicolon"):
        friend_minute_feature_query("minute_bars;")
