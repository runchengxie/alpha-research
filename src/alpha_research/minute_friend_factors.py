"""DuckDB SQL semantics for the friend-derived minute factor challenger.

The module is deliberately I/O free.  It defines the stock-day output contract
and builds a DuckDB query over a caller-provided relation.  Input bars must
already be normalized to a canonical, globally increasing ``bar_index`` within
each stock-day.  ``session_id`` identifies the morning and afternoon sessions
for diagnostics.

The five ``friend_*`` columns reproduce the selected ``volume_activity5``
definitions.  The realized-variation columns use stricter missing-bar
semantics: the first stock-day return is close/open, and close/close returns
are only formed between consecutive canonical bars.  A correctly normalized
``bar_index`` makes 11:30 and 13:01 consecutive, so the lunch return is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

MINUTE_FRIEND_FACTOR_SCHEMA = "alpha_research.minute_friend_factors.v1"

FRIEND_MINUTE_KEY_COLUMNS = ("trade_date", "symbol")

FRIEND_VOLUME_ACTIVITY5_COLUMNS = (
    "friend_volume_cv_replication",
    "friend_log_volume_sd_replication",
    "friend_volume_absdiff_ratio_replication",
    "friend_peak_runs_1std_replication",
    "friend_peak_runs_2std_replication",
)

FRIEND_HARDENED_RV_COLUMNS = (
    "log_rv_oc_cc_1m",
    "downside_rv_share_oc_cc_1m",
    "jump_share_bpv_oc_cc_1m",
)

FRIEND_RV_COMPONENT_COLUMNS = (
    "rv_oc_cc_1m",
    "bpv_oc_cc_1m",
)

FRIEND_MINUTE_MODEL_FEATURES = (
    *FRIEND_VOLUME_ACTIVITY5_COLUMNS,
    *FRIEND_HARDENED_RV_COLUMNS,
)

FRIEND_MINUTE_DIAGNOSTIC_COLUMNS = (
    "minute_bar_count",
    "minute_distinct_bar_count",
    "minute_positive_volume_bar_count",
    "minute_valid_return_count",
    "minute_duplicate_bar_count",
    "minute_internal_missing_bar_count",
    "minute_invalid_price_bar_count",
    "minute_nonpositive_volume_bar_count",
    "minute_bar_coverage",
)

FRIEND_MINUTE_OUTPUT_COLUMNS = (
    *FRIEND_MINUTE_KEY_COLUMNS,
    *FRIEND_VOLUME_ACTIVITY5_COLUMNS,
    *FRIEND_HARDENED_RV_COLUMNS,
    *FRIEND_RV_COMPONENT_COLUMNS,
    *FRIEND_MINUTE_DIAGNOSTIC_COLUMNS,
)


@dataclass(frozen=True)
class FriendMinuteSqlColumns:
    """Column mapping for :func:`friend_minute_feature_query`.

    ``bar_index`` must be unique and globally increasing across both sessions
    of a valid stock-day.  Duplicate indices are retained in diagnostics and
    make factor values NULL for that stock-day.
    """

    trade_date: str = "trade_date"
    symbol: str = "symbol"
    bar_index: str = "bar_index"
    session_id: str = "session_id"
    open: str = "open"
    close: str = "close"
    volume: str = "volume"


def _quoted_identifier(value: str) -> str:
    if not value:
        raise ValueError("SQL column identifiers must be non-empty")
    return '"' + value.replace('"', '""') + '"'


def _relation_sql(value: str, *, name: str) -> str:
    relation = value.strip()
    if not relation:
        raise ValueError(f"{name} must be non-empty")
    if relation.endswith(";"):
        raise ValueError(f"{name} must not end with a semicolon")
    return relation


def friend_minute_feature_query(
    relation: str = "minute_bars",
    *,
    raw_relation_sql: str | None = None,
    universe_relation_sql: str | None = None,
    columns: FriendMinuteSqlColumns | None = None,
    expected_bar_count: int = 240,
) -> str:
    """Return the complete DuckDB SELECT for friend-derived stock-day factors.

    ``relation`` and its explicit ``raw_relation_sql`` alias are trusted SQL
    fragments. ``universe_relation_sql`` optionally filters stock-day keys
    before windows and aggregates. Column identifiers are always quoted, while
    ``expected_bar_count`` only controls the coverage diagnostic denominator.
    """

    if expected_bar_count <= 0:
        raise ValueError("expected_bar_count must be positive")
    if raw_relation_sql is not None:
        if relation != "minute_bars" and relation.strip() != raw_relation_sql.strip():
            raise ValueError("relation and raw_relation_sql disagree")
        relation = raw_relation_sql
    raw_relation = _relation_sql(relation, name="relation")
    universe_relation = (
        _relation_sql(universe_relation_sql, name="universe_relation_sql")
        if universe_relation_sql is not None
        else None
    )
    source_columns = columns or FriendMinuteSqlColumns()
    trade_date = _quoted_identifier(source_columns.trade_date)
    symbol = _quoted_identifier(source_columns.symbol)
    bar_index = _quoted_identifier(source_columns.bar_index)
    session_id = _quoted_identifier(source_columns.session_id)
    open_price = _quoted_identifier(source_columns.open)
    close_price = _quoted_identifier(source_columns.close)
    volume = _quoted_identifier(source_columns.volume)

    universe_join = ""
    if universe_relation is not None:
        universe_join = f"""
        INNER JOIN (
            SELECT DISTINCT
                CAST(trade_date AS VARCHAR) AS trade_date,
                CAST(symbol AS VARCHAR) AS symbol
            FROM {universe_relation}
        ) AS eligible
          ON CAST(source.{trade_date} AS VARCHAR) = eligible.trade_date
         AND CAST(source.{symbol} AS VARCHAR) = eligible.symbol
        """

    return _FRIEND_MINUTE_QUERY_TEMPLATE.format(
        trade_date=trade_date,
        symbol=symbol,
        bar_index=bar_index,
        session_id=session_id,
        open_price=open_price,
        close_price=close_price,
        volume=volume,
        raw_relation=raw_relation,
        universe_join=universe_join,
        expected_bar_count=expected_bar_count,
    ).strip()


_FRIEND_MINUTE_QUERY_TEMPLATE = """
WITH raw_bars AS (
    SELECT
        CAST(source.{trade_date} AS VARCHAR) AS trade_date,
        CAST(source.{symbol} AS VARCHAR) AS symbol,
        CAST(source.{bar_index} AS BIGINT) AS bar_index,
        CAST(source.{session_id} AS VARCHAR) AS session_id,
        CAST(source.{open_price} AS DOUBLE) AS open_price,
        CAST(source.{close_price} AS DOUBLE) AS close_price,
        CAST(source.{volume} AS DOUBLE) AS volume
    FROM {raw_relation} AS source
    {universe_join}
),
day_diagnostics AS (
    SELECT
        trade_date,
        symbol,
        count(*) AS minute_bar_count,
        count(DISTINCT bar_index) AS minute_distinct_bar_count,
        count(*) - count(DISTINCT bar_index) AS minute_duplicate_bar_count,
        count(*) FILTER (WHERE volume > 0) AS minute_positive_volume_bar_count,
        count(*) FILTER (
            WHERE open_price IS NULL OR close_price IS NULL
               OR open_price <= 0 OR close_price <= 0
        ) AS minute_invalid_price_bar_count,
        count(*) FILTER (WHERE volume IS NULL OR volume <= 0)
            AS minute_nonpositive_volume_bar_count
    FROM raw_bars
    GROUP BY trade_date, symbol
),
ranked_bars AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY trade_date, symbol, bar_index
            ORDER BY session_id, open_price, close_price, volume
        ) AS duplicate_ordinal
    FROM raw_bars
    WHERE bar_index IS NOT NULL
),
deduplicated_bars AS (
    SELECT
        trade_date,
        symbol,
        bar_index,
        session_id,
        open_price,
        close_price,
        volume,
        CASE WHEN volume > 0 THEN volume END AS positive_volume
    FROM ranked_bars
    WHERE duplicate_ordinal = 1
),
ordered_bars AS (
    SELECT
        *,
        lag(bar_index) OVER stock_day AS previous_bar_index,
        lag(session_id) OVER stock_day AS previous_session_id,
        lag(close_price) OVER stock_day AS previous_close_price,
        lag(positive_volume) OVER stock_day AS previous_positive_volume
    FROM deduplicated_bars
    WINDOW stock_day AS (
        PARTITION BY trade_date, symbol ORDER BY bar_index
    )
),
bar_returns AS (
    SELECT
        *,
        CASE
            WHEN previous_bar_index IS NULL
            THEN CASE
                WHEN open_price > 0 AND close_price > 0
                THEN ln(close_price / open_price)
            END
            WHEN bar_index = previous_bar_index + 1
             AND previous_close_price > 0 AND close_price > 0
            THEN ln(close_price / previous_close_price)
        END AS minute_return,
        CASE
            WHEN previous_bar_index IS NOT NULL
             AND bar_index > previous_bar_index + 1
            THEN bar_index - previous_bar_index - 1
            ELSE 0
        END AS internal_missing_bars
    FROM ordered_bars
),
return_pairs AS (
    SELECT
        *,
        CASE
            WHEN previous_bar_index IS NOT NULL
             AND bar_index = previous_bar_index + 1
            THEN lag(minute_return) OVER (
                PARTITION BY trade_date, symbol ORDER BY bar_index
            )
        END AS previous_minute_return
    FROM bar_returns
),
day_values AS (
    SELECT
        trade_date,
        symbol,
        avg(positive_volume) AS volume_mean,
        stddev_pop(positive_volume) AS volume_sd,
        CASE
            WHEN count(positive_volume) > 1
            THEN stddev_pop(ln(positive_volume + 10.0))
        END AS friend_log_volume_volatility,
        avg(abs(positive_volume - previous_positive_volume)) FILTER (
            WHERE positive_volume IS NOT NULL
              AND previous_positive_volume IS NOT NULL
        ) / nullif(avg(positive_volume), 0)
            AS friend_diff_abs_mean_volume,
        sum(minute_return * minute_return) AS minute_rv,
        sum(
            CASE WHEN minute_return < 0
                 THEN minute_return * minute_return ELSE 0 END
        ) AS minute_downside_rv,
        count(minute_return) AS minute_valid_return_count,
        CASE
            WHEN count(previous_minute_return) FILTER (
                WHERE minute_return IS NOT NULL
                  AND previous_minute_return IS NOT NULL
            ) > 0
            THEN (pi() / 2.0)
                * count(minute_return)::DOUBLE
                / count(previous_minute_return) FILTER (
                    WHERE minute_return IS NOT NULL
                      AND previous_minute_return IS NOT NULL
                )
                * sum(abs(minute_return) * abs(previous_minute_return)) FILTER (
                    WHERE minute_return IS NOT NULL
                      AND previous_minute_return IS NOT NULL
                )
        END AS minute_bpv,
        sum(internal_missing_bars) AS minute_internal_missing_bar_count
    FROM return_pairs
    GROUP BY trade_date, symbol
),
volume_flags AS (
    SELECT
        bars.*,
        values.volume_mean,
        values.volume_sd,
        coalesce(
            bars.positive_volume > values.volume_mean + values.volume_sd,
            false
        ) AS above_1std,
        coalesce(
            bars.positive_volume > values.volume_mean + 2.0 * values.volume_sd,
            false
        ) AS above_2std
    FROM return_pairs AS bars
    INNER JOIN day_values AS values USING (trade_date, symbol)
),
volume_edges AS (
    SELECT
        *,
        lag(above_1std, 1, false) OVER (
            PARTITION BY trade_date, symbol ORDER BY bar_index
        ) AS previous_above_1std,
        lag(above_2std, 1, false) OVER (
            PARTITION BY trade_date, symbol ORDER BY bar_index
        ) AS previous_above_2std
    FROM volume_flags
),
peak_counts AS (
    SELECT
        trade_date,
        symbol,
        sum(CASE WHEN above_1std AND NOT previous_above_1std THEN 1 ELSE 0 END)
            AS friend_peak_count_1std,
        sum(CASE WHEN above_2std AND NOT previous_above_2std THEN 1 ELSE 0 END)
            AS friend_peak_count_2std
    FROM volume_edges
    GROUP BY trade_date, symbol
)
SELECT
    diagnostics.trade_date,
    diagnostics.symbol,
    CASE WHEN diagnostics.minute_duplicate_bar_count = 0
         THEN values.volume_sd / nullif(values.volume_mean, 0) END
        AS friend_volume_cv_replication,
    CASE WHEN diagnostics.minute_duplicate_bar_count = 0
         THEN values.friend_log_volume_volatility END
        AS friend_log_volume_sd_replication,
    CASE WHEN diagnostics.minute_duplicate_bar_count = 0
         THEN values.friend_diff_abs_mean_volume END
        AS friend_volume_absdiff_ratio_replication,
    CASE
        WHEN diagnostics.minute_duplicate_bar_count <> 0 THEN NULL
        WHEN values.volume_mean IS NULL OR values.volume_sd IS NULL THEN NULL
        ELSE peaks.friend_peak_count_1std
    END AS friend_peak_runs_1std_replication,
    CASE
        WHEN diagnostics.minute_duplicate_bar_count <> 0 THEN NULL
        WHEN values.volume_mean IS NULL OR values.volume_sd IS NULL THEN NULL
        ELSE peaks.friend_peak_count_2std
    END AS friend_peak_runs_2std_replication,
    CASE WHEN diagnostics.minute_duplicate_bar_count = 0 AND values.minute_rv > 0
         THEN ln(values.minute_rv) END AS log_rv_oc_cc_1m,
    CASE WHEN diagnostics.minute_duplicate_bar_count = 0 AND values.minute_rv > 0
         THEN values.minute_downside_rv / values.minute_rv END
        AS downside_rv_share_oc_cc_1m,
    CASE
        WHEN diagnostics.minute_duplicate_bar_count = 0
         AND values.minute_rv > 0 AND values.minute_bpv IS NOT NULL
        THEN greatest(values.minute_rv - values.minute_bpv, 0) / values.minute_rv
    END AS jump_share_bpv_oc_cc_1m,
    CASE WHEN diagnostics.minute_duplicate_bar_count = 0
         THEN values.minute_rv END AS rv_oc_cc_1m,
    CASE WHEN diagnostics.minute_duplicate_bar_count = 0
         THEN values.minute_bpv END AS bpv_oc_cc_1m,
    diagnostics.minute_bar_count,
    diagnostics.minute_distinct_bar_count,
    diagnostics.minute_positive_volume_bar_count,
    values.minute_valid_return_count,
    diagnostics.minute_duplicate_bar_count,
    values.minute_internal_missing_bar_count,
    diagnostics.minute_invalid_price_bar_count,
    diagnostics.minute_nonpositive_volume_bar_count,
    diagnostics.minute_distinct_bar_count::DOUBLE / {expected_bar_count}
        AS minute_bar_coverage
FROM day_diagnostics AS diagnostics
LEFT JOIN day_values AS values USING (trade_date, symbol)
LEFT JOIN peak_counts AS peaks USING (trade_date, symbol)
ORDER BY diagnostics.trade_date, diagnostics.symbol
"""


__all__ = [
    "FRIEND_HARDENED_RV_COLUMNS",
    "FRIEND_MINUTE_DIAGNOSTIC_COLUMNS",
    "FRIEND_MINUTE_KEY_COLUMNS",
    "FRIEND_MINUTE_MODEL_FEATURES",
    "FRIEND_MINUTE_OUTPUT_COLUMNS",
    "FRIEND_RV_COMPONENT_COLUMNS",
    "FRIEND_VOLUME_ACTIVITY5_COLUMNS",
    "MINUTE_FRIEND_FACTOR_SCHEMA",
    "FriendMinuteSqlColumns",
    "friend_minute_feature_query",
]
