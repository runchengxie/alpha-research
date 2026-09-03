from __future__ import annotations

from typing import cast

import pandas as pd

from alpha_research.dataset import DatasetSchema
from alpha_research.date_slices import build_trade_date_slices


def test_dataset_schema_preserves_declared_column_order() -> None:
    schema = DatasetSchema(
        date_col="trade_date",
        instrument_col="symbol",
        price_col="adj_open",
        label_col="label",
        tradable_col="tradable",
        feature_cols=["mom_20"],
        extra_cols=["symbol", "sector"],
    )

    assert schema.column_order() == [
        "trade_date",
        "symbol",
        "adj_open",
        "mom_20",
        "sector",
        "label",
        "tradable",
    ]


def test_date_slices_group_rows_by_sorted_trade_date() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-06", "2026-01-05", "2026-01-06"]),
            "symbol": ["BBB", "AAA", "AAA"],
        }
    )

    ordered, dates, start_rows, end_rows, date_to_pos = build_trade_date_slices(frame)

    assert ordered["symbol"].tolist() == ["AAA", "BBB", "AAA"]
    assert pd.to_datetime(dates).tolist() == [
        pd.Timestamp("2026-01-05"),
        pd.Timestamp("2026-01-06"),
    ]
    assert start_rows.tolist() == [0, 1]
    assert end_rows.tolist() == [1, 3]
    assert date_to_pos[cast(pd.Timestamp, pd.Timestamp("2026-01-06"))] == 1
