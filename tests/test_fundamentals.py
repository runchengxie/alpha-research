import numpy as np
import pandas as pd

from cstree.alpha.fundamentals import (
    derive_requested_fundamental_fields,
    fundamental_source_fields,
)


def test_fundamental_source_fields_include_raw_dependencies() -> None:
    fields = fundamental_source_fields(
        {"profit_margin", "delta_sales", "growth_debt", "net_debt_to_assets"}
    )

    assert {"revenue", "operating_revenue", "net_profit"}.issubset(fields)
    assert {"short_term_debt", "long_term_loans", "cash_and_equivalents"}.issubset(fields)
    assert {"sales", "debt"}.issubset(fields)


def test_derive_requested_fundamental_fields_builds_research_features() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-01-31"]),
            "symbol": ["AAA", "AAA", "BBB"],
            "revenue": [100.0, 140.0, 50.0],
            "operating_revenue": [np.nan, np.nan, np.nan],
            "net_profit": [10.0, 21.0, 5.0],
            "cash_flow_from_operating_activities": [12.0, 18.0, 4.0],
            "short_term_debt": [30.0, 45.0, np.nan],
            "long_term_loans": [70.0, 75.0, np.nan],
            "cash_and_equivalents": [20.0, 15.0, 2.0],
            "total_assets": [400.0, 500.0, 100.0],
        }
    )

    result = derive_requested_fundamental_fields(
        frame,
        {
            "sales",
            "profit_margin",
            "cfo_margin",
            "debt",
            "net_debt_to_assets",
            "delta_sales",
            "growth_debt",
            "days_since_report",
        },
    )

    assert result is not frame
    assert result["sales"].tolist() == [100.0, 140.0, 50.0]
    assert result["profit_margin"].tolist() == [0.1, 0.15, 0.1]
    assert result["cfo_margin"].tolist() == [0.12, 18.0 / 140.0, 0.08]
    assert result["debt"].tolist()[:2] == [100.0, 120.0]
    assert pd.isna(result.loc[2, "debt"])
    assert result["net_debt_to_assets"].tolist()[:2] == [0.2, 0.21]
    assert pd.isna(result.loc[0, "delta_sales"])
    assert result.loc[1, "delta_sales"] == 40.0
    assert pd.isna(result.loc[0, "growth_debt"])
    assert np.isclose(result.loc[1, "growth_debt"], 20.0 / 110.0)
    assert result["report_trade_date"].equals(frame["trade_date"])
