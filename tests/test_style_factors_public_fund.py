from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_research.style_factors.factor_calc import FACTOR_COLS
from alpha_research.style_factors.helpers._new_factors import (
    _add_public_fund_ownership_factors,
)


PUBLIC_FUND_FACTOR_COLUMNS = {
    "factor_fund_breadth",
    "factor_fund_breadth_change",
    "factor_fund_ownership",
    "factor_fund_ownership_change",
}


def _formation_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-08-29", "2025-08-29"]),
            "symbol": ["000001.SZ", "000002.SZ"],
        }
    )


def test_public_fund_factors_use_pit_materialized_formation_values() -> None:
    fund_portfolio = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-08-29", "2025-08-29"]),
            "symbol": ["000001.SZ", "000002.SZ"],
            "fund_count_holding_stock": [9.0, 99.0],
            "fund_count_holding_stock_change": [-3.0, 8.0],
            "fund_stk_float_ratio_sum": [1.0, 8.0],
            "fund_stk_float_ratio_sum_change": [-0.5, 1.2],
        }
    )

    result = _add_public_fund_ownership_factors(
        _formation_panel(),
        fund_portfolio=fund_portfolio,
    )

    assert np.isclose(result.loc[0, "factor_fund_breadth"], np.log1p(9.0))
    assert np.isclose(result.loc[1, "factor_fund_breadth"], np.log1p(99.0))
    assert np.isclose(result.loc[0, "factor_fund_breadth_change"], -np.log1p(3.0))
    assert np.isclose(result.loc[1, "factor_fund_breadth_change"], np.log1p(8.0))
    assert result["factor_fund_ownership"].tolist() == [1.0, 8.0]
    assert result["factor_fund_ownership_change"].tolist() == [-0.5, 1.2]


def test_public_fund_factors_are_missing_without_source_data() -> None:
    result = _add_public_fund_ownership_factors(
        _formation_panel(),
        fund_portfolio=None,
    )

    assert result[list(PUBLIC_FUND_FACTOR_COLUMNS)].isna().all().all()


def test_public_fund_factor_columns_are_registered() -> None:
    assert PUBLIC_FUND_FACTOR_COLUMNS <= set(FACTOR_COLS)
