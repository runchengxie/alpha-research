"""Equivalence/sanity tests for the migrated A-share style-factor kernel.

These cover the pure DataFrame-in / DataFrame-out computation moved from the
workspace ``src/style_factors`` package into ``alpha_research.style_factors``
(ADR-0006 R4 slice 7). Reporting/charting/backtest layers are owned elsewhere
and tested there.
"""

from __future__ import annotations

import pandas as pd

from alpha_research.style_factors import FACTOR_COLS, compute_factors


def _sample_market_frames(days: int = 90, symbols: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=days)
    symbol_values = [f"{index:06d}.SZ" for index in range(1, symbols + 1)]
    rows = []
    basic_rows = []
    for symbol_index, symbol in enumerate(symbol_values, start=1):
        for day_index, trade_date in enumerate(dates):
            close = 10 + symbol_index * 0.1 + day_index * 0.03
            pct_chg = 0.1 + ((symbol_index + day_index) % 7) * 0.02
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "close": close,
                    "pct_chg": pct_chg,
                    "amount": 1000 + symbol_index * 10 + day_index,
                }
            )
            basic_rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "total_mv": 10000 + symbol_index * 100,
                    "pb": 0.8 + symbol_index / 100,
                    "pe_ttm": 8 + symbol_index / 5,
                    "turnover_rate": 0.5 + symbol_index / 200,
                }
            )
    daily = pd.DataFrame(rows)
    basics = pd.DataFrame(basic_rows)
    return daily, basics


def _available_factor_z_columns(factors: pd.DataFrame) -> list[str]:
    return [f"{column}_z" for column in FACTOR_COLS if f"{column}_z" in factors.columns]


def test_compute_factors_emits_standardized_columns_without_fundamentals() -> None:
    # Without fina, the fundamental-derived factors (quality/growth/leverage)
    # are skipped; the price/basic-derived factors are still standardized.
    daily, basics = _sample_market_frames()
    factors = compute_factors(daily, basics)

    core_factors = (
        "factor_size",
        "factor_value",
        "factor_momentum",
        "factor_lowvol",
        "factor_liquidity",
    )
    for column in core_factors:
        assert f"{column}_z" in factors.columns
    assert "factor_quality_z" not in factors.columns
    assert "factor_growth_z" not in factors.columns
    assert "factor_leverage_z" not in factors.columns


def test_standardized_factors_have_near_zero_cross_section_mean() -> None:
    daily, basics = _sample_market_frames()
    factors = compute_factors(daily, basics)

    # After cross-sectional z-scoring, each available factor's per-date mean is ~0
    # (winsorization keeps it within a tiny tolerance). Factors with insufficient
    # history (e.g. beta's 252-day window on short samples) are all-NaN and skipped.
    for zcol in _available_factor_z_columns(factors):
        means = factors.groupby("trade_date")[zcol].mean().dropna()
        if means.empty:
            continue
        assert means.abs().max() < 1e-6, f"{zcol} cross-section mean not ~0"


def test_fundamental_factors_absent_without_fina() -> None:
    daily, basics = _sample_market_frames()
    factors = compute_factors(daily, basics)

    # Quality / Growth / Leverage depend on fundamentals; without fina skipped.
    assert "factor_growth_z" not in factors.columns
    assert "factor_leverage_z" not in factors.columns
    assert "factor_quality_z" not in factors.columns


def test_value_cluster_present_from_valuation() -> None:
    daily, basics = _sample_market_frames()
    factors = compute_factors(daily, basics)

    assert "factor_value_cluster_z" in factors.columns
    assert factors["factor_value_cluster_z"].notna().any()


def test_industry_membership_demeans_within_sw_l1() -> None:
    daily, basics = _sample_market_frames()
    # PIT SW-L1 long table format: [symbol, in_date, out_date, industry_l1].
    # Symbols 000001/000002 sit in industry A; 000003 in industry B.
    dates = sorted(daily["trade_date"].unique())
    membership_rows = []
    for symbol, ind in (("000001.SZ", "A"), ("000002.SZ", "A"), ("000003.SZ", "B")):
        membership_rows.append(
            {
                "symbol": symbol,
                "in_date": dates[0],
                "out_date": pd.NaT,
                "industry_l1": ind,
            }
        )
    membership = pd.DataFrame(membership_rows)
    factors = compute_factors(daily, basics, sw_membership=membership)

    assert "industry_l1" in factors.columns
    assert factors["industry_l1"].notna().any()
    grp_a = factors[factors["industry_l1"] == "A"]
    # Within-industry demeaning drives the A-group mean of any factor_z toward 0.
    for zcol in _available_factor_z_columns(factors):
        if grp_a[zcol].notna().any():
            assert abs(grp_a[zcol].mean()) < 1e-6, f"{zcol} A-group mean not demeaned"


def test_compute_factors_is_deterministic() -> None:
    daily, basics = _sample_market_frames()
    first = compute_factors(daily, basics)
    second = compute_factors(daily, basics)

    pd.testing.assert_frame_equal(first, second)
