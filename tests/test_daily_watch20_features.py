from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import alpha_research.daily_watch20_features as feature_module
from alpha_research.daily_watch20_features import (
    DAILY_WATCH20_FEATURES,
    DAILY_WATCH20_MARKET_SHADOW_DIAGNOSTICS,
    DAILY_WATCH20_MARKET_SHADOW_FEATURES,
    DEFAULT_LABEL_HORIZON_WEIGHTS,
    LEGACY_FIVE_DAY_LABEL_HORIZON_WEIGHTS,
    LIMIT_AWARE_NEXT_OPEN_LABEL_POLICY_ID,
    MINUTE_FEATURES,
    DailyWatch20FeatureConfig,
    _adjacent_trade_day_returns,
    build_daily_watch20_feature_frame,
)
from alpha_research.style_replica.factors import (
    compute_beta_factor,
    compute_hermite_stability_factor,
)


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _timestamp(value: object) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(cast(Any, value)))


def _is_missing(value: object) -> bool:
    return bool(pd.isna(value))


def _daily_panel(
    n_dates: int = 36,
    symbols: tuple[str, ...] = ("000001.SZ", "600000.SH"),
) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=n_dates)
    rows: list[dict[str, object]] = []
    for symbol_number, symbol in enumerate(symbols):
        for date_number, trade_date in enumerate(dates):
            if symbol_number == 0:
                price = 10.0 + date_number
            elif symbol.endswith(".BJ"):
                price = 20.0 + 2.0 * date_number
            else:
                price = 80.0 - 0.2 * date_number + symbol_number
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "open": price,
                    "adj_open": price,
                    "up_limit": price + 1.0,
                    "down_limit": price - 1.0,
                    "tr_close": price * (1.0 + 0.002 * ((date_number % 3) - 1)),
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "amount": float((symbol_number + 1) * 100_000_000 + date_number),
                    "turnover_rate": 1.0 + symbol_number * 0.1,
                    "total_mv": float((symbol_number + 1) * 10_000_000_000),
                    "pb": 1.5 + symbol_number * 0.1,
                    "pe_ttm": 10.0 + symbol_number,
                    "listed_days": 100 + date_number,
                    "is_st": False,
                    "is_suspended": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                }
            )
    return pd.DataFrame(rows)


def _row(frame: pd.DataFrame, date: pd.Timestamp, symbol: str) -> pd.Series:
    selected = frame.loc[frame["trade_date"].eq(date) & frame["symbol"].eq(symbol)]
    assert len(selected) == 1
    return selected.iloc[0]


def test_feature_config_defaults_to_weighted_horizons_and_supports_legacy_five_day() -> None:
    default = DailyWatch20FeatureConfig()
    legacy = DailyWatch20FeatureConfig(label_horizon_weights=LEGACY_FIVE_DAY_LABEL_HORIZON_WEIGHTS)

    assert default.label_horizon_weights == DEFAULT_LABEL_HORIZON_WEIGHTS
    assert default.minute_lag_trade_days == 0
    assert default.label_col == "forward_rank_blended"
    assert default.forward_return_col == "forward_return_blended"
    assert LIMIT_AWARE_NEXT_OPEN_LABEL_POLICY_ID.endswith("open_limit_aware.v3")
    assert legacy.label_col == "forward_rank_5d"
    with pytest.raises(ValueError, match="longest configured"):
        DailyWatch20FeatureConfig(forward_days=3)
    with pytest.raises(ValueError, match="non-negative integer"):
        DailyWatch20FeatureConfig(minute_lag_trade_days=cast(int, 0.5))


def test_rolling_features_are_symbol_local_and_input_order_invariant() -> None:
    daily = _daily_panel()
    shuffled = daily.sample(frac=1.0, random_state=7)
    shuffled.index = np.arange(len(shuffled)) * 11 + 3

    expected = build_daily_watch20_feature_frame(daily)
    actual = build_daily_watch20_feature_frame(shuffled)

    columns = ["trade_date", "symbol", "ret_1d", "mom_5", "vol_5", "amount_log_20"]
    assert_frame_equal(actual[columns], expected[columns])
    first_rows = actual.groupby("symbol", sort=False).head(1)
    assert bool(_series(first_rows, "ret_1d").isna().all())
    assert set(DAILY_WATCH20_FEATURES).issubset(actual.columns)
    assert "low_volatility_pct" in DAILY_WATCH20_FEATURES
    assert "low_resvol_pct" not in DAILY_WATCH20_FEATURES
    assert actual["low_volatility_pct"].equals(actual["low_resvol_pct"])
    assert set(DAILY_WATCH20_MARKET_SHADOW_FEATURES).isdisjoint(actual.columns)
    assert set(DAILY_WATCH20_MARKET_SHADOW_DIAGNOSTICS).isdisjoint(actual.columns)
    assert set(DAILY_WATCH20_MARKET_SHADOW_FEATURES).isdisjoint(DAILY_WATCH20_FEATURES)
    with pytest.raises(ValueError, match="must be a boolean"):
        DailyWatch20FeatureConfig(include_market_shadow_features=cast(bool, 1))


def test_market_shadow_feature_builder_is_not_called_without_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(_frame: pd.DataFrame) -> pd.DataFrame:
        raise AssertionError("research-only market shadow builder was called")

    monkeypatch.setattr(feature_module, "_add_market_shadow_features", unexpected)

    features = build_daily_watch20_feature_frame(_daily_panel())

    assert set(DAILY_WATCH20_MARKET_SHADOW_FEATURES).isdisjoint(features.columns)
    assert set(DAILY_WATCH20_MARKET_SHADOW_DIAGNOSTICS).isdisjoint(features.columns)


def test_market_shadow_beta_uses_aligned_ols_and_vol_regime_is_cross_sectional() -> None:
    symbols = ("000001.SZ", "000002.SZ", "600000.SH")
    daily = _daily_panel(n_dates=90, symbols=symbols)
    dates = pd.DatetimeIndex(_series(daily, "trade_date").drop_duplicates())
    market_returns = pd.Series(
        0.003 * np.sin(np.arange(len(dates)) / 4.0) + 0.002 * np.cos(np.arange(len(dates)) / 7.0),
        index=dates,
    )
    loadings = {"000001.SZ": 2.0, "000002.SZ": 1.0, "600000.SH": 0.0}
    for symbol, loading in loadings.items():
        mask = daily["symbol"].eq(symbol)
        returns = loading * market_returns.to_numpy(dtype=float)
        returns[0] = 0.0
        daily.loc[mask, "tr_close"] = 10.0 * np.cumprod(1.0 + returns)

    features = build_daily_watch20_feature_frame(
        daily,
        config=DailyWatch20FeatureConfig(include_market_shadow_features=True),
    )
    latest = features.loc[features["trade_date"].eq(dates[-1])].set_index("symbol")

    assert latest.loc["000001.SZ", "beta_60"] == pytest.approx(2.0, abs=1e-10)
    assert latest.loc["000002.SZ", "beta_60"] == pytest.approx(1.0, abs=1e-10)
    assert latest.loc["600000.SH", "beta_60"] == pytest.approx(0.0, abs=1e-10)
    raw_regime = -(latest["vol_20"] / latest["vol_60"])
    expected_rank = raw_regime.rank(pct=True)
    assert np.allclose(latest["vol_regime_20_60_pct"], expected_rank, equal_nan=True)
    assert latest["value_yield_pct"].equals(latest["value_yield"].rank(pct=True))


def test_style_beta_uses_consistent_ols_normalization() -> None:
    market = pd.Series([0.01, -0.02, 0.03, 0.015])
    returns = pd.DataFrame({"000001.SZ": 2.0 * market, "000002.SZ": -0.5 * market})

    beta = compute_beta_factor(returns, market, window=4, min_obs=4)

    assert beta.iloc[-1, 0] == pytest.approx(2.0)
    assert beta.iloc[-1, 1] == pytest.approx(-0.5)


def test_market_shadow_beta_does_not_treat_a_missing_session_as_one_day_return() -> None:
    symbols = ("000001.SZ", "000002.SZ", "600000.SH")
    daily = _daily_panel(n_dates=90, symbols=symbols)
    dates = pd.DatetimeIndex(_series(daily, "trade_date").drop_duplicates())
    market_returns = pd.Series(
        0.003 * np.sin(np.arange(len(dates)) / 5.0) + 0.002 * np.cos(np.arange(len(dates)) / 9.0),
        index=dates,
    )
    loadings = {"000001.SZ": 1.0, "000002.SZ": 0.0, "600000.SH": 2.0}
    for symbol, loading in loadings.items():
        mask = daily["symbol"].eq(symbol)
        returns = loading * market_returns.to_numpy(dtype=float)
        returns[0] = 0.0
        daily.loc[mask, "tr_close"] = 10.0 * np.cumprod(1.0 + returns)
    missing = daily["symbol"].eq("000001.SZ") & daily["trade_date"].eq(dates[50])
    daily = daily.loc[~missing]

    features = build_daily_watch20_feature_frame(
        daily,
        config=DailyWatch20FeatureConfig(include_market_shadow_features=True),
    )
    latest = _row(features, dates[-1], "000001.SZ")

    assert latest["beta_60"] == pytest.approx(1.0, abs=1e-10)


def test_shadow_returns_exclude_suspension_reopen_zero_amount_and_cross_gap() -> None:
    daily = _daily_panel(n_dates=10)
    dates = pd.DatetimeIndex(_series(daily, "trade_date").drop_duplicates())
    symbol = "000001.SZ"
    daily.loc[
        daily["symbol"].eq(symbol) & daily["trade_date"].eq(dates[2]),
        "is_suspended",
    ] = True
    daily.loc[
        daily["symbol"].eq(symbol) & daily["trade_date"].eq(dates[8]),
        "amount",
    ] = 0.0
    missing = daily["symbol"].eq(symbol) & daily["trade_date"].eq(dates[5])
    daily = daily.loc[~missing].reset_index(drop=True)

    returns = _adjacent_trade_day_returns(daily)

    def stock_return(trade_date: pd.Timestamp, stock: str = symbol) -> float:
        index = daily.index[daily["symbol"].eq(stock) & daily["trade_date"].eq(trade_date)]
        assert len(index) == 1
        return float(returns.loc[index[0]])

    assert np.isfinite(stock_return(dates[1]))
    for invalid_date in (dates[2], dates[3], dates[6], dates[8], dates[9]):
        assert np.isnan(stock_return(invalid_date))
    assert np.isfinite(stock_return(dates[2], "600000.SH"))


@pytest.mark.parametrize(("lag", "source_number", "target_number"), [(0, 4, 4), (1, 4, 5)])
def test_minute_features_carry_an_explicit_nonfuture_source_date(
    lag: int,
    source_number: int,
    target_number: int,
) -> None:
    daily = _daily_panel()
    dates = pd.Index(_series(daily, "trade_date").unique()).sort_values()
    minute = pd.DataFrame(
        {
            "source_trade_date": dates,
            "ts_code": "000001.SZ",
            "minute_realized_vol": np.arange(len(dates), dtype=float),
        }
    )

    features = build_daily_watch20_feature_frame(
        daily,
        minute,
        config=DailyWatch20FeatureConfig(minute_lag_trade_days=lag),
    )

    selected = _row(features, _timestamp(dates[target_number]), "000001.SZ")
    assert selected["minute_source_date"] == dates[source_number]
    assert selected["minute_realized_vol"] == float(source_number)
    assert selected["minute_source_date"] <= selected["trade_date"]
    assert bool(selected["minute_feature_available"])
    assert set(MINUTE_FEATURES).issubset(features.columns)


def test_daily_watch20_hermite_matches_style_replica_reference() -> None:
    daily = _daily_panel(n_dates=140)
    dates = pd.Index(_series(daily, "trade_date").unique()).sort_values()
    symbols = tuple(_series(daily, "symbol").drop_duplicates())
    rng = np.random.default_rng(19)
    panel = pd.DataFrame(
        rng.lognormal(mean=0.0, sigma=0.35, size=(len(dates), len(symbols))),
        index=pd.Index(dates),
        columns=pd.Index(symbols),
    )
    stacked = cast(
        pd.Series,
        panel.rename_axis(index="trade_date", columns="symbol").stack(),
    )
    minute = stacked.rename("minute_volume_activity").reset_index()

    features = build_daily_watch20_feature_frame(daily, minute)
    actual = features.pivot(index="trade_date", columns="symbol", values="hermite_stability")
    expected = compute_hermite_stability_factor(panel, ddof=1)

    assert expected is not None
    assert_frame_equal(actual, expected, check_names=False, atol=1e-12, rtol=1e-12)


def test_next_open_label_uses_t_plus_one_to_t_plus_six_and_eligible_universe() -> None:
    symbols = ("000001.SZ", "600000.SH", "830001.BJ")
    daily = _daily_panel(symbols=symbols)
    dates = pd.Index(_series(daily, "trade_date").unique()).sort_values()

    features = build_daily_watch20_feature_frame(daily)

    signal_number = 25
    signal_date = _timestamp(dates[signal_number])
    a_share = _row(features, signal_date, "000001.SZ")
    sh_share = _row(features, signal_date, "600000.SH")
    bse_share = _row(features, signal_date, "830001.BJ")
    expected_return = (10.0 + signal_number + 6) / (10.0 + signal_number + 1) - 1.0
    assert a_share["forward_return_5d"] == pytest.approx(expected_return)
    expected_1d = (10.0 + signal_number + 2) / (10.0 + signal_number + 1) - 1.0
    expected_3d = (10.0 + signal_number + 4) / (10.0 + signal_number + 1) - 1.0
    assert a_share["forward_return_1d"] == pytest.approx(expected_1d)
    assert a_share["forward_return_3d"] == pytest.approx(expected_3d)
    assert a_share["liquidity_pct"] == 0.5
    assert a_share["forward_label_start_date"] == dates[signal_number + 1]
    assert a_share["forward_label_end_date"] == dates[signal_number + 6]
    assert a_share["forward_rank_5d"] == 1.0
    assert a_share["forward_rank_blended"] == 1.0
    assert sh_share["forward_rank_5d"] == 0.5
    assert not bool(bse_share["hard_eligible"])
    assert _is_missing(bse_share["liquidity_pct"])
    assert _is_missing(bse_share["forward_rank_5d"])
    assert _is_missing(bse_share["forward_rank_blended"])
    trailing = features.loc[features["symbol"].eq("000001.SZ")].tail(6)
    assert bool(_series(trailing, "forward_return_5d").isna().all())
    assert bool(_series(trailing, "forward_rank_blended").isna().all())


@pytest.mark.parametrize(
    ("side", "offset", "return_column"),
    [
        ("buy", 1, "forward_return_1d"),
        ("sell", 2, "forward_return_1d"),
        ("sell", 4, "forward_return_3d"),
        ("sell", 6, "forward_return_5d"),
    ],
)
def test_next_open_labels_reject_limit_locked_entry_and_exit(
    side: str,
    offset: int,
    return_column: str,
) -> None:
    daily = _daily_panel()
    dates = pd.Index(_series(daily, "trade_date").unique()).sort_values()
    signal_number = 25
    locked = daily["trade_date"].eq(dates[signal_number + offset]) & daily["symbol"].eq("000001.SZ")
    limit_column = "up_limit" if side == "buy" else "down_limit"
    daily.loc[locked, "open"] = daily.loc[locked, limit_column]

    features = build_daily_watch20_feature_frame(daily)

    signal = _row(features, _timestamp(dates[signal_number]), "000001.SZ")
    assert _is_missing(signal[return_column])
    assert _is_missing(signal[return_column.replace("return", "rank")])
    assert _is_missing(signal["forward_return_blended"])
    assert _is_missing(signal["forward_rank_blended"])


def test_next_open_label_ignores_close_limit_flags_and_fails_closed_on_missing_raw_limit() -> None:
    daily = _daily_panel()
    dates = pd.Index(_series(daily, "trade_date").unique()).sort_values()
    signal_number = 25
    entry = daily["trade_date"].eq(dates[signal_number + 1]) & daily["symbol"].eq("000001.SZ")
    daily.loc[entry, "is_limit_up"] = True

    close_flag_features = build_daily_watch20_feature_frame(daily)
    close_flag_signal = _row(
        close_flag_features,
        _timestamp(dates[signal_number]),
        "000001.SZ",
    )
    assert not _is_missing(close_flag_signal["forward_return_1d"])

    daily.loc[entry, "up_limit"] = np.nan
    missing_limit_features = build_daily_watch20_feature_frame(daily)
    missing_limit_signal = _row(
        missing_limit_features,
        _timestamp(dates[signal_number]),
        "000001.SZ",
    )
    assert _is_missing(missing_limit_signal["forward_return_1d"])
    assert _is_missing(missing_limit_signal["forward_return_blended"])


def test_legacy_five_day_label_mode_keeps_the_single_horizon_contract() -> None:
    features = build_daily_watch20_feature_frame(
        _daily_panel(),
        config=DailyWatch20FeatureConfig(
            label_horizon_weights=LEGACY_FIVE_DAY_LABEL_HORIZON_WEIGHTS
        ),
    )

    assert "forward_rank_5d" in features
    assert "forward_rank_blended" not in features
    assert "forward_return_1d" not in features


def test_label_does_not_stretch_across_a_missing_symbol_date() -> None:
    daily = _daily_panel()
    dates = pd.Index(_series(daily, "trade_date").unique()).sort_values()
    signal_number = 25
    missing_entry = daily["trade_date"].eq(dates[signal_number + 1]) & daily["symbol"].eq(
        "000001.SZ"
    )

    features = build_daily_watch20_feature_frame(daily.loc[~missing_entry])

    signal = _row(features, _timestamp(dates[signal_number]), "000001.SZ")
    assert signal["forward_label_start_date"] == dates[signal_number + 1]
    assert signal["forward_label_end_date"] == dates[signal_number + 6]
    assert _is_missing(signal["forward_return_5d"])
    assert _is_missing(signal["forward_rank_5d"])


def test_hard_eligibility_fails_closed_for_status_market_and_data_gaps() -> None:
    symbols = (
        "000001.SZ",
        "600001.SH",
        "600002.SH",
        "600003.SH",
        "600004.SH",
        "600005.SH",
        "830001.BJ",
        "600006.SH",
    )
    daily = _daily_panel(symbols=symbols)
    audit_date = pd.Index(daily["trade_date"].unique()).sort_values()[25]
    daily["is_st"] = daily["is_st"].astype(object)
    daily.loc[daily["symbol"].eq("600006.SH"), "amount"] = 1.0
    daily.loc[daily["trade_date"].eq(audit_date) & daily["symbol"].eq("600001.SH"), "is_st"] = True
    daily.loc[
        daily["trade_date"].eq(audit_date) & daily["symbol"].eq("600002.SH"),
        "is_suspended",
    ] = True
    daily.loc[daily["trade_date"].eq(audit_date) & daily["symbol"].eq("600003.SH"), "is_st"] = ""
    daily.loc[
        daily["trade_date"].eq(audit_date) & daily["symbol"].eq("600004.SH"), "listed_days"
    ] = 10
    daily.loc[daily["trade_date"].eq(audit_date) & daily["symbol"].eq("600005.SH"), "amount"] = 0

    features = build_daily_watch20_feature_frame(daily)
    cross_section = features.loc[features["trade_date"].eq(audit_date)].set_index("symbol")

    assert bool(cross_section.loc["000001.SZ", "hard_eligible"])
    excluded = {
        "600001.SH",
        "600002.SH",
        "600003.SH",
        "600004.SH",
        "600005.SH",
        "600006.SH",
        "830001.BJ",
    }
    assert not bool(cast(pd.Series, cross_section.loc[list(excluded), "hard_eligible"]).any())
    assert bool(cast(pd.Series, cross_section.loc[list(excluded), "forward_rank_5d"]).isna().all())
