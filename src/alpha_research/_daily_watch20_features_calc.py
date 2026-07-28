"""DailyWatch20 point-in-time feature transforms.

Private helpers that build the daily / liquidity / style / market / minute /
Hermite / eligibility features and next-open labels. Split out of the
historical single-file :mod:`alpha_research.daily_watch20_features`
implementation to keep individual files smaller while preserving the exact
public/private symbol surface. The constants and config live in
:mod:`alpha_research._daily_watch20_features_config`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from ._daily_watch20_features_config import (
    _BETA_MIN_OBS,
    _BETA_WINDOW,
    MINUTE_FEATURES,
    DailyWatch20FeatureConfig,
    label_columns_for_horizon_weights,
    normalize_label_horizon_weights,
)


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"DailyWatch20 daily input is missing columns: {missing}")


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Narrow pandas' overloaded column access to this module's Series contract."""

    return cast(pd.Series, frame[column])


def _numeric_series(values: pd.Series) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(values, errors="coerce"))


def _target_market(symbols: pd.Series) -> pd.Series:
    return symbols.astype("string").str.fullmatch(r"\d{6}\.(?:SH|SZ)", na=False)


def _rolling(
    frame: pd.DataFrame,
    column: str,
    window: int,
    *,
    operation: str,
    min_periods: int,
) -> pd.Series:
    grouped = frame.groupby("symbol", sort=False)[column]
    roller = grouped.rolling(window, min_periods=min_periods)
    result = getattr(roller, operation)()
    return result.reset_index(level=0, drop=True).reindex(frame.index)


def _prepare_daily_input(daily: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        daily,
        {
            "trade_date",
            "symbol",
            "adj_open",
            "open",
            "up_limit",
            "down_limit",
            "tr_close",
            "high",
            "low",
            "close",
            "amount",
            "turnover_rate",
            "total_mv",
            "pb",
            "pe_ttm",
            "listed_days",
            "is_st",
            "is_suspended",
            "is_limit_up",
            "is_limit_down",
        },
    )
    out = daily.copy()
    dates = pd.to_datetime(_series(out, "trade_date"), errors="coerce")
    if dates.isna().any():
        raise ValueError("DailyWatch20 daily input contains invalid trade_date values")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    out["trade_date"] = dates.dt.normalize()
    symbols = _series(out, "symbol").astype("string").str.strip().str.upper()
    if symbols.isna().any() or symbols.eq("").any():
        raise ValueError("DailyWatch20 daily input contains empty symbols")
    out["symbol"] = symbols.astype(str)
    if out.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("DailyWatch20 daily input contains duplicate stock-date rows")
    numeric = [
        "adj_open",
        "open",
        "up_limit",
        "down_limit",
        "tr_close",
        "high",
        "low",
        "close",
        "amount",
        "turnover_rate",
        "total_mv",
        "pb",
        "pe_ttm",
        "listed_days",
        "volume_ratio",
    ]
    for column in numeric:
        if column in out.columns:
            out[column] = _numeric_series(_series(out, column)).replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["symbol", "trade_date"], kind="mergesort").reset_index(drop=True)


def _add_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    grouped = out.groupby("symbol", sort=False)
    out["ret_1d"] = grouped["tr_close"].pct_change(fill_method=None)
    for window in (5, 20, 60, 120):
        out[f"mom_{window}"] = grouped["tr_close"].pct_change(window, fill_method=None)
    for window, min_periods in ((5, 3), (20, 10), (60, 30)):
        out[f"vol_{window}"] = _rolling(
            out,
            "ret_1d",
            window,
            operation="std",
            min_periods=min_periods,
        )
    out["_downside_sq"] = out["ret_1d"].clip(upper=0).pow(2)
    downside_mean = _rolling(
        out,
        "_downside_sq",
        20,
        operation="mean",
        min_periods=10,
    )
    out["downside_vol_20"] = np.sqrt(downside_mean)
    out["range_pct"] = (out["high"] - out["low"]) / out["close"].where(out["close"] > 0)
    price_range = out["high"] - out["low"]
    out["close_location"] = (out["close"] - out["low"]) / price_range.where(price_range > 0)
    return out.drop(columns="_downside_sq")


def _add_liquidity_and_style_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_log_amount"] = np.log1p(out["amount"].clip(lower=0))
    out["amount_log_20"] = _rolling(
        out,
        "_log_amount",
        20,
        operation="mean",
        min_periods=10,
    )
    out["turnover_20"] = _rolling(
        out,
        "turnover_rate",
        20,
        operation="mean",
        min_periods=10,
    )
    out["size_log"] = np.log1p(out["total_mv"].where(out["total_mv"] > 0))
    out["value_yield"] = 1.0 / out["pb"].where(out["pb"] > 0)
    # This is a valuation yield only; it must not be presented as earnings quality.
    out["earnings_yield"] = 1.0 / out["pe_ttm"].where(out["pe_ttm"] > 0)
    out["vol_convergence"] = -(out["vol_5"] / out["vol_60"].where(out["vol_60"] > 0))
    rank_inputs = {
        "size_pct": "size_log",
        "liquidity_pct": "amount_log_20",
        "low_volatility_pct": "vol_20",
        "vol_convergence_pct": "vol_convergence",
        "mom20_pct": "mom_20",
        "mom120_pct": "mom_120",
    }
    target_market = _target_market(_series(out, "symbol"))
    for output, source in rank_inputs.items():
        source_values = _series(out, source)
        values = -source_values if output == "low_volatility_pct" else source_values
        out[output] = (
            values.where(target_market)
            .groupby(_series(out, "trade_date"), sort=False)
            .rank(pct=True)
        )
    # Compatibility-only alias. New model contracts must use low_volatility_pct.
    out["low_resvol_pct"] = out["low_volatility_pct"]
    return out.drop(columns="_log_amount")


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(ends - window, 0)
    return cumulative[ends] - cumulative[starts]


def _rolling_ols_slope(
    dependent: np.ndarray,
    independent: np.ndarray,
    *,
    window: int,
    min_obs: int,
) -> np.ndarray:
    """Return an aligned rolling OLS slope from pairwise finite observations."""

    valid = np.isfinite(dependent) & np.isfinite(independent)
    y = np.where(valid, dependent, 0.0)
    x = np.where(valid, independent, 0.0)
    count = _rolling_sum(valid.astype(float), window)
    sum_x = _rolling_sum(x, window)
    sum_y = _rolling_sum(y, window)
    sum_xx = _rolling_sum(x * x, window)
    sum_xy = _rolling_sum(x * y, window)
    safe_count = np.where(count > 0, count, 1.0)
    denominator = sum_xx - sum_x * sum_x / safe_count
    numerator = sum_xy - sum_x * sum_y / safe_count
    slope = np.full(len(dependent), np.nan, dtype=float)
    usable = (count >= min_obs) & (denominator > np.finfo(float).eps)
    slope[usable] = numerator[usable] / denominator[usable]
    return slope


def _adjacent_trade_day_returns(frame: pd.DataFrame) -> pd.Series:
    """Return close changes that represent one observable, tradable session."""

    trade_dates = pd.DatetimeIndex(_series(frame, "trade_date").unique()).sort_values()
    date_position = pd.Series(np.arange(len(trade_dates)), index=trade_dates)
    current_position = _series(frame, "trade_date").map(date_position)
    grouped = frame.groupby("symbol", sort=False)
    previous_date = grouped["trade_date"].shift(1)
    previous_position = previous_date.map(date_position)
    adjacent = current_position.sub(previous_position).eq(1)
    current_tradable = _known_false_flag(_series(frame, "is_suspended")) & _series(
        frame, "amount"
    ).gt(0)
    previous_tradable = grouped["is_suspended"].shift(1).pipe(_known_false_flag) & grouped[
        "amount"
    ].shift(1).gt(0)
    returns = grouped["tr_close"].pct_change(fill_method=None)
    valid_prices = _series(frame, "tr_close").gt(0) & grouped["tr_close"].shift(1).gt(0)
    valid = adjacent & current_tradable & previous_tradable & valid_prices & np.isfinite(returns)
    return cast(pd.Series, returns.where(valid))


def _equal_weight_market_return(frame: pd.DataFrame, returns: pd.Series) -> pd.Series:
    target = frame.loc[_target_market(_series(frame, "symbol"))]
    return cast(
        pd.Series,
        returns.loc[target.index].groupby(_series(target, "trade_date"), sort=True).mean(),
    )


def _add_market_shadow_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach research-only beta and medium/long volatility-regime candidates."""

    out = frame.copy()
    aligned_returns = _adjacent_trade_day_returns(out)
    market_return = _equal_weight_market_return(out, aligned_returns)
    market_dates = pd.DatetimeIndex(market_return.index)
    market_values = market_return.to_numpy(dtype=float)
    target_mask = _target_market(_series(out, "symbol"))
    out["value_yield_pct"] = (
        _series(out, "value_yield")
        .where(target_mask)
        .groupby(_series(out, "trade_date"), sort=False)
        .rank(pct=True)
    )
    beta = pd.Series(np.nan, index=out.index, dtype=float)
    vol_regime = pd.Series(np.nan, index=out.index, dtype=float)
    target = out.loc[target_mask]
    for _symbol, group in target.groupby("symbol", sort=False):
        returns_by_date = pd.Series(
            aligned_returns.loc[group.index].to_numpy(dtype=float),
            index=pd.DatetimeIndex(_series(group, "trade_date")),
        ).reindex(market_dates)
        slopes = _rolling_ols_slope(
            returns_by_date.to_numpy(dtype=float),
            market_values,
            window=_BETA_WINDOW,
            min_obs=_BETA_MIN_OBS,
        )
        slope_by_date = pd.Series(slopes, index=market_dates)
        beta.loc[group.index] = _series(group, "trade_date").map(slope_by_date).to_numpy()
        vol_20 = returns_by_date.rolling(20, min_periods=10).std()
        vol_60 = returns_by_date.rolling(60, min_periods=30).std()
        ratio_by_date = -(vol_20 / vol_60.where(vol_60 > 0))
        vol_regime.loc[group.index] = _series(group, "trade_date").map(ratio_by_date).to_numpy()
    out["beta_60"] = beta
    out["vol_regime_20_60_pct"] = (
        vol_regime.where(target_mask).groupby(_series(out, "trade_date"), sort=False).rank(pct=True)
    )
    return out


def _add_market_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    target = frame.loc[_target_market(_series(frame, "symbol"))]
    daily_market = (
        target.groupby("trade_date", sort=True)
        .agg(
            market_ret_1d=("ret_1d", "mean"),
            market_breadth=("ret_1d", lambda values: float((values > 0).mean())),
        )
        .sort_index()
    )
    daily_market["market_mom_20"] = daily_market["market_ret_1d"].rolling(20, 10).mean()
    daily_market["market_vol_20"] = daily_market["market_ret_1d"].rolling(20, 10).std()
    daily_market["breadth_20"] = daily_market["market_breadth"].rolling(20, 10).mean()
    regime = daily_market[["market_mom_20", "market_vol_20", "breadth_20"]].reset_index()
    return frame.merge(regime, on="trade_date", how="left", validate="many_to_one")


def _lag_minute_features(
    minute_daily: pd.DataFrame,
    trade_dates: pd.Index,
    *,
    lag_trade_days: int,
) -> pd.DataFrame:
    minute = minute_daily.copy()
    rename: dict[str, str] = {}
    if "symbol" not in minute.columns and "ts_code" in minute.columns:
        rename["ts_code"] = "symbol"
    if "trade_date" not in minute.columns and "source_trade_date" in minute.columns:
        rename["source_trade_date"] = "trade_date"
    minute = minute.rename(columns=rename)
    _require_columns(minute, {"trade_date", "symbol"})
    minute_dates = pd.to_datetime(_series(minute, "trade_date"), errors="coerce")
    if minute_dates.dt.tz is not None:
        minute_dates = minute_dates.dt.tz_localize(None)
    minute_symbols = _series(minute, "symbol").astype("string").str.strip().str.upper()
    valid_keys = minute_dates.notna() & minute_symbols.notna() & minute_symbols.ne("")
    if not valid_keys.all():
        raise ValueError("DailyWatch20 minute input contains invalid stock-date keys")
    minute["trade_date"] = minute_dates.dt.normalize()
    minute["symbol"] = minute_symbols.astype(str)
    if minute.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("DailyWatch20 minute input contains duplicate stock-date rows")
    for feature in MINUTE_FEATURES:
        if feature in minute.columns:
            minute[feature] = _numeric_series(_series(minute, feature)).replace(
                [np.inf, -np.inf], np.nan
            )
    ordered_dates = pd.Index(pd.to_datetime(trade_dates).unique()).sort_values()
    minute["minute_source_date"] = minute["trade_date"]
    if lag_trade_days:
        effective = cast(
            pd.Series,
            pd.Series(ordered_dates, index=ordered_dates).shift(-lag_trade_days),
        )
        minute["trade_date"] = _series(minute, "trade_date").map(effective)
        minute = minute.dropna(subset=["trade_date"])
    return minute


def _join_minute_features(
    frame: pd.DataFrame,
    minute_daily: pd.DataFrame | None,
    *,
    lag_trade_days: int,
) -> pd.DataFrame:
    out = frame.copy()
    if minute_daily is None or minute_daily.empty:
        for feature in MINUTE_FEATURES:
            out[feature] = np.nan
        out["minute_source_date"] = pd.NaT
        out["minute_feature_available"] = False
        return out
    minute = _lag_minute_features(
        minute_daily,
        pd.Index(out["trade_date"]),
        lag_trade_days=lag_trade_days,
    )
    available = [feature for feature in MINUTE_FEATURES if feature in minute.columns]
    keep = ["trade_date", "symbol", *available]
    if "minute_source_date" in minute.columns:
        keep.append("minute_source_date")
    out = out.merge(minute[keep], on=["trade_date", "symbol"], how="left", validate="one_to_one")
    for feature in MINUTE_FEATURES:
        if feature not in out.columns:
            out[feature] = np.nan
    out["minute_feature_available"] = out[list(MINUTE_FEATURES)].notna().any(axis=1)
    return out


def _future_date(frame: pd.DataFrame, offset: int) -> pd.Series:
    trade_dates = pd.Index(_series(frame, "trade_date").unique()).sort_values()
    shifted = cast(pd.Series, pd.Series(trade_dates, index=trade_dates).shift(-offset))
    return _series(frame, "trade_date").map(shifted)


def _lookup_on_dates(
    frame: pd.DataFrame,
    values: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    key_frame = cast(pd.DataFrame, frame.loc[:, ["symbol", "trade_date"]])
    index = pd.MultiIndex.from_frame(key_frame)
    keys = pd.MultiIndex.from_arrays([_series(frame, "symbol"), dates], names=index.names)
    lookup = pd.Series(values.to_numpy(), index=index)
    return pd.Series(lookup.reindex(keys).to_numpy(), index=frame.index)


def _weighted_complete_row_sum(
    frame: pd.DataFrame,
    columns_and_weights: Sequence[tuple[str, float]],
) -> pd.Series:
    weighted = pd.concat(
        [_series(frame, column).mul(weight) for column, weight in columns_and_weights],
        axis=1,
    )
    return weighted.sum(axis=1, min_count=len(columns_and_weights))


def _open_is_away_from_limit(
    open_price: pd.Series,
    limit_price: pd.Series,
    *,
    side: str,
) -> pd.Series:
    raw_open = _numeric_series(open_price)
    raw_limit = _numeric_series(limit_price)
    finite = (
        raw_open.notna()
        & raw_limit.notna()
        & np.isfinite(raw_open)
        & np.isfinite(raw_limit)
        & raw_open.gt(0)
        & raw_limit.gt(0)
    )
    tolerance = np.maximum(raw_limit.abs() * 1e-8, 1e-8)
    if side == "buy":
        away = raw_open < raw_limit - tolerance
    elif side == "sell":
        away = raw_open > raw_limit + tolerance
    else:
        raise ValueError("side must be 'buy' or 'sell'")
    return finite & away


def _add_next_open_labels(
    frame: pd.DataFrame,
    *,
    horizon_weights: Mapping[int, float] | Sequence[tuple[int, float]],
) -> pd.DataFrame:
    out = frame.sort_values(["symbol", "trade_date"], kind="mergesort").copy()
    normalized_weights = normalize_label_horizon_weights(horizon_weights)
    entry_date = _future_date(out, 1)
    entry = _numeric_series(_lookup_on_dates(out, _series(out, "adj_open"), entry_date))
    tradable = _known_false_flag(_series(out, "is_suspended"))
    entry_not_limit_up = _lookup_on_dates(
        out,
        _open_is_away_from_limit(
            _series(out, "open"),
            _series(out, "up_limit"),
            side="buy",
        ),
        entry_date,
    ).eq(True)
    entry_tradable = _lookup_on_dates(out, tradable, entry_date).eq(True) & entry_not_limit_up
    out["forward_label_start_date"] = entry_date
    return_parts: list[tuple[str, float]] = []
    rank_parts: list[tuple[str, float]] = []
    for horizon, weight in normalized_weights:
        exit_date = _future_date(out, horizon + 1)
        exit_price = _numeric_series(_lookup_on_dates(out, _series(out, "adj_open"), exit_date))
        exit_not_limit_down = _lookup_on_dates(
            out,
            _open_is_away_from_limit(
                _series(out, "open"),
                _series(out, "down_limit"),
                side="sell",
            ),
            exit_date,
        ).eq(True)
        exit_tradable = _lookup_on_dates(out, tradable, exit_date).eq(True) & exit_not_limit_down
        valid = entry.gt(0) & exit_price.gt(0) & entry_tradable & exit_tradable
        return_col = f"forward_return_{horizon}d"
        rank_col = f"forward_rank_{horizon}d"
        out[return_col] = (exit_price / entry - 1.0).where(valid)
        eligible_return = _series(out, return_col).where(_series(out, "hard_eligible"))
        out[rank_col] = eligible_return.groupby(_series(out, "trade_date"), sort=False).rank(
            method="average",
            pct=True,
        )
        return_parts.append((return_col, weight))
        rank_parts.append((rank_col, weight))
        if horizon == normalized_weights[-1][0]:
            out["forward_label_end_date"] = exit_date
    target_rank_col, target_return_col = label_columns_for_horizon_weights(normalized_weights)
    if len(normalized_weights) > 1:
        out[target_return_col] = _weighted_complete_row_sum(out, return_parts)
        out[target_rank_col] = _weighted_complete_row_sum(out, rank_parts)
    return out


def _known_false_flag(values: pd.Series) -> pd.Series:
    numeric = _numeric_series(values)
    text = values.astype("string").str.strip().str.lower()
    return numeric.eq(0).fillna(False) | text.isin({"false", "f", "no", "n"})


_HERMITE_WINDOW: int = 60
_HERMITE_MIN_PERIODS: int = 36


def _add_hermite_stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-day stability of the daily minute-volume-activity series.

    Each input is one stock-day's already-aggregated ``minute_volume_activity``.
    The two 60-trade-day rolling transforms describe the time-series shape of
    those daily values; this is not a Gaussianity test on one day's minute bars.

    ``closeness = -log(1 + h3² + h4²)``; higher is more stable over time.

    This production contract uses pandas' sample rolling standard deviation
    (``ddof=1``). StyleReplica defaults to ``ddof=0``; pass ``ddof=1`` there for
    a bit-for-bit long/wide comparison. Changing this requires a feature-policy
    version bump because Hermite also participates in the B-sleeve guard.
    """
    va_col = "minute_volume_activity"
    if va_col not in frame.columns or bool(_series(frame, va_col).isna().all()):
        frame["hermite_stability"] = np.nan
        return frame

    out = frame.sort_values(["symbol", "trade_date"], kind="mergesort").copy()

    roll_mean = _rolling(
        out, va_col, _HERMITE_WINDOW, operation="mean", min_periods=_HERMITE_MIN_PERIODS
    )
    roll_std = _rolling(
        out, va_col, _HERMITE_WINDOW, operation="std", min_periods=_HERMITE_MIN_PERIODS
    )
    roll_std = roll_std.where(roll_std > 1e-8, np.nan)

    z = ((_series(out, va_col) - roll_mean) / roll_std).clip(-8.0, 8.0)
    z2 = z * z

    # Hermite polynomials h3 (skewness proxy) and h4 (kurtosis proxy)
    h3_raw = (z * z2 - 3.0 * z) / np.sqrt(6.0)
    h4_raw = (z2 * z2 - 6.0 * z2 + 3.0) / np.sqrt(24.0)

    out["_h3_raw"] = h3_raw
    out["_h4_raw"] = h4_raw
    h3_roll = _rolling(
        out, "_h3_raw", _HERMITE_WINDOW, operation="mean", min_periods=_HERMITE_MIN_PERIODS
    )
    h4_roll = _rolling(
        out, "_h4_raw", _HERMITE_WINDOW, operation="mean", min_periods=_HERMITE_MIN_PERIODS
    )

    energy = h3_roll.pow(2) + h4_roll.pow(2)
    out["hermite_stability"] = -np.log1p(energy)

    if "minute_feature_available" in out.columns:
        out["minute_feature_available"] = (
            _series(out, "minute_feature_available") | _series(out, "hermite_stability").notna()
        )

    return out.drop(columns=["_h3_raw", "_h4_raw"])


def _add_eligibility(frame: pd.DataFrame, config: DailyWatch20FeatureConfig) -> pd.DataFrame:
    out = frame.copy()
    market_ok = _target_market(_series(out, "symbol"))
    status_ok = _known_false_flag(_series(out, "is_st")) & _known_false_flag(
        _series(out, "is_suspended")
    )
    core_columns = ["tr_close", "mom_20", "vol_20", "amount_log_20", "size_log"]
    core_values = out[core_columns].to_numpy(dtype=float)
    core_ok = pd.Series(
        np.isfinite(core_values).all(axis=1),
        index=out.index,
        dtype=bool,
    )
    liquid = _series(out, "liquidity_pct").ge(config.liquidity_floor_quantile).fillna(False)
    out["hard_eligible"] = (
        market_ok
        & status_ok
        & _series(out, "listed_days").ge(config.min_listed_days)
        & _series(out, "tr_close").gt(0)
        & _series(out, "amount").gt(0)
        & core_ok
        & liquid
    ).fillna(False)
    return out
