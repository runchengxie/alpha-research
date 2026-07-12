"""Point-in-time daily and lagged intraday features for DailyWatch20."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_LABEL_HORIZON_WEIGHTS = ((1, 0.20), (3, 0.30), (5, 0.50))
LEGACY_FIVE_DAY_LABEL_HORIZON_WEIGHTS = ((5, 1.0),)
BLENDED_FORWARD_RANK_COL = "forward_rank_blended"
BLENDED_FORWARD_RETURN_COL = "forward_return_blended"

DAILY_WATCH20_FEATURES = (
    "ret_1d",
    "mom_5",
    "mom_20",
    "mom_60",
    "mom_120",
    "vol_5",
    "vol_20",
    "vol_60",
    "downside_vol_20",
    "amount_log_20",
    "turnover_20",
    "size_pct",
    "liquidity_pct",
    "low_volatility_pct",
    "vol_convergence_pct",
    "mom20_pct",
    "mom120_pct",
    "value_yield",
    "earnings_yield",
    "range_pct",
    "close_location",
    "market_mom_20",
    "market_vol_20",
    "breadth_20",
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
    "hermite_stability",
)

_MINUTE_PREFIX = "minute_"
MINUTE_ORIGIN_FEATURES = tuple(
    name for name in DAILY_WATCH20_FEATURES if name.startswith(_MINUTE_PREFIX)
)
DERIVED_MINUTE_FEATURES = ("hermite_stability",)
MINUTE_FEATURES = MINUTE_ORIGIN_FEATURES + DERIVED_MINUTE_FEATURES


@dataclass(frozen=True)
class DailyWatch20FeatureConfig:
    """Feature and label timing for a close-to-next-open daily watchlist."""

    forward_days: int = 5
    label_horizon_weights: Mapping[int, float] | Sequence[tuple[int, float]] = (
        DEFAULT_LABEL_HORIZON_WEIGHTS
    )
    minute_lag_trade_days: int = 1
    min_listed_days: int = 60
    liquidity_floor_quantile: float = 0.20

    def __post_init__(self) -> None:
        horizon_weights = normalize_label_horizon_weights(self.label_horizon_weights)
        if int(self.forward_days) != max(horizon for horizon, _weight in horizon_weights):
            raise ValueError("forward_days must equal the longest configured label horizon")
        try:
            minute_lag = int(self.minute_lag_trade_days)
            min_listed = int(self.min_listed_days)
        except (TypeError, ValueError) as exc:
            raise ValueError("trade-day settings must be integers") from exc
        if minute_lag != self.minute_lag_trade_days or minute_lag < 0:
            raise ValueError("minute_lag_trade_days must be a non-negative integer")
        if min_listed != self.min_listed_days or min_listed < 0:
            raise ValueError("min_listed_days must be a non-negative integer")
        if not 0 <= self.liquidity_floor_quantile < 1:
            raise ValueError("liquidity_floor_quantile must be in [0, 1)")
        object.__setattr__(self, "minute_lag_trade_days", minute_lag)
        object.__setattr__(self, "min_listed_days", min_listed)
        object.__setattr__(self, "forward_days", int(self.forward_days))
        object.__setattr__(self, "label_horizon_weights", horizon_weights)

    @property
    def label_col(self) -> str:
        """Training target column implied by the configured horizon mix."""

        return label_columns_for_horizon_weights(self.label_horizon_weights)[0]

    @property
    def forward_return_col(self) -> str:
        """Forward-return column implied by the configured horizon mix."""

        return label_columns_for_horizon_weights(self.label_horizon_weights)[1]


def normalize_label_horizon_weights(
    weights: Mapping[int, float] | Sequence[tuple[int, float]],
) -> tuple[tuple[int, float], ...]:
    """Validate, sort, and normalize positive trading-day horizon weights."""

    items = tuple(weights.items()) if isinstance(weights, Mapping) else tuple(weights)
    if not items:
        raise ValueError("label_horizon_weights must not be empty")
    normalized: list[tuple[int, float]] = []
    seen: set[int] = set()
    for raw_horizon, raw_weight in items:
        try:
            horizon = int(raw_horizon)
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("label horizons and weights must be numeric") from exc
        if horizon != raw_horizon or horizon <= 0:
            raise ValueError("label horizons must be positive integers")
        if horizon in seen:
            raise ValueError("label horizons must be unique")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("label horizon weights must be finite and positive")
        seen.add(horizon)
        normalized.append((horizon, weight))
    total = sum(weight for _horizon, weight in normalized)
    return tuple((horizon, weight / total) for horizon, weight in sorted(normalized))


def label_columns_for_horizon_weights(
    weights: Mapping[int, float] | Sequence[tuple[int, float]],
) -> tuple[str, str]:
    """Return canonical rank and return target columns for a horizon mix."""

    normalized = normalize_label_horizon_weights(weights)
    if len(normalized) == 1:
        horizon = normalized[0][0]
        return f"forward_rank_{horizon}d", f"forward_return_{horizon}d"
    return BLENDED_FORWARD_RANK_COL, BLENDED_FORWARD_RETURN_COL


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"DailyWatch20 daily input is missing columns: {missing}")


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
        },
    )
    out = daily.copy()
    dates = pd.to_datetime(out["trade_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("DailyWatch20 daily input contains invalid trade_date values")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    out["trade_date"] = dates.dt.normalize()
    symbols = out["symbol"].astype("string").str.strip().str.upper()
    if symbols.isna().any() or symbols.eq("").any():
        raise ValueError("DailyWatch20 daily input contains empty symbols")
    out["symbol"] = symbols.astype(str)
    if out.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("DailyWatch20 daily input contains duplicate stock-date rows")
    numeric = [
        "adj_open",
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
            out[column] = pd.to_numeric(out[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
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
    target_market = _target_market(out["symbol"])
    for output, source in rank_inputs.items():
        values = -out[source] if output == "low_volatility_pct" else out[source]
        out[output] = (
            values.where(target_market).groupby(out["trade_date"], sort=False).rank(pct=True)
        )
    # Compatibility-only alias. New model contracts must use low_volatility_pct.
    out["low_resvol_pct"] = out["low_volatility_pct"]
    return out.drop(columns="_log_amount")


def _add_market_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    target = frame.loc[_target_market(frame["symbol"])]
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
    minute_dates = pd.to_datetime(minute["trade_date"], errors="coerce")
    if minute_dates.dt.tz is not None:
        minute_dates = minute_dates.dt.tz_localize(None)
    minute_symbols = minute["symbol"].astype("string").str.strip().str.upper()
    valid_keys = minute_dates.notna() & minute_symbols.notna() & minute_symbols.ne("")
    if not valid_keys.all():
        raise ValueError("DailyWatch20 minute input contains invalid stock-date keys")
    minute["trade_date"] = minute_dates.dt.normalize()
    minute["symbol"] = minute_symbols.astype(str)
    if minute.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("DailyWatch20 minute input contains duplicate stock-date rows")
    for feature in MINUTE_FEATURES:
        if feature in minute.columns:
            minute[feature] = pd.to_numeric(minute[feature], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    ordered_dates = pd.Index(pd.to_datetime(trade_dates).unique()).sort_values()
    minute["minute_source_date"] = minute["trade_date"]
    if lag_trade_days:
        effective = pd.Series(ordered_dates, index=ordered_dates).shift(-lag_trade_days)
        minute["trade_date"] = minute["trade_date"].map(effective)
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
    trade_dates = pd.Index(frame["trade_date"].unique()).sort_values()
    shifted = pd.Series(trade_dates, index=trade_dates).shift(-offset)
    return frame["trade_date"].map(shifted)


def _lookup_on_dates(
    frame: pd.DataFrame,
    values: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    index = pd.MultiIndex.from_frame(frame[["symbol", "trade_date"]])
    keys = pd.MultiIndex.from_arrays([frame["symbol"], dates], names=index.names)
    lookup = pd.Series(values.to_numpy(), index=index)
    return pd.Series(lookup.reindex(keys).to_numpy(), index=frame.index)


def _weighted_complete_row_sum(
    frame: pd.DataFrame,
    columns_and_weights: Sequence[tuple[str, float]],
) -> pd.Series:
    weighted = pd.concat(
        [frame[column].mul(weight) for column, weight in columns_and_weights],
        axis=1,
    )
    return weighted.sum(axis=1, min_count=len(columns_and_weights))


def _add_next_open_labels(
    frame: pd.DataFrame,
    *,
    horizon_weights: Mapping[int, float] | Sequence[tuple[int, float]],
) -> pd.DataFrame:
    out = frame.sort_values(["symbol", "trade_date"], kind="mergesort").copy()
    normalized_weights = normalize_label_horizon_weights(horizon_weights)
    entry_date = _future_date(out, 1)
    entry = pd.to_numeric(_lookup_on_dates(out, out["adj_open"], entry_date), errors="coerce")
    tradable = _known_false_flag(out["is_suspended"])
    entry_tradable = _lookup_on_dates(out, tradable, entry_date).eq(True)
    out["forward_label_start_date"] = entry_date
    return_parts: list[tuple[str, float]] = []
    rank_parts: list[tuple[str, float]] = []
    for horizon, weight in normalized_weights:
        exit_date = _future_date(out, horizon + 1)
        exit_price = pd.to_numeric(
            _lookup_on_dates(out, out["adj_open"], exit_date), errors="coerce"
        )
        exit_tradable = _lookup_on_dates(out, tradable, exit_date).eq(True)
        valid = entry.gt(0) & exit_price.gt(0) & entry_tradable & exit_tradable
        return_col = f"forward_return_{horizon}d"
        rank_col = f"forward_rank_{horizon}d"
        out[return_col] = (exit_price / entry - 1.0).where(valid)
        eligible_return = out[return_col].where(out["hard_eligible"])
        out[rank_col] = eligible_return.groupby(out["trade_date"], sort=False).rank(
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
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip().str.lower()
    return numeric.eq(0).fillna(False) | text.isin({"false", "f", "no", "n"})


_HERMITE_WINDOW: int = 60
_HERMITE_MIN_PERIODS: int = 36


def _add_hermite_stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute Hermite stability from minute_volume_activity.

    Hermite stability measures how Gaussian (stable) the distribution of
    intraday volume activity is over a rolling window.

    ``closeness = -log(1 + h3² + h4²)``  —  HIGHER = more stable/Gaussian

    Matches StyleReplica ``compute_hermite_stability_factor`` variant="closeness",
    applied in long format per symbol.
    """
    va_col = "minute_volume_activity"
    if va_col not in frame.columns or frame[va_col].isna().all():
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

    z = ((out[va_col] - roll_mean) / roll_std).clip(-8.0, 8.0)
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
            out["minute_feature_available"] | out["hermite_stability"].notna()
        )

    return out.drop(columns=["_h3_raw", "_h4_raw"])


def _add_eligibility(frame: pd.DataFrame, config: DailyWatch20FeatureConfig) -> pd.DataFrame:
    out = frame.copy()
    market_ok = _target_market(out["symbol"])
    status_ok = _known_false_flag(out["is_st"]) & _known_false_flag(out["is_suspended"])
    core_columns = ["tr_close", "mom_20", "vol_20", "amount_log_20", "size_log"]
    core_values = out[core_columns].to_numpy(dtype=float)
    core_ok = pd.Series(
        np.isfinite(core_values).all(axis=1),
        index=out.index,
        dtype=bool,
    )
    liquid = out["liquidity_pct"].ge(config.liquidity_floor_quantile).fillna(False)
    out["hard_eligible"] = (
        market_ok
        & status_ok
        & out["listed_days"].ge(config.min_listed_days)
        & out["tr_close"].gt(0)
        & out["amount"].gt(0)
        & core_ok
        & liquid
    ).fillna(False)
    return out


def build_daily_watch20_feature_frame(
    daily: pd.DataFrame,
    minute_daily: pd.DataFrame | None = None,
    *,
    config: DailyWatch20FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build point-in-time features and next-open multi-horizon rank labels."""

    cfg = config or DailyWatch20FeatureConfig()
    out = _prepare_daily_input(daily)
    out = _add_price_features(out)
    out = _add_liquidity_and_style_features(out)
    out = _add_market_regime_features(out)
    out = _join_minute_features(out, minute_daily, lag_trade_days=cfg.minute_lag_trade_days)
    out = _add_hermite_stability(out)
    out = _add_eligibility(out, cfg)
    out = _add_next_open_labels(out, horizon_weights=cfg.label_horizon_weights)
    return out.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)


__all__ = [
    "DAILY_WATCH20_FEATURES",
    "MINUTE_FEATURES",
    "DailyWatch20FeatureConfig",
    "build_daily_watch20_feature_frame",
]
