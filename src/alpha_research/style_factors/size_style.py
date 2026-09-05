"""Pure size-style crowding and relative-strength signal computation.

This module owns the numerical kernel migrated from market-intel. It performs no
file I/O, report rendering, publication, or delivery: callers supply aligned
large/small index frames and receive a deterministic result plus bounded series
for evidence/visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

MOMENTUM_WINDOWS: Final[tuple[int, ...]] = (10, 20, 30, 40, 50, 60)
SMALL_HIGH_THRESHOLD: Final[float] = 0.90
LARGE_LOW_THRESHOLD: Final[float] = 0.10
CROWDING_LOOKBACK_DAYS: Final[int] = 20


@dataclass(frozen=True, slots=True)
class SizeStyleSignal:
    """Current size-style state plus evidence series bounded by ``as_of``."""

    as_of: pd.Timestamp
    small_crowding: float
    large_crowding: float
    crowding_zone: str
    signal: str
    short_window: int
    long_window: int
    small_series: pd.Series
    large_series: pd.Series
    relative_strength: pd.Series

    @property
    def n_days(self) -> int:
        return len(self.small_series)


def _timestamp(value: object, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {value}") from exc
    if not isinstance(parsed, pd.Timestamp):
        raise ValueError(f"invalid {label}: {value}")
    return parsed.normalize()


def _validate_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = {"close", "amount"} - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{label} index must be DatetimeIndex")
    if frame.index.has_duplicates:
        raise ValueError(f"{label} index contains duplicate dates")
    result = frame.sort_index().loc[:, ["close", "amount"]].astype(float)
    if result.empty:
        raise ValueError(f"{label} is empty")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{label} contains non-finite close/amount values")
    if (result["close"] <= 0).any() or (result["amount"] <= 0).any():
        raise ValueError(f"{label} close/amount values must be positive")
    return result


def _top3_average(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) >= 3:
        return float(np.mean(np.sort(valid)[-3:]))
    if len(valid):
        return float(np.mean(valid))
    return 0.5


def _rolling_momentum(prices: np.ndarray, window: int) -> np.ndarray:
    n = len(prices)
    result = np.full(n, np.nan)
    result[window:] = prices[window:] / prices[:-window] - 1.0
    return result


def compute_crowding_series(
    large: pd.DataFrame,
    small: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Compute daily small/large crowding on the common observed dates."""

    large_frame = _validate_frame(large, "large")
    small_frame = _validate_frame(small, "small")
    common_dates = large_frame.index.intersection(small_frame.index).sort_values()
    if len(common_dates) <= max(MOMENTUM_WINDOWS):
        raise ValueError("insufficient aligned index history for size-style crowding")

    large_close = large_frame.loc[common_dates, "close"].to_numpy(dtype=float)
    small_close = small_frame.loc[common_dates, "close"].to_numpy(dtype=float)
    large_amount = large_frame.loc[common_dates, "amount"].to_numpy(dtype=float)
    small_amount = small_frame.loc[common_dates, "amount"].to_numpy(dtype=float)
    n = len(common_dates)

    momentum_diff = {
        window: _rolling_momentum(small_close, window) - _rolling_momentum(large_close, window)
        for window in MOMENTUM_WINDOWS
    }
    turnover = small_amount / large_amount
    turnover_rolling: dict[int, np.ndarray] = {}
    for window in MOMENTUM_WINDOWS:
        cumsum = np.cumsum(np.insert(turnover, 0, 0.0))
        rolling = np.full(n, np.nan)
        rolling[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
        turnover_rolling[window] = rolling

    momentum_pct: dict[int, np.ndarray] = {}
    turnover_pct: dict[int, np.ndarray] = {}
    for window in MOMENTUM_WINDOWS:
        diff = momentum_diff[window]
        diff_pct = np.full(n, np.nan)
        rolling = turnover_rolling[window]
        turnover_window_pct = np.full(n, np.nan)
        for index in range(window, n):
            diff_history = diff[window : index + 1]
            diff_pct[index] = float(np.mean(diff_history <= diff[index]))
            turnover_history = rolling[window : index + 1]
            turnover_window_pct[index] = float(np.mean(turnover_history <= rolling[index]))
        momentum_pct[window] = diff_pct
        turnover_pct[window] = turnover_window_pct

    small_values = np.full(n, np.nan)
    large_values = np.full(n, np.nan)
    for index in range(max(MOMENTUM_WINDOWS), n):
        momentum = np.array([momentum_pct[w][index] for w in MOMENTUM_WINDOWS])
        turnover_rank = np.array([turnover_pct[w][index] for w in MOMENTUM_WINDOWS])
        small_values[index] = (_top3_average(momentum) + _top3_average(turnover_rank)) / 2
        large_values[index] = (
            _top3_average(1.0 - momentum) + _top3_average(1.0 - turnover_rank)
        ) / 2

    return (
        pd.Series(small_values, index=common_dates, name="small_crowding"),
        pd.Series(large_values, index=common_dates, name="large_crowding"),
    )


def compute_size_style_signal(
    large: pd.DataFrame,
    small: pd.DataFrame,
    *,
    as_of: object | None = None,
    expected_through: object | None = None,
) -> SizeStyleSignal:
    """Return the crowding regime and MA direction at an explicit cutoff.

    ``expected_through`` is a production freshness contract. When supplied, a
    valid crowding observation must exist through that date. ``as_of`` can be
    used for historical replay and always bounds every returned evidence series.
    """

    large_frame = _validate_frame(large, "large")
    small_frame = _validate_frame(small, "small")
    small_series, large_series = compute_crowding_series(large_frame, small_frame)
    crowding = pd.DataFrame({"small": small_series, "large": large_series}).dropna()
    if crowding.empty:
        raise ValueError("no aligned index data for size-style computation")

    expected = _timestamp(expected_through, "expected-through") if expected_through else None
    if expected is not None:
        available = crowding.loc[:expected]
        actual = available.index.max().normalize() if not available.empty else None
        if actual is None or actual < expected:
            actual_label = actual.strftime("%Y-%m-%d") if actual is not None else "none"
            raise ValueError(
                f"size-style data is stale: actual={actual_label}, expected={expected:%Y-%m-%d}"
            )

    cutoff = _timestamp(as_of, "as-of cutoff") if as_of is not None else expected
    if cutoff is not None:
        crowding = crowding.loc[:cutoff]
    if len(crowding) < CROWDING_LOOKBACK_DAYS:
        raise ValueError(
            "insufficient history for crowding computation: "
            f"need {CROWDING_LOOKBACK_DAYS} valid observations, got {len(crowding)}"
        )

    last_index = crowding.index[-1]
    recent = crowding.tail(CROWDING_LOOKBACK_DAYS)
    high_crowding = bool(
        (recent["small"] > SMALL_HIGH_THRESHOLD).any()
        or (recent["large"] < LARGE_LOW_THRESHOLD).any()
    )
    crowding_zone = "high_crowding" if high_crowding else "low_crowding"
    short_window, long_window = (5, 20) if high_crowding else (20, 60)

    small_series = small_series.loc[:last_index]
    large_series = large_series.loc[:last_index]
    common_dates = small_series.index
    relative_strength = (
        large_frame.loc[common_dates, "close"] / small_frame.loc[common_dates, "close"]
    ).rename("large_to_small_relative_strength")
    short_ma = relative_strength.rolling(short_window).mean().iloc[-1]
    long_ma = relative_strength.rolling(long_window).mean().iloc[-1]
    if np.isnan(short_ma) or np.isnan(long_ma):
        signal = "insufficient"
    else:
        signal = "large_cap" if short_ma > long_ma else "small_cap"

    return SizeStyleSignal(
        as_of=last_index.normalize(),
        small_crowding=float(crowding.loc[last_index, "small"]),
        large_crowding=float(crowding.loc[last_index, "large"]),
        crowding_zone=crowding_zone,
        signal=signal,
        short_window=short_window,
        long_window=long_window,
        small_series=small_series,
        large_series=large_series,
        relative_strength=relative_strength.loc[:last_index],
    )
