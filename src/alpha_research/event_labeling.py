"""Event-based financial labels, including triple-barrier and meta-labeling.

The functions in this module are deliberately data-frame oriented and avoid
depending on the orchestration layer. They operate on point-in-time price
frames and produce a stable event table that can also be reused by purged
cross-validation and sample-weight calculations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

BarrierName = Literal["profit_taking", "stop_loss", "vertical"]


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Configuration for path-dependent triple-barrier labels."""

    profit_taking: float = 1.0
    stop_loss: float = 1.0
    vertical_horizon: int = 20
    min_target: float = 0.0
    vertical_label: Literal["sign", "zero"] = "sign"

    def __post_init__(self) -> None:
        if self.profit_taking < 0:
            raise ValueError("profit_taking must be >= 0")
        if self.stop_loss < 0:
            raise ValueError("stop_loss must be >= 0")
        if self.vertical_horizon <= 0:
            raise ValueError("vertical_horizon must be > 0")
        if self.min_target < 0:
            raise ValueError("min_target must be >= 0")
        if self.vertical_label not in {"sign", "zero"}:
            raise ValueError("vertical_label must be one of: sign, zero")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_ewm_volatility(
    prices: pd.DataFrame,
    *,
    span: int = 100,
    symbol_col: str = "symbol",
    time_col: str = "trade_date",
    price_col: str = "close",
    output_col: str = "target",
) -> pd.DataFrame:
    """Estimate point-in-time exponentially weighted return volatility."""

    if span <= 1:
        raise ValueError("span must be > 1")
    _require_columns(prices, (symbol_col, time_col, price_col), "prices")
    data = prices[[symbol_col, time_col, price_col]].copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    if data[time_col].isna().any():
        raise ValueError(f"{time_col} must be datetime-like")
    data = data.sort_values([symbol_col, time_col], kind="mergesort")
    returns = data.groupby(symbol_col, sort=False)[price_col].pct_change()
    data[output_col] = returns.groupby(data[symbol_col], sort=False).transform(
        lambda values: values.ewm(span=span, adjust=False).std(bias=False)
    )
    return data[[symbol_col, time_col, output_col]]


def make_volatility_events(
    prices: pd.DataFrame,
    *,
    span: int = 100,
    event_filter: pd.DataFrame | None = None,
    symbol_col: str = "symbol",
    time_col: str = "trade_date",
    price_col: str = "close",
    target_col: str = "target",
) -> pd.DataFrame:
    """Create an event seed table with volatility-scaled targets."""

    volatility = estimate_ewm_volatility(
        prices,
        span=span,
        symbol_col=symbol_col,
        time_col=time_col,
        price_col=price_col,
        output_col=target_col,
    )
    if event_filter is None:
        events = volatility
    else:
        _require_columns(event_filter, (symbol_col, time_col), "event_filter")
        selected = event_filter[[symbol_col, time_col]].copy()
        selected[time_col] = pd.to_datetime(selected[time_col], errors="coerce")
        events = selected.merge(volatility, on=[symbol_col, time_col], how="left")
    return events.dropna(subset=[target_col]).reset_index(drop=True)


def label_triple_barrier(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: TripleBarrierConfig | None = None,
    symbol_col: str = "symbol",
    price_time_col: str = "trade_date",
    event_time_col: str = "signal_date",
    price_col: str = "close",
    target_col: str = "target",
    side_col: str = "side",
    event_id_col: str = "event_id",
) -> pd.DataFrame:
    """Apply triple-barrier labels to a panel of price paths.

    ``events`` must contain a symbol, event timestamp, and a positive target
    width. An optional side column enables meta-labeling semantics. The output
    records the exact label interval, so downstream validation does not need to
    reconstruct it independently.
    """

    cfg = config or TripleBarrierConfig()
    _require_columns(prices, (symbol_col, price_time_col, price_col), "prices")
    _require_columns(events, (symbol_col, event_time_col, target_col), "events")

    price_data = prices[[symbol_col, price_time_col, price_col]].copy()
    price_data[price_time_col] = pd.to_datetime(price_data[price_time_col], errors="coerce")
    price_data[price_col] = pd.to_numeric(price_data[price_col], errors="coerce")
    if price_data[price_time_col].isna().any():
        raise ValueError(f"{price_time_col} must be datetime-like")
    price_data = price_data.dropna(subset=[price_col]).sort_values(
        [symbol_col, price_time_col], kind="mergesort"
    )

    event_data = events.copy()
    event_data[event_time_col] = pd.to_datetime(event_data[event_time_col], errors="coerce")
    event_data[target_col] = pd.to_numeric(event_data[target_col], errors="coerce")
    if event_data[event_time_col].isna().any():
        raise ValueError(f"{event_time_col} must be datetime-like")
    if event_id_col not in event_data.columns:
        event_data[event_id_col] = np.arange(len(event_data), dtype=int)
    if side_col not in event_data.columns:
        event_data[side_col] = 1.0
    event_data[side_col] = pd.to_numeric(event_data[side_col], errors="coerce").fillna(1.0)
    if bool((event_data[side_col] == 0).any()):
        raise ValueError("side values must be non-zero")

    grouped_prices = {
        symbol: group.reset_index(drop=True)
        for symbol, group in price_data.groupby(symbol_col, sort=False)
    }
    records: list[dict[str, object]] = []
    for event in event_data.to_dict("records"):
        record = _label_event(
            event,
            grouped_prices,
            cfg=cfg,
            symbol_col=symbol_col,
            price_time_col=price_time_col,
            event_time_col=event_time_col,
            price_col=price_col,
            target_col=target_col,
            side_col=side_col,
            event_id_col=event_id_col,
        )
        if record is not None:
            records.append(record)

    if not records:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    event_id_col,
                    symbol_col,
                    event_time_col,
                    "label_start",
                    "label_end",
                    "first_touch",
                    "barrier",
                    "target",
                    "side",
                    "realized_return",
                    "side_adjusted_return",
                    "label",
                    "meta_label",
                    "vertical_end",
                ]
            )
        )
    result = pd.DataFrame.from_records(records)
    return result.sort_values(
        [event_time_col, symbol_col, event_id_col], kind="mergesort"
    ).reset_index(drop=True)


def _label_event(
    event: dict[str, object],
    grouped_prices: dict[object, pd.DataFrame],
    *,
    cfg: TripleBarrierConfig,
    symbol_col: str,
    price_time_col: str,
    event_time_col: str,
    price_col: str,
    target_col: str,
    side_col: str,
    event_id_col: str,
) -> dict[str, object] | None:
    target = float(cast(Any, event[target_col]))
    if not np.isfinite(target) or target < cfg.min_target or target <= 0:
        return None
    panel = grouped_prices.get(cast(Any, event[symbol_col]))
    if panel is None or panel.empty:
        return None
    event_time = pd.Timestamp(cast(Any, event[event_time_col]))
    times = pd.DatetimeIndex(panel[price_time_col])
    start_idx = int(times.searchsorted(cast(Any, event_time), side="left"))
    if start_idx >= len(panel):
        return None
    end_idx = min(start_idx + cfg.vertical_horizon, len(panel) - 1)
    if end_idx <= start_idx:
        return None

    start_price = float(panel.iloc[start_idx][price_col])
    path = panel.iloc[start_idx : end_idx + 1]
    side = float(cast(Any, event[side_col]))
    adjusted_returns = (path[price_col].astype(float) / start_price - 1.0) * side
    barrier, touch_pos = _first_barrier_touch(adjusted_returns, cfg=cfg, target=target)
    touch_row = path.iloc[touch_pos]
    adjusted_return = float(adjusted_returns.iloc[touch_pos])

    record = dict(event)
    record.update(
        {
            event_id_col: event[event_id_col],
            "label_start": pd.Timestamp(panel.iloc[start_idx][price_time_col]),
            "label_end": pd.Timestamp(touch_row[price_time_col]),
            "first_touch": pd.Timestamp(touch_row[price_time_col]),
            "barrier": barrier,
            "target": target,
            "side": side,
            "realized_return": adjusted_return * side,
            "side_adjusted_return": adjusted_return,
            "label": _barrier_label(barrier, adjusted_return, cfg.vertical_label),
            "meta_label": int(adjusted_return > 0),
            "vertical_end": pd.Timestamp(panel.iloc[end_idx][price_time_col]),
        }
    )
    return record


def _first_barrier_touch(
    adjusted_returns: pd.Series,
    *,
    cfg: TripleBarrierConfig,
    target: float,
) -> tuple[BarrierName, int]:
    upper = cfg.profit_taking * target if cfg.profit_taking > 0 else np.inf
    lower = -cfg.stop_loss * target if cfg.stop_loss > 0 else -np.inf
    values = adjusted_returns.to_numpy(dtype=float)
    upper_hits = np.flatnonzero(values >= upper)
    lower_hits = np.flatnonzero(values <= lower)
    upper_pos = int(upper_hits[0]) if upper_hits.size else None
    lower_pos = int(lower_hits[0]) if lower_hits.size else None
    if upper_pos is not None and (lower_pos is None or upper_pos <= lower_pos):
        return "profit_taking", upper_pos
    if lower_pos is not None:
        return "stop_loss", lower_pos
    return "vertical", len(adjusted_returns) - 1


def _barrier_label(
    barrier: BarrierName,
    adjusted_return: float,
    vertical_label: Literal["sign", "zero"],
) -> int:
    if barrier == "profit_taking":
        return 1
    if barrier == "stop_loss":
        return -1
    if vertical_label == "zero":
        return 0
    return int(np.sign(adjusted_return))


def meta_label_from_predictions(
    events: pd.DataFrame,
    *,
    return_col: str = "side_adjusted_return",
    output_col: str = "meta_label",
) -> pd.DataFrame:
    """Derive binary take/pass labels from side-adjusted event outcomes."""

    _require_columns(events, (return_col,), "events")
    result = events.copy()
    values = pd.to_numeric(result[return_col], errors="coerce")
    result[output_col] = (values > 0).astype("Int64")
    result.loc[values.isna(), output_col] = pd.NA
    return result


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


__all__ = [
    "TripleBarrierConfig",
    "estimate_ewm_volatility",
    "label_triple_barrier",
    "make_volatility_events",
    "meta_label_from_predictions",
]
