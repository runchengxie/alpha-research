from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from market_data_platform.symbols import canonicalize_symbol_columns


@dataclass(frozen=True)
class SignalStabilityResult:
    summary: dict[str, Any]
    by_window: pd.DataFrame
    by_symbol: pd.DataFrame
    by_feature: pd.DataFrame


def compute_signal_stability_diagnostics(
    positions_by_rebalance: pd.DataFrame | None,
    scored_data: pd.DataFrame | None,
    *,
    feature_columns: Sequence[str] | None = None,
    signal_col: str = "signal_backtest",
    buffer_width: int = 20,
) -> SignalStabilityResult:
    positions = _normalize_positions(positions_by_rebalance)
    scored = _normalize_scored(scored_data, signal_col=signal_col)
    if positions.empty:
        return _empty_result("no_positions")
    if scored.empty:
        return _empty_result("no_scored_data")

    rank_panel = _rank_panel(scored, signal_col=signal_col)
    holdings = _merge_rank_panel(positions, rank_panel)
    holdings = _fill_missing_position_ranks(holdings)
    by_window, churn_symbols = _window_rows(holdings, rank_panel, buffer_width=buffer_width)
    by_feature = _feature_rows(churn_symbols, scored, feature_columns or [])
    return SignalStabilityResult(
        summary=_summary("ok", by_window, churn_symbols),
        by_window=by_window,
        by_symbol=churn_symbols,
        by_feature=by_feature,
    )


def _empty_result(status: str) -> SignalStabilityResult:
    return SignalStabilityResult(
        summary={"status": status, "windows": 0},
        by_window=pd.DataFrame(),
        by_symbol=pd.DataFrame(),
        by_feature=pd.DataFrame(),
    )


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    compact = text.str.replace("-", "", regex=False).str.slice(0, 8)
    parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    return parsed.fillna(pd.to_datetime(values, errors="coerce"))


def _normalize_positions(positions: pd.DataFrame | None) -> pd.DataFrame:
    if positions is None or positions.empty:
        return pd.DataFrame()
    required = {"rebalance_date", "symbol", "weight"}
    if not required.issubset(positions.columns):
        return pd.DataFrame()
    out = canonicalize_symbol_columns(positions.copy(), context="signal stability positions")
    out["rebalance_date"] = _parse_dates(out["rebalance_date"])
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    return out.dropna(subset=["rebalance_date", "symbol"]).copy()


def _normalize_scored(scored: pd.DataFrame | None, *, signal_col: str) -> pd.DataFrame:
    if scored is None or scored.empty:
        return pd.DataFrame()
    if not {"trade_date", "symbol", signal_col}.issubset(scored.columns):
        return pd.DataFrame()
    out = canonicalize_symbol_columns(scored.copy(), context="signal stability scored data")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out[signal_col] = pd.to_numeric(out[signal_col], errors="coerce")
    return out.dropna(subset=["trade_date", "symbol", signal_col]).copy()


def _rank_panel(scored: pd.DataFrame, *, signal_col: str) -> pd.DataFrame:
    out = scored.copy()
    out["signal_rank"] = out.groupby("trade_date")[signal_col].rank(
        ascending=False,
        method="first",
    )
    out["universe_count"] = out.groupby("trade_date")["symbol"].transform("nunique")
    return out


def _merge_rank_panel(positions: pd.DataFrame, rank_panel: pd.DataFrame) -> pd.DataFrame:
    rank_cols = ["trade_date", "symbol", "signal_rank", "universe_count"]
    return positions.merge(
        rank_panel[rank_cols].drop_duplicates(subset=["trade_date", "symbol"]),
        left_on=["rebalance_date", "symbol"],
        right_on=["trade_date", "symbol"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")


def _fill_missing_position_ranks(holdings: pd.DataFrame) -> pd.DataFrame:
    out = holdings.copy()
    if "rank" not in out.columns:
        return out
    rank_values = pd.to_numeric(out["rank"], errors="coerce")
    missing = out["signal_rank"].isna()
    out.loc[missing, "signal_rank"] = rank_values.loc[missing]
    return out


def _window_rows(
    holdings: pd.DataFrame,
    rank_panel: pd.DataFrame,
    *,
    buffer_width: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []
    grouped = list(holdings.sort_values("rebalance_date").groupby("rebalance_date", sort=True))
    previous: pd.DataFrame | None = None
    for rebalance_date, current in grouped:
        if previous is None:
            previous = current.copy()
            continue
        window_row, changes = _one_window(previous, current, rank_panel, buffer_width)
        window_row["rebalance_date"] = pd.Timestamp(rebalance_date).strftime("%Y%m%d")
        rows.append(window_row)
        symbol_rows.extend(changes)
        previous = current.copy()
    return pd.DataFrame(rows), pd.DataFrame(symbol_rows)


def _one_window(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    rank_panel: pd.DataFrame,
    buffer_width: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prev_symbols = set(previous["symbol"])
    curr_symbols = set(current["symbol"])
    entrants = sorted(curr_symbols - prev_symbols)
    exits = sorted(prev_symbols - curr_symbols)
    prev_date = pd.Timestamp(previous["rebalance_date"].iloc[0])
    curr_date = pd.Timestamp(current["rebalance_date"].iloc[0])
    prev_ranks = _date_ranks(rank_panel, prev_date)
    curr_ranks = _date_ranks(rank_panel, curr_date)
    prev_held_ranks = _held_ranks(previous)
    curr_held_ranks = _held_ranks(current)
    prev_rank_lookup = {**prev_ranks, **prev_held_ranks}
    curr_rank_lookup = {**curr_ranks, **curr_held_ranks}
    prev_rank_values = [prev_rank_lookup.get(symbol, np.nan) for symbol in entrants]
    exit_curr_values = [curr_rank_lookup.get(symbol, np.nan) for symbol in exits]
    top_k = int((current["weight"] != 0).sum())
    row = {
        "previous_rebalance_date": prev_date.strftime("%Y%m%d"),
        "entrant_count": len(entrants),
        "exit_count": len(exits),
        "overlap_count": len(prev_symbols & curr_symbols),
        "rank_correlation": _rank_correlation(prev_rank_lookup, curr_rank_lookup),
        "entrant_prev_rank_mean": _nanmean(prev_rank_values),
        "exit_curr_rank_mean": _nanmean(exit_curr_values),
        "entrant_from_buffer_count": _buffer_count(prev_rank_values, top_k, buffer_width),
        "exit_to_buffer_count": _buffer_count(exit_curr_values, top_k, buffer_width),
    }
    changes = _symbol_rows(
        entrants,
        exits,
        prev_date,
        curr_date,
        prev_rank_lookup,
        curr_rank_lookup,
    )
    return row, changes


def _held_ranks(positions: pd.DataFrame) -> dict[str, float]:
    if "signal_rank" not in positions.columns:
        return {}
    return dict(zip(positions["symbol"], positions["signal_rank"], strict=False))


def _date_ranks(rank_panel: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]:
    frame = rank_panel.loc[rank_panel["trade_date"] == date]
    return dict(zip(frame["symbol"], frame["signal_rank"], strict=False))


def _rank_correlation(prev_ranks: dict[str, float], curr_ranks: dict[str, float]) -> float:
    common = sorted(set(prev_ranks) & set(curr_ranks))
    if len(common) < 2:
        return float("nan")
    prev = pd.Series([prev_ranks[symbol] for symbol in common])
    curr = pd.Series([curr_ranks[symbol] for symbol in common])
    return float(prev.corr(curr, method="spearman"))


def _nanmean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if pd.notna(value)]
    return float(np.mean(clean)) if clean else float("nan")


def _buffer_count(values: Sequence[float], top_k: int, width: int) -> int:
    upper = top_k + max(int(width), 0)
    return sum(1 for value in values if pd.notna(value) and top_k < float(value) <= upper)


def _symbol_rows(
    entrants: Sequence[str],
    exits: Sequence[str],
    prev_date: pd.Timestamp,
    curr_date: pd.Timestamp,
    prev_ranks: dict[str, float],
    curr_ranks: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for side, symbols in (("exit", exits), ("entrant", entrants)):
        for symbol in symbols:
            rows.append(
                {
                    "rebalance_date": curr_date.strftime("%Y%m%d"),
                    "previous_rebalance_date": prev_date.strftime("%Y%m%d"),
                    "symbol": symbol,
                    "change_type": side,
                    "previous_rank": prev_ranks.get(symbol, np.nan),
                    "current_rank": curr_ranks.get(symbol, np.nan),
                }
            )
    return rows


def _feature_rows(
    churn_symbols: pd.DataFrame,
    scored: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    if churn_symbols.empty or not features:
        return pd.DataFrame()
    churn = churn_symbols.copy()
    churn["rebalance_date"] = pd.to_datetime(churn["rebalance_date"], errors="coerce")
    merged = churn.merge(
        scored,
        left_on=["rebalance_date", "symbol"],
        right_on=["trade_date", "symbol"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in merged.columns:
            continue
        for keys, group in merged.groupby(["rebalance_date", "change_type"], sort=True):
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            if values.empty:
                continue
            rebalance_date, change_type = keys
            rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "change_type": change_type,
                    "feature": feature,
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "symbol_count": int(values.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _summary(status: str, by_window: pd.DataFrame, by_symbol: pd.DataFrame) -> dict[str, Any]:
    return {
        "status": status,
        "windows": len(by_window),
        "avg_rank_correlation": _column_mean(by_window, "rank_correlation"),
        "avg_entrant_count": _column_mean(by_window, "entrant_count"),
        "avg_exit_count": _column_mean(by_window, "exit_count"),
        "avg_entrant_from_buffer_count": _column_mean(by_window, "entrant_from_buffer_count"),
        "avg_exit_to_buffer_count": _column_mean(by_window, "exit_to_buffer_count"),
        "churn_symbols": int(by_symbol["symbol"].nunique()) if not by_symbol.empty else 0,
    }


def _column_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").mean())
