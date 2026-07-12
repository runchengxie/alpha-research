"""Portfolio construction helpers for the StyleReplica strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd

from .signal_generator import StyleReplicaConfig

A_BUFFER_EXIT_MULTIPLIER = 1.3
B_BUFFER_EXIT_RANK = 35
B_BUFFER_ENTRY_RANK = 20
B_MAX_DAILY_REPLACEMENTS = 5


@dataclass
class StyleReplicaPortfolioConfig:
    """Configuration for converting StyleReplica scores into target positions."""

    a_slots: int = 80
    a_capital_weight: float = 0.80
    b_slots: int = 20
    b_capital_weight: float = 0.20
    theme_quotas: dict[str, int] = field(default_factory=dict)
    b_industry_cap: int = 3
    a_buffer_exit_multiplier: float = A_BUFFER_EXIT_MULTIPLIER
    b_buffer_exit_rank: int = B_BUFFER_EXIT_RANK
    b_buffer_entry_rank: int = B_BUFFER_ENTRY_RANK
    b_max_daily_replacements: int = B_MAX_DAILY_REPLACEMENTS
    overlap_policy: str = "aggregate"
    normal_slot_weight: float = 0.01
    max_name_weight: float = 0.02
    max_daily_replacements: int = 15
    model_version: str = "StyleReplica-A80B20-v0"

    @classmethod
    def from_signal_config(cls, config: StyleReplicaConfig) -> Self:
        """Build portfolio settings from the shared signal configuration."""

        return cls(
            a_slots=config.a_slots,
            a_capital_weight=config.a_capital_weight,
            b_slots=config.b_slots,
            b_capital_weight=config.b_capital_weight,
            theme_quotas=dict(config.theme_quotas),
            b_industry_cap=config.b_industry_cap,
            overlap_policy=config.overlap_policy,
            normal_slot_weight=config.normal_slot_weight,
            max_name_weight=config.max_name_weight,
            max_daily_replacements=config.max_daily_replacements,
            model_version=config.model_version,
        )


def _prepare_signals_frame(signals: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol"}
    missing = sorted(required - set(signals.columns))
    if missing:
        raise ValueError("StyleReplica signals are missing column(s): " + ", ".join(missing))

    frame = signals.copy()
    if "signal_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    elif "trade_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    else:
        raise ValueError("StyleReplica signals require signal_date or trade_date.")

    for column, default in (
        ("score_a", np.nan),
        ("score_b", np.nan),
        ("leg", None),
        ("theme", None),
        ("industry", None),
    ):
        if column not in frame.columns:
            frame[column] = default

    frame["symbol"] = frame["symbol"].astype(str)
    return frame.dropna(subset=["trade_date", "symbol"])


def _ranked_symbols(frame: pd.DataFrame, score_col: str) -> list[str]:
    return frame.sort_values(score_col, ascending=False)["symbol"].astype(str).tolist()


def _select_a_leg_for_date(
    day_signals: pd.DataFrame,
    previous: set[str],
    *,
    theme_quotas: dict[str, int],
    buffer_exit_multiplier: float,
) -> list[str]:
    selected: list[str] = []
    selected_set: set[str] = set()

    for theme, quota in theme_quotas.items():
        if quota <= 0:
            continue
        candidates = day_signals.loc[
            day_signals["theme"].eq(theme) & day_signals["score_a"].notna()
        ]
        ranked = _ranked_symbols(candidates, "score_a")
        exit_rank = max(quota + 1, int(quota * buffer_exit_multiplier))
        pool = [
            symbol
            for rank, symbol in enumerate(ranked, start=1)
            if (symbol in previous and rank <= exit_rank)
            or (symbol not in previous and rank <= quota)
        ][:quota]
        if len(pool) < quota:
            pool.extend(symbol for symbol in ranked if symbol not in pool)
            pool = pool[:quota]
        for symbol in pool:
            if symbol not in selected_set:
                selected.append(symbol)
                selected_set.add(symbol)

    return selected


def _industry_value(row: pd.Series) -> str:
    value = row.get("industry")
    return "" if pd.isna(value) else str(value)


def _select_b_leg_for_date(
    day_signals: pd.DataFrame,
    previous: set[str],
    *,
    slots: int,
    industry_cap: int,
    exit_rank: int,
    entry_rank: int,
    max_replacements: int,
) -> list[str]:
    ranked = day_signals.loc[day_signals["score_b"].notna()].sort_values("score_b", ascending=False)
    selected: list[str] = []
    industry_counts: dict[str, int] = {}
    replacements = 0

    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        symbol = str(row["symbol"])
        held = symbol in previous
        if held and rank > exit_rank:
            continue
        if not held and (rank > entry_rank or replacements >= max_replacements):
            continue

        industry = _industry_value(row)
        if industry_counts.get(industry, 0) >= industry_cap:
            continue
        selected.append(symbol)
        if industry:
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if not held:
            replacements += 1
        if len(selected) >= slots:
            return selected

    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        symbol = str(row["symbol"])
        if symbol in selected or rank > exit_rank * 2:
            continue
        selected.append(symbol)
        if len(selected) >= slots:
            break
    return selected


def _resolve_overlap(
    a_holdings: list[str],
    b_holdings: list[str],
    *,
    policy: str,
    normal_weight: float,
    max_weight: float,
) -> tuple[list[str], list[str], dict[str, float]]:
    if policy not in {"aggregate", "deduplicate"}:
        raise ValueError("overlap_policy must be aggregate or deduplicate.")

    a_unique = list(dict.fromkeys(a_holdings))
    b_unique = list(dict.fromkeys(b_holdings))
    overlap = set(a_unique) & set(b_unique)
    if policy == "deduplicate":
        b_unique = [symbol for symbol in b_unique if symbol not in overlap]

    weights = dict.fromkeys(a_unique, normal_weight)
    for symbol in b_unique:
        weights[symbol] = (
            min(max_weight, normal_weight * 2)
            if symbol in overlap and policy == "aggregate"
            else normal_weight
        )
    return a_unique, b_unique, weights


def _fill_a_leg(day: pd.DataFrame, holdings: list[str], slots: int) -> list[str]:
    result = list(holdings)
    candidates = day.loc[day["theme"].notna() & day["score_a"].notna()].sort_values(
        "score_a", ascending=False
    )
    for symbol in candidates["symbol"].astype(str):
        if symbol not in result:
            result.append(symbol)
        if len(result) >= slots:
            break
    return result


def _fill_b_leg(day: pd.DataFrame, holdings: list[str], slots: int) -> list[str]:
    result = list(holdings)
    candidates = day.loc[day["score_b"].notna()].sort_values("score_b", ascending=False)
    for symbol in candidates["symbol"].astype(str):
        if symbol not in result:
            result.append(symbol)
        if len(result) >= slots:
            break
    return result


def _build_position_rows(
    date_text: str,
    a_holdings: list[str],
    b_holdings: list[str],
    weights: dict[str, float],
    day_signals: pd.DataFrame,
) -> list[dict[str, Any]]:
    lookup = day_signals.drop_duplicates("symbol", keep="last").set_index("symbol")
    a_set = set(a_holdings)
    b_set = set(b_holdings)
    rows: list[dict[str, Any]] = []

    for symbol in sorted(a_set | b_set):
        leg = "A+B" if symbol in a_set and symbol in b_set else ("A" if symbol in a_set else "B")
        signal_row = lookup.loc[symbol] if symbol in lookup.index else pd.Series(dtype=object)
        score_a = pd.to_numeric(signal_row.get("score_a"), errors="coerce")
        score_b = pd.to_numeric(signal_row.get("score_b"), errors="coerce")
        signal = score_a if pd.notna(score_a) else score_b
        rows.append(
            {
                "rebalance_date": date_text,
                "entry_date": date_text,
                "symbol": symbol,
                "weight": weights.get(symbol, 0.0),
                "side": "long",
                "leg": leg,
                "signal": signal,
                "score_a": score_a,
                "score_b": score_b,
                "theme": signal_row.get("theme"),
                "industry": signal_row.get("industry"),
            }
        )
    return rows


def build_style_replica_positions(
    signals: pd.DataFrame,
    *,
    config: StyleReplicaPortfolioConfig | None = None,
) -> pd.DataFrame:
    """Convert StyleReplica scores into a positions-by-rebalance frame."""

    cfg = config or StyleReplicaPortfolioConfig()
    frame = _prepare_signals_frame(signals)
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    previous_a: set[str] = set()
    previous_b: set[str] = set()
    for date in sorted(frame["trade_date"].unique()):
        day = frame.loc[frame["trade_date"].eq(date)]
        a_holdings = _select_a_leg_for_date(
            day,
            previous_a,
            theme_quotas=cfg.theme_quotas,
            buffer_exit_multiplier=cfg.a_buffer_exit_multiplier,
        )
        b_holdings = _select_b_leg_for_date(
            day,
            previous_b,
            slots=cfg.b_slots,
            industry_cap=cfg.b_industry_cap,
            exit_rank=cfg.b_buffer_exit_rank,
            entry_rank=cfg.b_buffer_entry_rank,
            max_replacements=cfg.b_max_daily_replacements,
        )
        a_holdings = _fill_a_leg(day, a_holdings, cfg.a_slots)
        b_holdings = _fill_b_leg(day, b_holdings, cfg.b_slots)
        a_final, b_final, weights = _resolve_overlap(
            a_holdings,
            b_holdings,
            policy=cfg.overlap_policy,
            normal_weight=cfg.normal_slot_weight,
            max_weight=cfg.max_name_weight,
        )
        date_text = pd.Timestamp(date).strftime("%Y%m%d")
        rows.extend(_build_position_rows(date_text, a_final, b_final, weights, day))
        previous_a = set(a_final)
        previous_b = set(b_final)

    positions = pd.DataFrame(rows)
    if positions.empty:
        return positions
    positions["rank"] = (
        positions.groupby("rebalance_date", sort=False)["signal"]
        .rank(ascending=False, method="first", na_option="bottom")
        .astype("Int64")
    )
    return positions.sort_values(["rebalance_date", "rank", "symbol"]).reset_index(drop=True)


def compute_daily_changes(positions: pd.DataFrame) -> pd.DataFrame:
    """Compare adjacent position snapshots and classify each change."""

    if positions.empty:
        return pd.DataFrame()
    dates = sorted(positions["rebalance_date"].unique())
    changes: list[dict[str, Any]] = []
    previous: dict[str, float] = {}

    for date in dates:
        day = positions.loc[positions["rebalance_date"].eq(date)]
        current = dict(zip(day["symbol"], day["weight"], strict=True))
        legs = dict(zip(day["symbol"], day.get("leg", [None] * len(day)), strict=True))
        for symbol in sorted(set(current) | set(previous)):
            weight = float(current.get(symbol, 0.0))
            previous_weight = float(previous.get(symbol, 0.0))
            if weight > 0 and previous_weight == 0:
                action = "new"
            elif weight == 0 and previous_weight > 0:
                action = "exit"
            elif weight != previous_weight:
                action = "weight_change"
            else:
                action = "stay"
            changes.append(
                {
                    "rebalance_date": date,
                    "symbol": symbol,
                    "action": action,
                    "leg": legs.get(symbol),
                    "weight": weight,
                    "prev_weight": previous_weight,
                    "weight_change": weight - previous_weight,
                }
            )
        previous = current
    return pd.DataFrame(changes)


def compute_style_exposure_summary(
    positions: pd.DataFrame,
    *,
    factor_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Summarize leg, theme, and industry exposure for one position snapshot."""

    del factor_frame
    if positions.empty:
        return {}
    legs = positions["leg"].astype("string")
    summary: dict[str, Any] = {
        "rebalance_date": str(positions["rebalance_date"].iloc[0]),
        "total_stocks": len(positions),
        "a_leg_count": int(legs.str.contains("A", na=False).sum()),
        "b_leg_count": int(legs.str.contains("B", na=False).sum()),
        "overlap_count": int(legs.eq("A+B").sum()),
        "total_weight": float(positions["weight"].sum()),
        "a_weight": float(positions.loc[legs.str.contains("A", na=False), "weight"].sum()),
        "b_weight": float(positions.loc[legs.str.contains("B", na=False), "weight"].sum()),
    }
    if "theme" in positions.columns:
        summary["theme_distribution"] = positions["theme"].value_counts().to_dict()
    if "industry" in positions.columns:
        counts = positions["industry"].value_counts().to_dict()
        summary["industry_distribution"] = counts
        if counts:
            largest = max(counts, key=counts.get)
            summary["max_industry_pct"] = round(counts[largest] / len(positions), 4)
    return summary


def compute_daily_exposure(positions: pd.DataFrame) -> pd.DataFrame:
    """Compute one exposure summary for each rebalance date."""

    if positions.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        compute_style_exposure_summary(positions.loc[positions["rebalance_date"].eq(date)])
        for date in sorted(positions["rebalance_date"].unique())
    )
