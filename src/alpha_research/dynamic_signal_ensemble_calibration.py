from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .dynamic_signal_ensemble_types import DynamicSignalEnsembleConfig


def _compute_raw_rank_ic(
    panels: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    values: dict[str, list[float]] = {name: [] for name in panels}
    for date in forward_returns.index:
        realized = forward_returns.loc[date]
        for name, panel in panels.items():
            joined = (
                pd.concat(
                    [panel.loc[date].rename("signal"), realized.rename("return")],
                    axis=1,
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            values[name].append(
                float(joined["signal"].corr(joined["return"], method="spearman"))
                if len(joined) >= 5
                else np.nan
            )
    return pd.DataFrame(values, index=forward_returns.index, dtype=float)


def calibrate_signal_directions(
    raw_rank_ic: pd.DataFrame,
    config: DynamicSignalEnsembleConfig,
    *,
    initial_directions: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    initial = initial_directions or {}
    shifted = raw_rank_ic.shift(1)
    rolling_mean = shifted.rolling(
        config.evaluation_window,
        min_periods=config.evaluation_window,
    ).mean()
    pos_share = (
        shifted.gt(0)
        .rolling(
            config.evaluation_window,
            min_periods=config.evaluation_window,
        )
        .mean()
    )
    neg_share = (
        shifted.lt(0)
        .rolling(
            config.evaluation_window,
            min_periods=config.evaluation_window,
        )
        .mean()
    )
    directions = pd.DataFrame(index=raw_rank_ic.index, columns=raw_rank_ic.columns, dtype=float)
    rows: list[dict[str, Any]] = []

    for factor in raw_rank_ic.columns:
        previous = 1.0 if initial.get(factor, 1.0) >= 0 else -1.0
        for date in raw_rank_ic.index:
            mean_ic = rolling_mean.loc[date, factor]
            pos = pos_share.loc[date, factor]
            neg = neg_share.loc[date, factor]
            consistency = max(pos, neg) if pd.notna(pos) and pd.notna(neg) else np.nan
            inferred = previous if pd.isna(mean_ic) else (1.0 if float(mean_ic) >= 0 else -1.0)
            reason = "insufficient_history"
            direction = previous
            flip_allowed = False
            if pd.notna(mean_ic) and pd.notna(consistency):
                flip_allowed = float(inferred) == float(previous) or (
                    abs(float(mean_ic)) >= config.flip_mean_threshold
                    and float(consistency) >= config.flip_consistency_threshold
                )
                if flip_allowed:
                    direction = inferred
                    reason = "flipped" if float(direction) != float(previous) else "confirmed"
                else:
                    reason = "inertia"
            directions.loc[date, factor] = direction
            rows.append(
                {
                    "date": date,
                    "factor": factor,
                    "direction": direction,
                    "previous_direction": previous,
                    "rolling_rank_ic_mean": None if pd.isna(mean_ic) else float(mean_ic),
                    "positive_rank_ic_ratio": None if pd.isna(pos) else float(pos),
                    "negative_rank_ic_ratio": None if pd.isna(neg) else float(neg),
                    "direction_consistency": None if pd.isna(consistency) else float(consistency),
                    "flip_allowed": bool(flip_allowed),
                    "reason": reason,
                }
            )
            previous = direction

    return directions, pd.DataFrame(rows)


def _apply_direction_panels(
    panels: dict[str, pd.DataFrame],
    directions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        name: panel.mul(directions[name].reindex(panel.index).fillna(1.0), axis=0)
        for name, panel in panels.items()
    }
