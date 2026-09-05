"""Chronological, formation-balanced forecast-skill gating."""

import numpy as np
import pandas as pd


def forecast_skill_gate(history, decision_date, model_col, control_col):
    """Gate using mature OOF forecast MSE, never the current formation's outcome.

    Caller must supply genuinely out-of-fold predictions and their provenance.
    Eight formations with at least twenty matched observations are required.
    This fixed heuristic is not a probability or significance calculation.
    """
    work = history.copy()
    for column in ("formation_date", "label_end_date"):
        work[column] = pd.to_datetime(work[column])
    if work.formation_date.isna().any() or work.symbol.isna().any():
        raise ValueError("formation and symbol keys must be present")
    if work.duplicated(["formation_date", "symbol"]).any():
        raise ValueError("duplicate forecast keys")
    decision = pd.Timestamp(decision_date)
    if pd.isna(decision):
        raise ValueError("decision date must be present")
    work = work.loc[work.formation_date.lt(decision)]
    gains = []
    for _, group in work.groupby("formation_date"):
        if group.label_end_date.isna().any() or not group.label_end_date.lt(decision).all():
            continue
        values = group[["target", model_col, control_col]].to_numpy(dtype=float)
        values = values[np.isfinite(values).all(axis=1)]
        if len(values) < 20:
            continue
        actual, model, control = values.T
        gains.append(float(np.mean((control - actual) ** 2 - (model - actual) ** 2)))
    mean_gain = float(np.mean(gains)) if gains else None
    win_fraction = float(np.mean(np.asarray(gains) > 0)) if gains else None
    return {
        "enabled": bool(
            len(gains) >= 8
            and mean_gain is not None
            and win_fraction is not None
            and mean_gain > 0
            and win_fraction >= 0.6
        ),
        "formations": len(gains),
        "mean_mse_gain": mean_gain,
        "win_fraction": win_fraction,
    }
