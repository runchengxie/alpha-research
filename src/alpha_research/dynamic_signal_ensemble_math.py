from __future__ import annotations

import numpy as np
import pandas as pd


def _cross_sectional_zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def _zscore_series(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    std = clean.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=clean.index, dtype=float).where(clean.notna())
    return (clean - clean.mean()) / std


def _cap_positive_weights(weights: pd.Series, max_weight: float | None) -> pd.Series:
    clean = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    if clean.empty or clean.sum() <= 0:
        return pd.Series(dtype=float)
    out = clean / clean.sum()
    if max_weight is None or max_weight <= 0 or max_weight >= 1:
        return out
    cap = pd.Series(float(max_weight), index=out.index, dtype=float)
    fixed = pd.Series(False, index=out.index)
    for _ in range(50):
        over = (out > cap + 1e-12) & (~fixed)
        if not over.any():
            break
        fixed |= over
        out.loc[fixed] = cap.loc[fixed]
        residual = 1.0 - float(out.loc[fixed].sum())
        free = ~fixed
        if residual <= 1e-12 or not free.any():
            break
        base = clean.loc[free]
        if base.sum() <= 0:
            out.loc[free] = residual / int(free.sum())
        else:
            out.loc[free] = residual * base / base.sum()
    total = out.sum()
    return out / total if total > 0 else out


def _apply_turnover_budget(
    *,
    previous: pd.Series,
    target: pd.Series,
    max_l1_turnover: float,
) -> pd.Series:
    index = previous.index.union(target.index)
    prev = previous.reindex(index).fillna(0.0)
    tgt = target.reindex(index).fillna(0.0)
    if prev.sum() <= 0 or max_l1_turnover >= 2.0:
        return tgt
    delta = tgt - prev
    turnover = float(delta.abs().sum())
    if turnover <= max_l1_turnover or turnover <= 1e-12:
        return tgt
    adjusted = prev + delta * (float(max_l1_turnover) / turnover)
    adjusted = adjusted.clip(lower=0.0)
    total = adjusted.sum()
    return adjusted / total if total > 0 else tgt
