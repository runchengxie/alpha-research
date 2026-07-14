"""Point-in-time probability calibration for model sizing evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class CalibrationSummary:
    method: str
    observations: int
    calibration_windows: int
    brier_before: float
    brier_after: float


def expanding_probability_calibration(
    frame: pd.DataFrame,
    *,
    score_col: str,
    outcome_col: str,
    date_col: str = "trade_date",
    method: str = "isotonic",
    min_train_observations: int = 200,
    output_col: str = "calibrated_probability",
) -> tuple[pd.DataFrame, CalibrationSummary]:
    """Calibrate scores using only observations strictly before each date."""

    if method not in {"isotonic", "platt"}:
        raise ValueError("method must be one of: isotonic, platt")
    missing = [col for col in (score_col, outcome_col, date_col) if col not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {', '.join(missing)}")
    data = frame.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data[outcome_col] = pd.to_numeric(data[outcome_col], errors="coerce")
    data = data.sort_values(date_col, kind="mergesort")
    probabilities = pd.Series(np.nan, index=data.index, dtype=float)
    windows = 0

    for date, test in data.groupby(date_col, sort=True):
        train = data.loc[data[date_col] < date].dropna(subset=[score_col, outcome_col])
        train = train.loc[train[outcome_col].isin([0, 1])]
        if len(train) < min_train_observations or train[outcome_col].nunique() < 2:
            continue
        test_scores = data.loc[test.index, score_col]
        valid = test_scores.notna()
        if not bool(valid.any()):
            continue
        if method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(
                train[score_col].to_numpy(dtype=float),
                train[outcome_col].to_numpy(dtype=float),
            )
            predicted = model.predict(test_scores.loc[valid].to_numpy(dtype=float))
        else:
            model = LogisticRegression(solver="lbfgs")
            model.fit(
                train[[score_col]].to_numpy(dtype=float),
                train[outcome_col].to_numpy(dtype=int),
            )
            predicted = model.predict_proba(
                test_scores.loc[valid].to_numpy(dtype=float).reshape(-1, 1)
            )[:, 1]
        probabilities.loc[test_scores.loc[valid].index] = predicted
        windows += 1

    result = data.copy()
    result[output_col] = probabilities
    evaluable = result.dropna(subset=[output_col, outcome_col, score_col])
    raw_probability = _score_to_unit_interval(evaluable[score_col])
    outcome = evaluable[outcome_col].to_numpy(dtype=float)
    calibrated = evaluable[output_col].to_numpy(dtype=float)
    summary = CalibrationSummary(
        method=method,
        observations=len(evaluable),
        calibration_windows=windows,
        brier_before=_brier(raw_probability, outcome),
        brier_after=_brier(calibrated, outcome),
    )
    return result.sort_index(), summary


def probability_to_bet_size(
    probability: pd.Series,
    *,
    classes: int = 2,
    side: pd.Series | float = 1.0,
    step_size: float | None = None,
) -> pd.Series:
    """Map calibrated probabilities to signed AFML-style bet sizes."""

    if classes < 2:
        raise ValueError("classes must be >= 2")
    values = pd.to_numeric(probability, errors="coerce").clip(0.0, 1.0)
    denominator = np.sqrt(values * (1.0 - values))
    z = (values - 1.0 / classes).div(denominator.replace(0.0, np.nan))
    size = 2.0 * _normal_cdf(z) - 1.0
    if isinstance(side, pd.Series):
        side_values = pd.to_numeric(side, errors="coerce").reindex(values.index)
    else:
        side_values = pd.Series(float(side), index=values.index)
    signed = size * np.sign(side_values)
    if step_size is not None:
        if not 0 < step_size <= 1:
            raise ValueError("step_size must be in (0, 1]")
        signed = (signed / step_size).round() * step_size
    return signed.clip(-1.0, 1.0).rename("bet_size")


def _score_to_unit_interval(score: pd.Series) -> np.ndarray:
    values = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)
    if values.size == 0:
        return values
    ranks = pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)
    return np.clip(ranks, 0.0, 1.0)


def _brier(probability: np.ndarray, outcome: np.ndarray) -> float:
    if probability.size == 0:
        return float("nan")
    return float(np.mean(np.square(probability - outcome)))


def _normal_cdf(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    sign = np.sign(x)
    absolute = x.abs() / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * absolute)
    coefficients = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
    polynomial = sum(coef * t.pow(index + 1) for index, coef in enumerate(coefficients))
    erf = sign * (1.0 - polynomial * np.exp(-absolute * absolute))
    return 0.5 * (1.0 + erf)


__all__ = [
    "CalibrationSummary",
    "expanding_probability_calibration",
    "probability_to_bet_size",
]
