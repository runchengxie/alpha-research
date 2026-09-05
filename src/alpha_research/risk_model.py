"""Framework-neutral factor risk-model primitives for research and portfolio owners."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

FACTOR_RISK_MODEL_SCHEMA = "alpha_research.factor_risk_model.v1"


def _numeric_frame(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    if frame.empty or frame.shape[1] == 0:
        raise ValueError(f"{label} must not be empty")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must contain only finite numeric values")
    if numeric.index.has_duplicates:
        raise ValueError(f"{label} index must be unique")
    if numeric.columns.has_duplicates:
        raise ValueError(f"{label} columns must be unique")
    numeric.index = (
        numeric.index.map(str) if not isinstance(numeric.index, pd.DatetimeIndex) else numeric.index
    )
    numeric.columns = numeric.columns.map(str)
    return numeric.astype(float)


def _comparison_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if not isinstance(timestamp, pd.Timestamp):
        raise ValueError("comparison timestamp must be present")
    return timestamp.tz_convert(None) if timestamp.tzinfo is not None else timestamp


def _history_frame(frame: pd.DataFrame, *, label: str, as_of: pd.Timestamp) -> pd.DataFrame:
    numeric = _numeric_frame(frame, label=label)
    if not isinstance(numeric.index, pd.DatetimeIndex):
        raise ValueError(f"{label} must use a DatetimeIndex")
    index = pd.DatetimeIndex(numeric.index)
    if index.tz is not None:
        index = index.tz_convert(None)
    normalized_as_of = _comparison_timestamp(as_of)
    if (index > normalized_as_of).any():
        raise ValueError(f"{label} contains observations after as_of")
    numeric.index = index
    return numeric.sort_index()


@dataclass(frozen=True)
class FactorRiskModelEstimate:
    """One as-of factor risk estimate using platform-native pandas objects."""

    as_of: pd.Timestamp
    exposures: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_risk: pd.Series
    history_start: pd.Timestamp
    history_end: pd.Timestamp
    observations: int
    covariance_shrinkage: float
    schema_version: str = FACTOR_RISK_MODEL_SCHEMA

    def validate(self) -> None:
        if self.schema_version != FACTOR_RISK_MODEL_SCHEMA:
            raise ValueError(f"unsupported risk model schema {self.schema_version!r}")
        if self.exposures.empty:
            raise ValueError("exposures must not be empty")
        factors = tuple(map(str, self.exposures.columns))
        assets = tuple(map(str, self.exposures.index))
        if list(self.factor_covariance.index.map(str)) != list(factors):
            raise ValueError("factor_covariance index must match exposure factors")
        if list(self.factor_covariance.columns.map(str)) != list(factors):
            raise ValueError("factor_covariance columns must match exposure factors")
        if list(self.specific_risk.index.map(str)) != list(assets):
            raise ValueError("specific_risk index must match exposure assets")
        if (self.specific_risk <= 0).any():
            raise ValueError("specific_risk must be positive")
        if self.observations < 2:
            raise ValueError("observations must be >= 2")
        if not 0.0 <= self.covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        arrays = (
            self.exposures.to_numpy(dtype=float),
            self.factor_covariance.to_numpy(dtype=float),
            self.specific_risk.to_numpy(dtype=float),
        )
        if any(not np.isfinite(values).all() for values in arrays):
            raise ValueError("risk model values must be finite")

        factor_covariance = self.factor_covariance.to_numpy(dtype=float)
        if not np.allclose(
            factor_covariance,
            factor_covariance.T,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("factor_covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh((factor_covariance + factor_covariance.T) / 2.0)
        if float(eigenvalues.min()) < -1e-10:
            raise ValueError("factor_covariance must be positive semidefinite")

        as_of = _comparison_timestamp(self.as_of)
        history_start = _comparison_timestamp(self.history_start)
        history_end = _comparison_timestamp(self.history_end)
        if history_start > history_end:
            raise ValueError("history_start must be <= history_end")
        if history_end > as_of:
            raise ValueError("history_end must be at or before as_of")

    def asset_covariance(self) -> pd.DataFrame:
        """Project factor covariance plus specific variance into asset space."""

        self.validate()
        exposures = self.exposures.to_numpy(dtype=float)
        factor_covariance = self.factor_covariance.to_numpy(dtype=float)
        specific_variance = np.diag(np.square(self.specific_risk.to_numpy(dtype=float)))
        covariance = exposures @ factor_covariance @ exposures.T + specific_variance
        covariance = (covariance + covariance.T) / 2.0
        return pd.DataFrame(
            covariance,
            index=self.exposures.index,
            columns=self.exposures.index,
        )

    def receipt(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "as_of": self.as_of.isoformat(),
            "history_start": self.history_start.isoformat(),
            "history_end": self.history_end.isoformat(),
            "observations": self.observations,
            "asset_count": int(self.exposures.shape[0]),
            "factor_count": int(self.exposures.shape[1]),
            "covariance_shrinkage": self.covariance_shrinkage,
        }


def build_factor_risk_model(
    *,
    exposures: pd.DataFrame,
    factor_returns: pd.DataFrame,
    specific_returns: pd.DataFrame,
    as_of: pd.Timestamp,
    covariance_shrinkage: float = 0.0,
    min_observations: int = 20,
) -> FactorRiskModelEstimate:
    """Build a factor covariance + specific-risk estimate from precomputed histories.

    The caller owns factor construction. This function deliberately does not infer
    factor returns from security returns, which keeps research assumptions explicit.
    All historical observations must be at or before ``as_of``.
    """

    normalized_as_of = pd.Timestamp(as_of)
    if not isinstance(normalized_as_of, pd.Timestamp):
        raise ValueError("as_of timestamp must be present")
    exposure_frame = _numeric_frame(exposures, label="exposures")
    factor_history = _history_frame(
        factor_returns,
        label="factor_returns",
        as_of=normalized_as_of,
    )
    specific_history = _history_frame(
        specific_returns,
        label="specific_returns",
        as_of=normalized_as_of,
    )
    if not 0.0 <= covariance_shrinkage <= 1.0:
        raise ValueError("covariance_shrinkage must be in [0, 1]")
    if min_observations < 2:
        raise ValueError("min_observations must be >= 2")

    factors = tuple(map(str, exposure_frame.columns))
    assets = tuple(map(str, exposure_frame.index))
    if set(factor_history.columns) != set(factors):
        raise ValueError("factor columns must match exposures")
    if set(specific_history.columns) != set(assets):
        raise ValueError("specific return columns must match exposure assets")
    factor_history = factor_history.reindex(columns=factors)
    specific_history = specific_history.reindex(columns=assets)

    common_index = factor_history.index.intersection(specific_history.index)
    if len(common_index) < min_observations:
        raise ValueError(f"risk model requires at least {min_observations} common observations")
    factor_history = factor_history.loc[common_index]
    specific_history = specific_history.loc[common_index]

    factor_covariance = factor_history.cov()
    if covariance_shrinkage > 0:
        diagonal = pd.DataFrame(
            np.diag(np.diag(factor_covariance.to_numpy(dtype=float))),
            index=factor_covariance.index,
            columns=factor_covariance.columns,
        )
        factor_covariance = (
            factor_covariance * (1.0 - covariance_shrinkage) + diagonal * covariance_shrinkage
        )
    specific_variance = specific_history.var(ddof=1)
    if (specific_variance <= 0).any() or specific_variance.isna().any():
        raise ValueError("specific return history must imply positive finite variance")
    specific_risk = np.sqrt(specific_variance)
    if not all(isfinite(float(value)) for value in specific_risk):
        raise ValueError("specific risk must be finite")

    history_start = pd.Timestamp(common_index.min())
    history_end = pd.Timestamp(common_index.max())
    if not isinstance(history_start, pd.Timestamp) or not isinstance(history_end, pd.Timestamp):
        raise ValueError("history timestamps must be present")
    estimate = FactorRiskModelEstimate(
        as_of=normalized_as_of,
        exposures=exposure_frame,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
        history_start=history_start,
        history_end=history_end,
        observations=len(common_index),
        covariance_shrinkage=covariance_shrinkage,
    )
    estimate.validate()
    return estimate


__all__ = [
    "FACTOR_RISK_MODEL_SCHEMA",
    "FactorRiskModelEstimate",
    "build_factor_risk_model",
]
