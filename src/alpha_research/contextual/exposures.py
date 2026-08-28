from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

_NORMALIZATIONS = {"rank_pct", "zscore_clip"}
_MISSING_POLICIES = {"ignore_modifier", "missing_exposure"}
_UNKNOWN_INDUSTRY_POLICIES = {"zero_prior", "missing_exposure"}


@dataclass(frozen=True)
class FundamentalModifier:
    field: str
    direction: float
    weight: float
    normalization: str
    missing: str = "ignore_modifier"

    def __post_init__(self) -> None:
        if not str(self.field).strip():
            raise ValueError("FundamentalModifier.field must be non-empty")
        if not math.isfinite(float(self.direction)) or float(self.direction) == 0:
            raise ValueError("FundamentalModifier.direction must be a finite non-zero number")
        if not math.isfinite(float(self.weight)) or float(self.weight) < 0:
            raise ValueError("FundamentalModifier.weight must be finite and non-negative")
        if self.normalization not in _NORMALIZATIONS:
            raise ValueError(
                f"FundamentalModifier.normalization must be one of {sorted(_NORMALIZATIONS)}"
            )
        if self.missing not in _MISSING_POLICIES:
            raise ValueError(
                f"FundamentalModifier.missing must be one of {sorted(_MISSING_POLICIES)}"
            )


@dataclass(frozen=True)
class ExposureSpec:
    name: str
    industry_prior_map: Mapping[str, float]
    fundamental_modifiers: tuple[FundamentalModifier, ...] = ()
    clip_min: float = -1.0
    clip_max: float = 1.0
    unknown_industry: str = "missing_exposure"
    version: str = "v1"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("ExposureSpec.name must be non-empty")
        if not str(self.version).strip():
            raise ValueError("ExposureSpec.version must be non-empty")
        if not math.isfinite(float(self.clip_min)) or not math.isfinite(float(self.clip_max)):
            raise ValueError("ExposureSpec clip bounds must be finite")
        if float(self.clip_min) >= float(self.clip_max):
            raise ValueError("ExposureSpec clip_min must be below clip_max")
        if self.unknown_industry not in _UNKNOWN_INDUSTRY_POLICIES:
            raise ValueError(
                f"ExposureSpec.unknown_industry must be one of {sorted(_UNKNOWN_INDUSTRY_POLICIES)}"
            )
        priors: dict[str, float] = {}
        for raw_name, raw_value in self.industry_prior_map.items():
            name = str(raw_name).strip()
            value = float(raw_value)
            if not name:
                raise ValueError("ExposureSpec industry names must be non-empty")
            if not math.isfinite(value):
                raise ValueError("ExposureSpec industry prior values must be finite")
            priors[name] = value
        object.__setattr__(self, "industry_prior_map", MappingProxyType(priors))
        object.__setattr__(self, "fundamental_modifiers", tuple(self.fundamental_modifiers))


def _rank_pct(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    output = pd.Series(np.nan, index=values.index, dtype=float)
    count = int(valid.sum())
    if count == 0:
        return output
    if count == 1:
        output.loc[valid] = 0.0
        return output
    ranks = numeric.loc[valid].rank(method="average")
    output.loc[valid] = 2.0 * (ranks - 1.0) / float(count - 1) - 1.0
    return output


def _zscore_clip(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    output = pd.Series(np.nan, index=values.index, dtype=float)
    if not valid.any():
        return output
    subset = numeric.loc[valid].astype(float)
    std = float(subset.std(ddof=0))
    if std == 0.0 or not math.isfinite(std):
        output.loc[valid] = 0.0
        return output
    zscore = (subset - float(subset.mean())) / std
    output.loc[valid] = zscore.clip(-3.0, 3.0) / 3.0
    return output


def _normalize_by_date(frame: pd.DataFrame, *, field: str, date_col: str, method: str) -> pd.Series:
    if field not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    normalizer = _rank_pct if method == "rank_pct" else _zscore_clip
    pieces = [normalizer(group[field]) for _, group in frame.groupby(date_col, sort=False)]
    if not pieces:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(pieces).sort_index()


def _prior_values(
    frame: pd.DataFrame,
    spec: ExposureSpec,
    *,
    industry_col: str,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    priors: list[float] = []
    evidence: list[dict[str, Any]] = []
    for industry in frame[industry_col].astype(str):
        if industry in spec.industry_prior_map:
            prior = float(spec.industry_prior_map[industry])
            priors.append(prior)
            evidence.append({"industry": industry, "industry_prior": prior, "modifiers": []})
        elif spec.unknown_industry == "zero_prior":
            priors.append(0.0)
            evidence.append({"industry": industry, "industry_prior": 0.0, "modifiers": []})
        else:
            priors.append(np.nan)
            evidence.append({"industry": industry, "industry_prior": None, "modifiers": []})
    return pd.Series(priors, index=frame.index, dtype=float), evidence


def _apply_modifier(
    base: pd.Series,
    evidence: list[dict[str, Any]],
    normalized: pd.Series,
    modifier: FundamentalModifier,
) -> pd.Series:
    result = base.copy()
    for position, index in enumerate(result.index):
        value = normalized.loc[index]
        if pd.isna(value):
            evidence[position]["modifiers"].append(
                {
                    "field": modifier.field,
                    "status": "missing",
                    "normalization": modifier.normalization,
                }
            )
            if modifier.missing == "missing_exposure":
                result.loc[index] = np.nan
            continue
        contribution = float(modifier.direction) * float(modifier.weight) * float(value)
        if not pd.isna(result.loc[index]):
            result.loc[index] = float(result.loc[index]) + contribution
        evidence[position]["modifiers"].append(
            {
                "field": modifier.field,
                "status": "applied",
                "normalization": modifier.normalization,
                "normalized_value": float(value),
                "direction": float(modifier.direction),
                "weight": float(modifier.weight),
                "contribution": contribution,
            }
        )
    return result


def build_company_exposures(
    stock_frame: pd.DataFrame,
    specs: Sequence[ExposureSpec],
    *,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    industry_col: str = "industry",
) -> pd.DataFrame:
    """Build bounded, explainable company sensitivities from PIT stock inputs.

    Fundamental modifiers are normalized independently inside each ``date_col``
    cross-section. The function deliberately performs no as-of filling; callers must
    supply industry and fundamental fields valid for each stock/date row.
    """

    required = {date_col, symbol_col, industry_col}
    missing = sorted(required.difference(stock_frame.columns))
    if missing:
        raise ValueError(f"stock_frame is missing columns: {', '.join(missing)}")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate ExposureSpec.name values are not allowed")

    frame = stock_frame.copy().reset_index(drop=True)
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    outputs: list[pd.DataFrame] = []
    for spec in specs:
        values, evidence = _prior_values(frame, spec, industry_col=industry_col)
        for modifier in spec.fundamental_modifiers:
            normalized = _normalize_by_date(
                frame,
                field=modifier.field,
                date_col=date_col,
                method=modifier.normalization,
            )
            values = _apply_modifier(values, evidence, normalized, modifier)
        values = values.clip(float(spec.clip_min), float(spec.clip_max))
        outputs.append(
            pd.DataFrame(
                {
                    "trade_date": frame[date_col].to_numpy(),
                    "symbol": frame[symbol_col].astype(str).to_numpy(),
                    "exposure_name": spec.name,
                    "exposure_value": values.to_numpy(),
                    "source_components": evidence,
                    "exposure_version": spec.version,
                }
            )
        )
    if not outputs:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "trade_date",
                    "symbol",
                    "exposure_name",
                    "exposure_value",
                    "source_components",
                    "exposure_version",
                ]
            )
        )
    return pd.concat(outputs, ignore_index=True)
