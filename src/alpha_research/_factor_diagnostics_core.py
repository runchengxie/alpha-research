"""Factor diagnostics: orchestration and summary assembly.

Public entry point ``compute_factor_diagnostics`` plus the diagnostic-frame
assembly, result construction, input normalization, factor resolution, and
summary helpers. Split out of the historical single-file
:mod:`alpha_research.factor_diagnostics` implementation to keep individual
files smaller while preserving the exact public/private symbol surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from ._factor_diagnostics_rows import (
    _by_factor_summary,
    _correlation_rows,
    _drift_rows,
    _exposure_and_residual_rows,
    _factor_date_rows,
    _industry_rows,
    _size_bucket_rows,
)
from .factor_diagnostics_config import (
    DEFAULT_AUTOCORR_LAGS,
    DEFAULT_INDUSTRY_COLUMNS,
    DEFAULT_SIZE_BUCKET_LABELS,
    DEFAULT_STYLE_COLUMNS,
    FactorDiagnosticsResult,
)
from .factor_diagnostics_math import column_mean as _column_mean
from .symbols import canonicalize_symbol_columns


def compute_factor_diagnostics(
    scored_data: pd.DataFrame | None,
    *,
    feature_columns: Sequence[str] | None = None,
    feature_importance: pd.DataFrame | None = None,
    target_col: str = "future_return",
    style_columns: Sequence[str] = DEFAULT_STYLE_COLUMNS,
    market_cap_col: str | None = None,
    industry_columns: Sequence[str] = DEFAULT_INDUSTRY_COLUMNS,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    top_n: int = 30,
    min_obs: int = 20,
    min_bucket_obs: int = 10,
    size_bucket_count: int = 3,
    size_bucket_labels: Sequence[str] = DEFAULT_SIZE_BUCKET_LABELS,
    include_industry_neutralization: bool = True,
    correlation_threshold: float = 0.90,
    autocorr_lags: Sequence[int] = DEFAULT_AUTOCORR_LAGS,
) -> FactorDiagnosticsResult:
    scored = _normalize_scored(scored_data, date_col=date_col, symbol_col=symbol_col)
    if scored.empty:
        return _empty_result("no_scored_data")
    if target_col not in scored.columns:
        return _empty_result("missing_target", rows=len(scored), target_col=target_col)

    factors = _resolve_factor_columns(
        scored,
        feature_columns=feature_columns,
        feature_importance=feature_importance,
        target_col=target_col,
        top_n=max(int(top_n), 1),
    )
    if not factors:
        return _empty_result("no_numeric_features", rows=len(scored), target_col=target_col)

    style_cols = _existing_numeric_columns(scored, style_columns)
    industry_col = _first_existing(cast(Sequence[str], scored.columns), industry_columns)
    cap_col = _resolve_market_cap_col(cast(Sequence[str], scored.columns), market_cap_col)
    warnings = _input_warnings(style_cols=style_cols, industry_col=industry_col, cap_col=cap_col)

    frames = _diagnostic_frames(
        scored,
        factors,
        target_col,
        style_cols,
        industry_col=industry_col,
        cap_col=cap_col,
        include_industry_neutralization=include_industry_neutralization,
        autocorr_lags=autocorr_lags,
        min_obs=min_obs,
        min_bucket_obs=min_bucket_obs,
        size_bucket_count=size_bucket_count,
        size_bucket_labels=size_bucket_labels,
        correlation_threshold=correlation_threshold,
    )
    summary = _summary(
        status="ok",
        scored=scored,
        factors=factors,
        target_col=target_col,
        style_cols=style_cols,
        industry_col=industry_col,
        cap_col=cap_col,
        warnings=warnings,
        residual_ic=frames["residual_ic"],
        size_bucket=frames["size_bucket"],
        correlation=frames["correlation"],
        correlation_threshold=correlation_threshold,
        min_obs=min_obs,
        min_bucket_obs=min_bucket_obs,
    )
    return _result_from_frames(summary, **frames)


def _diagnostic_frames(
    scored: pd.DataFrame,
    factors: Sequence[str],
    target_col: str,
    style_cols: Sequence[str],
    *,
    industry_col: str | None,
    cap_col: str | None,
    include_industry_neutralization: bool,
    autocorr_lags: Sequence[int],
    min_obs: int,
    min_bucket_obs: int,
    size_bucket_count: int,
    size_bucket_labels: Sequence[str],
    correlation_threshold: float,
) -> dict[str, pd.DataFrame]:
    by_factor_date = _factor_date_rows(scored, factors, target_col, min_obs=min_obs)
    drift = _drift_rows(scored, factors, lags=tuple(sorted(set(autocorr_lags))))
    style_exposure, residual_ic = _exposure_and_residual_rows(
        scored,
        factors,
        target_col,
        style_cols,
        industry_col=industry_col if include_industry_neutralization else None,
        min_obs=min_obs,
    )
    size_bucket = _size_bucket_rows(
        scored,
        factors,
        target_col,
        cap_col,
        bucket_count=size_bucket_count,
        labels=size_bucket_labels,
        min_obs=min_obs,
        min_bucket_obs=min_bucket_obs,
    )
    industry = _industry_rows(
        scored,
        factors,
        target_col,
        industry_col,
        min_bucket_obs=min_bucket_obs,
    )
    correlation = _correlation_rows(scored, factors, threshold=correlation_threshold)
    by_factor = _by_factor_summary(
        factors,
        factor_date=by_factor_date,
        drift=drift,
        style_exposure=style_exposure,
        residual_ic=residual_ic,
        size_bucket=size_bucket,
        correlation=correlation,
    )
    return {
        "by_factor": by_factor,
        "by_factor_date": by_factor_date,
        "style_exposure": style_exposure,
        "size_bucket": size_bucket,
        "industry": industry,
        "residual_ic": residual_ic,
        "correlation": correlation,
        "drift": drift,
    }


def _result_from_frames(
    summary: dict[str, Any],
    *,
    by_factor: pd.DataFrame,
    by_factor_date: pd.DataFrame,
    style_exposure: pd.DataFrame,
    size_bucket: pd.DataFrame,
    industry: pd.DataFrame,
    residual_ic: pd.DataFrame,
    correlation: pd.DataFrame,
    drift: pd.DataFrame,
) -> FactorDiagnosticsResult:
    return FactorDiagnosticsResult(
        summary=summary,
        by_factor=by_factor,
        by_factor_date=by_factor_date,
        style_exposure=style_exposure,
        size_bucket=size_bucket,
        industry=industry,
        residual_ic=residual_ic,
        correlation=correlation,
        drift=drift,
    )


def _empty_result(status: str, **extra: Any) -> FactorDiagnosticsResult:
    summary = {"status": status, "factors": 0, "warnings": []}
    summary.update(extra)
    empty = pd.DataFrame()
    return FactorDiagnosticsResult(
        summary=summary,
        by_factor=empty,
        by_factor_date=empty,
        style_exposure=empty,
        size_bucket=empty,
        industry=empty,
        residual_ic=empty,
        correlation=empty,
        drift=empty,
    )


def _normalize_scored(
    scored: pd.DataFrame | None,
    *,
    date_col: str,
    symbol_col: str,
) -> pd.DataFrame:
    if scored is None or scored.empty:
        return pd.DataFrame()
    if date_col not in scored.columns or symbol_col not in scored.columns:
        return pd.DataFrame()
    out = scored.copy()
    if date_col != "trade_date":
        out = out.rename(columns={date_col: "trade_date"})
    if symbol_col != "symbol":
        out = out.rename(columns={symbol_col: "symbol"})
    out = canonicalize_symbol_columns(out, context="factor diagnostics scored data")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.strip()
    return out.dropna(subset=["trade_date", "symbol"]).copy()


def _resolve_factor_columns(
    scored: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None,
    feature_importance: pd.DataFrame | None,
    target_col: str,
    top_n: int,
) -> list[str]:
    candidates: list[str] = []
    has_importance = (
        feature_importance is not None
        and not feature_importance.empty
        and "feature" in feature_importance
    )
    if has_importance:
        importance = feature_importance.copy()
        if "importance" in importance:
            importance = importance.sort_values("importance", ascending=False)
        candidates.extend(importance["feature"].astype(str).tolist())
    if feature_columns:
        candidates.extend(str(column) for column in feature_columns)

    numeric_cols = set(scored.select_dtypes(include=[np.number]).columns)
    excluded = {"trade_date", "symbol", target_col}
    resolved: list[str] = []
    seen: set[str] = set()
    for column in candidates:
        if column in seen or column in excluded or column not in numeric_cols:
            continue
        seen.add(column)
        resolved.append(column)
        if len(resolved) >= top_n:
            break
    return resolved


def _existing_numeric_columns(scored: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    numeric_cols = set(scored.select_dtypes(include=[np.number]).columns)
    seen: set[str] = set()
    out: list[str] = []
    for column in columns:
        if column in numeric_cols and column not in seen:
            seen.add(column)
            out.append(column)
    return out


def _first_existing(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    present = set(columns)
    return next((column for column in candidates if column in present), None)


def _resolve_market_cap_col(columns: Sequence[str], configured: str | None) -> str | None:
    if configured and configured in columns:
        return configured
    return _first_existing(columns, ("log_mkt_cap", "market_cap", "mkt_cap", "size"))


def _input_warnings(
    *,
    style_cols: Sequence[str],
    industry_col: str | None,
    cap_col: str | None,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if not style_cols:
        warnings.append({"type": "missing_style_columns", "message": "No style columns found."})
    if industry_col is None:
        warnings.append(
            {"type": "missing_industry_column", "message": "Industry diagnostics skipped."}
        )
    if cap_col is None:
        warnings.append(
            {
                "type": "missing_market_cap",
                "message": "Size bucket diagnostics skipped.",
            }
        )
    return warnings


def _summary(
    *,
    status: str,
    scored: pd.DataFrame,
    factors: Sequence[str],
    target_col: str,
    style_cols: Sequence[str],
    industry_col: str | None,
    cap_col: str | None,
    warnings: list[dict[str, Any]],
    residual_ic: pd.DataFrame,
    size_bucket: pd.DataFrame,
    correlation: pd.DataFrame,
    correlation_threshold: float,
    min_obs: int,
    min_bucket_obs: int,
) -> dict[str, Any]:
    style_dominated = 0
    if not residual_ic.empty:
        grouped = residual_ic.groupby("factor", sort=False)
        for _, group in grouped:
            r2 = _column_mean(group, "style_r2")
            ratio = abs(_column_mean(group, "residual_ic_ratio"))
            if np.isfinite(r2) and r2 >= 0.70 and (not np.isfinite(ratio) or ratio < 0.50):
                style_dominated += 1
    return {
        "schema_version": 1,
        "artifact_type": "alpha_research.factor_diagnostics",
        "status": status,
        "rows": len(scored),
        "dates": int(scored["trade_date"].nunique()),
        "factors": len(factors),
        "factor_list": list(factors),
        "target_col": target_col,
        "style_columns": list(style_cols),
        "industry_column": industry_col,
        "market_cap_col": cap_col,
        "min_obs": int(min_obs),
        "min_bucket_obs": int(min_bucket_obs),
        "correlation_threshold": float(correlation_threshold),
        "residual_ic_available": bool(not residual_ic.empty),
        "size_bucket_available": bool(not size_bucket.empty),
        "style_dominated_feature_count": int(style_dominated),
        "high_redundancy_pair_count": int(correlation["is_high_corr"].sum())
        if not correlation.empty
        else 0,
        "warnings": warnings,
    }
