from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from market_data_platform.symbols import canonicalize_symbol_columns

from .factor_diagnostics_config import (
    DEFAULT_AUTOCORR_LAGS,
    DEFAULT_INDUSTRY_COLUMNS,
    DEFAULT_SIZE_BUCKET_LABELS,
    DEFAULT_STYLE_COLUMNS,
    FactorDiagnosticsResult,
    factor_diagnostics_options_from_config as factor_diagnostics_options_from_config,
)
from .factor_diagnostics_math import (
    bucket_labels as _bucket_labels,
    column_mean as _column_mean,
    column_min as _column_min,
    date_text as _date_text,
    dominant_style as _dominant_style,
    factor_correlation_rows as _factor_correlation_rows,
    ir as _ir,
    long_short_return as _long_short_return,
    max_abs_group_mean as _max_abs_group_mean,
    r2_score as _r2,
    safe_mean as _safe_mean,
    safe_ratio as _safe_ratio,
    safe_std as _safe_std,
    size_bucket_ic_spread as _size_bucket_ic_spread,
    size_buckets as _size_buckets,
    spearman as _spearman,
    zscore as _zscore,
)


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
    industry_col = _first_existing(scored.columns, industry_columns)
    cap_col = _resolve_market_cap_col(scored.columns, market_cap_col)
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


def _factor_date_rows(
    scored: pd.DataFrame,
    factors: Sequence[str],
    target_col: str,
    *,
    min_obs: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    date_sizes = scored.groupby("trade_date")["symbol"].nunique()
    for date, group in scored.groupby("trade_date", sort=True):
        total = int(date_sizes.loc[date])
        target = pd.to_numeric(group[target_col], errors="coerce")
        for factor in factors:
            values = pd.to_numeric(group[factor], errors="coerce")
            valid_factor = values.notna()
            valid = valid_factor & target.notna()
            rows.append(
                {
                    "trade_date": _date_text(date),
                    "factor": factor,
                    "n_obs": int(valid.sum()),
                    "coverage": float(valid_factor.sum() / total) if total else np.nan,
                    "raw_rank_ic": _spearman(values.loc[valid], target.loc[valid])
                    if int(valid.sum()) >= min_obs
                    else np.nan,
                    "factor_mean": _safe_mean(values),
                    "factor_std": _safe_std(values),
                }
            )
    return pd.DataFrame(rows)


def _drift_rows(
    scored: pd.DataFrame,
    factors: Sequence[str],
    *,
    lags: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not lags:
        return pd.DataFrame()
    dates = sorted(scored["trade_date"].dropna().unique())
    by_date = {date: group.set_index("symbol") for date, group in scored.groupby("trade_date")}
    for date_idx, date in enumerate(dates):
        current = by_date[date]
        for lag in lags:
            if date_idx - lag < 0:
                continue
            previous_date = dates[date_idx - lag]
            previous = by_date[previous_date]
            symbols = current.index.intersection(previous.index)
            if len(symbols) < 2:
                continue
            for factor in factors:
                current_z = _zscore(current.loc[symbols, factor])
                previous_z = _zscore(previous.loc[symbols, factor])
                valid = current_z.notna() & previous_z.notna()
                rows.append(
                    {
                        "trade_date": _date_text(date),
                        "previous_trade_date": _date_text(previous_date),
                        "factor": factor,
                        "lag": int(lag),
                        "n_obs": int(valid.sum()),
                        "rank_autocorr": _spearman(
                            current_z.loc[valid],
                            previous_z.loc[valid],
                        ),
                        "delta_z_std": _safe_std(current_z.loc[valid] - previous_z.loc[valid]),
                    }
                )
    return pd.DataFrame(rows)


def _exposure_and_residual_rows(
    scored: pd.DataFrame,
    factors: Sequence[str],
    target_col: str,
    style_cols: Sequence[str],
    *,
    industry_col: str | None,
    min_obs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposure_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    if not style_cols and industry_col is None:
        return pd.DataFrame(), pd.DataFrame()

    for date, group in scored.groupby("trade_date", sort=True):
        target = pd.to_numeric(group[target_col], errors="coerce")
        for factor in factors:
            y = _zscore(group[factor])
            valid = y.notna() & target.notna()
            for col in style_cols:
                style_values = pd.to_numeric(group[col], errors="coerce")
                style_valid = valid & style_values.notna()
                rank_corr = (
                    _spearman(y.loc[style_valid], style_values.loc[style_valid])
                    if int(style_valid.sum()) >= min_obs
                    else np.nan
                )
                exposure_rows.append(
                    {
                        "trade_date": _date_text(date),
                        "factor": factor,
                        "style": col,
                        "rank_corr": rank_corr,
                        "beta": np.nan,
                        "r2": np.nan,
                        "n_obs": int(style_valid.sum()),
                    }
                )

            regression = _fit_exposure_model(
                group,
                y,
                target,
                style_cols,
                industry_col=industry_col,
                min_obs=min_obs,
            )
            if regression is None:
                continue
            betas, r2, residual, regression_valid, neutralizer_count = regression
            for row in exposure_rows[-len(style_cols) :]:
                if row["factor"] == factor and row["trade_date"] == _date_text(date):
                    row["beta"] = betas.get(str(row["style"]), np.nan)
                    row["r2"] = r2
                    row["n_obs"] = int(regression_valid.sum())
            raw_ic = _spearman(y.loc[regression_valid], target.loc[regression_valid])
            residual_ic = _spearman(residual, target.loc[regression_valid])
            residual_rows.append(
                {
                    "trade_date": _date_text(date),
                    "factor": factor,
                    "raw_rank_ic": raw_ic,
                    "residual_rank_ic": residual_ic,
                    "residual_ic_ratio": _safe_ratio(residual_ic, raw_ic),
                    "style_r2": r2,
                    "n_obs": int(regression_valid.sum()),
                    "neutralizer_count": neutralizer_count,
                }
            )
    return pd.DataFrame(exposure_rows), pd.DataFrame(residual_rows)


def _fit_exposure_model(
    group: pd.DataFrame,
    y: pd.Series,
    target: pd.Series,
    style_cols: Sequence[str],
    *,
    industry_col: str | None,
    min_obs: int,
) -> tuple[dict[str, float], float, pd.Series, pd.Series, int] | None:
    parts: list[pd.DataFrame] = []
    style_frame = pd.DataFrame(index=group.index)
    for col in style_cols:
        style_frame[col] = _zscore(group[col])
    if not style_frame.empty:
        parts.append(style_frame)
    if industry_col and industry_col in group.columns:
        dummies = pd.get_dummies(group[industry_col].astype(str), prefix=industry_col)
        if dummies.shape[1] > 1:
            parts.append(dummies.iloc[:, 1:].astype(float))
    if not parts:
        return None
    x_frame = pd.concat(parts, axis=1)
    valid = y.notna() & target.notna() & x_frame.notna().all(axis=1)
    if int(valid.sum()) < min_obs:
        return None
    x_values = x_frame.loc[valid].to_numpy(dtype=float)
    if x_values.shape[0] <= x_values.shape[1] + 1:
        return None
    x_design = np.column_stack([np.ones(x_values.shape[0]), x_values])
    y_values = y.loc[valid].to_numpy(dtype=float)
    try:
        coef, *_ = np.linalg.lstsq(x_design, y_values, rcond=None)
    except np.linalg.LinAlgError:
        return None
    fitted = x_design @ coef
    residual_values = y_values - fitted
    r2 = _r2(y_values, fitted)
    betas = {
        str(col): float(coef[idx + 1])
        for idx, col in enumerate(x_frame.columns)
        if str(col) in style_cols
    }
    residual = pd.Series(residual_values, index=group.index[valid])
    return betas, r2, residual, valid, int(x_frame.shape[1])


def _size_bucket_rows(
    scored: pd.DataFrame,
    factors: Sequence[str],
    target_col: str,
    cap_col: str | None,
    *,
    bucket_count: int,
    labels: Sequence[str],
    min_obs: int,
    min_bucket_obs: int,
) -> pd.DataFrame:
    if cap_col is None:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    bucket_labels = _bucket_labels(bucket_count, labels)
    for date, group in scored.groupby("trade_date", sort=True):
        buckets = _size_buckets(group[cap_col], bucket_count=bucket_count, labels=bucket_labels)
        target = pd.to_numeric(group[target_col], errors="coerce")
        total = int(group["symbol"].nunique())
        for factor in factors:
            factor_z = _zscore(group[factor])
            for bucket in bucket_labels:
                mask = buckets == bucket
                valid = mask & factor_z.notna() & target.notna()
                rows.append(
                    {
                        "trade_date": _date_text(date),
                        "factor": factor,
                        "size_bucket": bucket,
                        "n_obs": int(valid.sum()),
                        "coverage": float(mask.sum() / total) if total else np.nan,
                        "factor_z_mean": _safe_mean(factor_z.loc[mask]),
                        "rank_ic": _spearman(factor_z.loc[valid], target.loc[valid])
                        if int(valid.sum()) >= min_obs
                        else np.nan,
                        "long_short_return": _long_short_return(
                            factor_z.loc[valid],
                            target.loc[valid],
                            min_obs=min_bucket_obs,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _industry_rows(
    scored: pd.DataFrame,
    factors: Sequence[str],
    target_col: str,
    industry_col: str | None,
    *,
    min_bucket_obs: int,
) -> pd.DataFrame:
    if industry_col is None:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for date, group in scored.groupby("trade_date", sort=True):
        target = pd.to_numeric(group[target_col], errors="coerce")
        total = int(group["symbol"].nunique())
        for factor in factors:
            factor_z = _zscore(group[factor])
            for industry, segment in group.groupby(industry_col, sort=True):
                idx = segment.index
                valid = factor_z.loc[idx].notna() & target.loc[idx].notna()
                rows.append(
                    {
                        "trade_date": _date_text(date),
                        "factor": factor,
                        "industry_column": industry_col,
                        "industry": str(industry),
                        "n_obs": int(valid.sum()),
                        "coverage": float(len(segment) / total) if total else np.nan,
                        "factor_z_mean": _safe_mean(factor_z.loc[idx]),
                        "rank_ic": _spearman(
                            factor_z.loc[idx].loc[valid],
                            target.loc[idx].loc[valid],
                        )
                        if int(valid.sum()) >= min_bucket_obs
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _correlation_rows(
    scored: pd.DataFrame,
    factors: Sequence[str],
    *,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    latest_date = scored["trade_date"].max()
    for left_idx, left in enumerate(factors):
        for right in factors[left_idx + 1 :]:
            values: list[float] = []
            latest = np.nan
            for date, group in scored.groupby("trade_date", sort=True):
                corr = _spearman(group[left], group[right])
                if np.isfinite(corr):
                    values.append(corr)
                if date == latest_date:
                    latest = corr
            corr_mean = float(np.mean(values)) if values else np.nan
            corr_abs_mean = float(np.mean(np.abs(values))) if values else np.nan
            rows.append(
                {
                    "factor_a": left,
                    "factor_b": right,
                    "corr_mean": corr_mean,
                    "corr_abs_mean": corr_abs_mean,
                    "corr_latest": latest,
                    "threshold": float(threshold),
                    "is_high_corr": bool(np.isfinite(corr_abs_mean) and corr_abs_mean >= threshold),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    clusters = _correlation_clusters(list(factors), out)
    sizes = {
        cluster_id: sum(1 for value in clusters.values() if value == cluster_id)
        for cluster_id in set(clusters.values())
    }
    out["cluster_id"] = [
        clusters[row.factor_a] if bool(row.is_high_corr) else np.nan
        for row in out.itertuples(index=False)
    ]
    out["cluster_size"] = [
        sizes.get(clusters[row.factor_a], 1) if bool(row.is_high_corr) else np.nan
        for row in out.itertuples(index=False)
    ]
    return out


def _correlation_clusters(factors: list[str], pairs: pd.DataFrame) -> dict[str, int]:
    parent = {factor: factor for factor in factors}

    def find(factor: str) -> str:
        while parent[factor] != factor:
            parent[factor] = parent[parent[factor]]
            factor = parent[factor]
        return factor

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in pairs.itertuples(index=False):
        if bool(row.is_high_corr):
            union(str(row.factor_a), str(row.factor_b))
    root_ids = {root: idx + 1 for idx, root in enumerate(sorted({find(f) for f in factors}))}
    return {factor: root_ids[find(factor)] for factor in factors}


def _by_factor_summary(
    factors: Sequence[str],
    *,
    factor_date: pd.DataFrame,
    drift: pd.DataFrame,
    style_exposure: pd.DataFrame,
    residual_ic: pd.DataFrame,
    size_bucket: pd.DataFrame,
    correlation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for factor in factors:
        date_rows = (
            factor_date.loc[factor_date["factor"] == factor]
            if not factor_date.empty
            else pd.DataFrame()
        )
        drift_rows = drift.loc[drift["factor"] == factor] if not drift.empty else pd.DataFrame()
        residual_rows = (
            residual_ic.loc[residual_ic["factor"] == factor]
            if not residual_ic.empty
            else pd.DataFrame()
        )
        exposure_rows = (
            style_exposure.loc[style_exposure["factor"] == factor]
            if not style_exposure.empty
            else pd.DataFrame()
        )
        size_rows = (
            size_bucket.loc[size_bucket["factor"] == factor]
            if not size_bucket.empty
            else pd.DataFrame()
        )
        corr_rows = _factor_correlation_rows(correlation, factor)
        row = {
            "factor": factor,
            "coverage_mean": _column_mean(date_rows, "coverage"),
            "coverage_min": _column_min(date_rows, "coverage"),
            "raw_ic_mean": _column_mean(date_rows, "raw_rank_ic"),
            "raw_ic_ir": _ir(date_rows.get("raw_rank_ic", pd.Series(dtype=float))),
            "residual_ic_mean": _column_mean(residual_rows, "residual_rank_ic"),
            "residual_ic_ir": _ir(residual_rows.get("residual_rank_ic", pd.Series(dtype=float))),
            "residual_ic_ratio_mean": _column_mean(residual_rows, "residual_ic_ratio"),
            "style_r2_mean": _column_mean(residual_rows, "style_r2"),
            "dominant_style": _dominant_style(exposure_rows),
            "max_abs_style_rank_corr_mean": _max_abs_group_mean(
                exposure_rows,
                group_col="style",
                value_col="rank_corr",
            ),
            "size_bucket_ic_spread": _size_bucket_ic_spread(size_rows),
            "high_corr_peer_count": int(corr_rows["is_high_corr"].sum())
            if not corr_rows.empty
            else 0,
        }
        for lag in sorted(drift_rows["lag"].dropna().unique()) if not drift_rows.empty else []:
            lag_rows = drift_rows.loc[drift_rows["lag"] == lag]
            row[f"rank_autocorr_{int(lag)}"] = _column_mean(lag_rows, "rank_autocorr")
            row[f"delta_z_std_{int(lag)}"] = _column_mean(lag_rows, "delta_z_std")
        rows.append(row)
    return pd.DataFrame(rows)


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
