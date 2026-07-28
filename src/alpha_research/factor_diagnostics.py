"""Factor diagnostics orchestration.

This module is a thin public surface for the factor diagnostics implementation.
The historical single-file implementation has been split into private
submodules (``_factor_diagnostics_core`` / ``_factor_diagnostics_rows``) plus the
pre-existing ``factor_diagnostics_config`` / ``factor_diagnostics_math``
helpers, to keep individual files smaller while preserving the exact public and
private symbol surface. Everything below is re-exported so existing
``alpha_research.factor_diagnostics`` imports keep working unchanged.
"""

from __future__ import annotations

from ._factor_diagnostics_core import (
    _diagnostic_frames,
    _empty_result,
    _existing_numeric_columns,
    _first_existing,
    _input_warnings,
    _normalize_scored,
    _resolve_factor_columns,
    _resolve_market_cap_col,
    _result_from_frames,
    _summary,
    compute_factor_diagnostics,
)
from ._factor_diagnostics_rows import (
    _by_factor_summary,
    _correlation_clusters,
    _correlation_rows,
    _drift_rows,
    _exposure_and_residual_rows,
    _factor_date_rows,
    _fit_exposure_model,
    _industry_rows,
    _size_bucket_rows,
)
from .factor_diagnostics_config import (
    DEFAULT_AUTOCORR_LAGS,
    DEFAULT_INDUSTRY_COLUMNS,
    DEFAULT_SIZE_BUCKET_LABELS,
    DEFAULT_STYLE_COLUMNS,
    FactorDiagnosticsResult,
    factor_diagnostics_options_from_config,
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

__all__ = [
    "DEFAULT_AUTOCORR_LAGS",
    "DEFAULT_INDUSTRY_COLUMNS",
    "DEFAULT_SIZE_BUCKET_LABELS",
    "DEFAULT_STYLE_COLUMNS",
    "FactorDiagnosticsResult",
    "_bucket_labels",
    "_by_factor_summary",
    "_column_mean",
    "_column_min",
    "_correlation_clusters",
    "_correlation_rows",
    "_date_text",
    "_diagnostic_frames",
    "_dominant_style",
    "_drift_rows",
    "_empty_result",
    "_existing_numeric_columns",
    "_exposure_and_residual_rows",
    "_factor_correlation_rows",
    "_factor_date_rows",
    "_first_existing",
    "_fit_exposure_model",
    "_industry_rows",
    "_input_warnings",
    "_ir",
    "_long_short_return",
    "_max_abs_group_mean",
    "_normalize_scored",
    "_r2",
    "_resolve_factor_columns",
    "_resolve_market_cap_col",
    "_result_from_frames",
    "_safe_mean",
    "_safe_ratio",
    "_safe_std",
    "_size_bucket_ic_spread",
    "_size_bucket_rows",
    "_size_buckets",
    "_spearman",
    "_summary",
    "_zscore",
    "compute_factor_diagnostics",
    "factor_diagnostics_options_from_config",
]
