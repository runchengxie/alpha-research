from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

DEFAULT_STYLE_COLUMNS = (
    "log_mkt_cap",
    "market_cap",
    "size",
    "bm",
    "pb",
    "pe_ttm",
    "turnover_rate",
    "volatility_20d",
    "beta_120",
    "ret_20",
    "ret_60",
    "ret_20d",
    "momentum_20d",
)
DEFAULT_INDUSTRY_COLUMNS = (
    "first_industry_name",
    "sw_l1_name",
    "citic_l1_name",
    "industry",
)
DEFAULT_SIZE_BUCKET_LABELS = ("small", "mid", "large")
DEFAULT_AUTOCORR_LAGS = (1, 5, 20)


@dataclass(frozen=True)
class FactorDiagnosticsResult:
    summary: dict[str, Any]
    by_factor: pd.DataFrame
    by_factor_date: pd.DataFrame
    style_exposure: pd.DataFrame
    size_bucket: pd.DataFrame
    industry: pd.DataFrame
    residual_ic: pd.DataFrame
    correlation: pd.DataFrame
    drift: pd.DataFrame


def factor_diagnostics_options_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = _factor_diagnostics_config(config)
    if raw is False:
        return {"enabled": False}
    if raw is True or raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        return {"enabled": True}

    raw_any = cast(dict[str, Any], raw)
    size_cfg = cast(dict[str, Any], raw.get("size_buckets")) if isinstance(raw.get("size_buckets"), Mapping) else {}
    neutralize_cfg = cast(dict[str, Any], raw.get("neutralize")) if isinstance(raw.get("neutralize"), Mapping) else {}
    corr_cfg = cast(dict[str, Any], raw.get("correlation")) if isinstance(raw.get("correlation"), Mapping) else {}
    drift_cfg = cast(dict[str, Any], raw.get("drift")) if isinstance(raw.get("drift"), Mapping) else {}

    options: dict[str, Any] = {
        "enabled": bool(raw_any.get("enabled", True)),
        "top_n": int(raw_any.get("top_n", raw_any.get("max_features", 30)) or 30),
        "target_col": str(raw_any.get("target_col") or "future_return"),
        "market_cap_col": _optional_str(raw_any.get("market_cap_col")),
        "style_columns": _string_list(raw_any.get("style_columns"), DEFAULT_STYLE_COLUMNS),
        "industry_columns": _string_list(raw_any.get("industry_columns"), DEFAULT_INDUSTRY_COLUMNS),
        "min_obs": int(raw_any.get("min_obs", neutralize_cfg.get("min_obs", 20)) or 20),
        "min_bucket_obs": int(raw_any.get("min_bucket_obs", 10) or 10),
        "size_bucket_count": int(size_cfg.get("count", raw_any.get("size_bucket_count", 3)) or 3),
        "size_bucket_labels": _string_list(
            size_cfg.get("labels"),
            DEFAULT_SIZE_BUCKET_LABELS,
        ),
        "include_industry_neutralization": bool(
            neutralize_cfg.get("include_industry", raw_any.get("include_industry", True))
        ),
        "correlation_threshold": float(
            corr_cfg.get("threshold", raw_any.get("correlation_threshold", 0.90)) or 0.90
        ),
        "autocorr_lags": tuple(
            int(value)
            for value in _string_list(
                drift_cfg.get("autocorr_lags", raw_any.get("autocorr_lags")),
                tuple(str(value) for value in DEFAULT_AUTOCORR_LAGS),
            )
            if int(value) > 0
        ),
    }
    if "max_features" in corr_cfg:
        options["top_n"] = min(options["top_n"], int(corr_cfg["max_features"]))
    return options


def _factor_diagnostics_config(config: Mapping[str, Any]) -> object:
    eval_cfg = config.get("eval")
    if isinstance(eval_cfg, Mapping) and "factor_diagnostics" in eval_cfg:
        return eval_cfg.get("factor_diagnostics")
    return config.get("factor_diagnostics", {})


def _optional_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: object, default: Sequence[str]) -> list[str]:
    if value is None:
        return [str(item) for item in default]
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(item) for item in default]
