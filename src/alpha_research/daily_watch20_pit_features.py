"""PIT-safe quality and growth feature construction for research shadows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

FUNDAMENTAL_FEATURE_SCHEMA = "daily_watch20.fundamental_features.research.v1"
PIT_MAX_OBSERVATION_AGE_DAYS = 3
PIT_MAX_REPORT_AGE_DAYS = 250

QUALITY_FEATURES = (
    "pit_quality_roa_pct",
    "pit_quality_gross_margin_pct",
    "pit_quality_cfo_to_assets_pct",
    "pit_quality_negative_accrual_pct",
)
GROWTH_FEATURES = (
    "pit_growth_revenue_yoy_pct",
    "pit_growth_netprofit_yoy_pct",
)
PIT_SOURCE_FIELDS = (
    "roa",
    "grossprofit_margin",
    "n_cashflow_act",
    "n_income_attr_p",
    "total_assets",
    "or_yoy",
    "netprofit_yoy",
)
PIT_LINEAGE_SUFFIXES = (
    "report_period",
    "available_date",
    "disclosure_date",
    "source_dataset",
    "source_raw_asset",
    "source_run_id",
    "source_retrieved_at",
    "source_bundle_retrieval_start_date",
    "source_bundle_available_date",
    "revision_id",
)


@dataclass(frozen=True)
class FundamentalFeaturePanel:
    """DailyWatch20 frame enriched only from audited PIT as-of states."""

    frame: pd.DataFrame
    coverage_daily: pd.DataFrame
    receipt: dict[str, Any]


_FINANCIAL_INDUSTRY_PATTERN = r"(?:银行|保险|证券|金融|信托)"


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _date_key(value: object) -> str:
    date = cast(pd.Timestamp, pd.Timestamp(str(value)))
    if pd.isna(date):
        raise ValueError("fundamental shadow contains an invalid date")
    return date.strftime("%Y%m%d")


def _snapshot_parts(value: Any) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    frame = getattr(value, "frame", None)
    audit = getattr(value, "audit", None)
    if isinstance(value, pd.DataFrame):
        frame = value
        audit = value.attrs.get("pit_audit")
    if not isinstance(frame, pd.DataFrame) or not isinstance(audit, Mapping):
        raise TypeError("PIT as-of snapshot must expose a DataFrame and audit mapping")
    return frame, audit


def _validate_snapshot_audit(audit: Mapping[str, Any], *, as_of_date: str) -> dict[str, Any]:
    audit_date = _date_key(audit.get("as_of_date"))
    bundle_available = _date_key(audit.get("bundle_available_date"))
    if audit_date != as_of_date:
        raise ValueError(
            f"PIT as-of audit date mismatch: expected={as_of_date}, actual={audit_date}"
        )
    if bundle_available > as_of_date:
        raise ValueError(f"PIT bundle vintage is after as_of_date {as_of_date}")
    if audit.get("provenance_policy") != "require_observed":
        raise ValueError("PIT fundamental shadow requires observed source provenance")
    if audit.get("revision_safe") is not True:
        raise ValueError("PIT fundamental shadow requires revision_safe=true")
    if audit.get("freshness_verified") is not True:
        raise ValueError("PIT fundamental shadow requires freshness_verified=true")
    if audit.get("production_eligible") is not True:
        raise ValueError("PIT fundamental shadow requires production_eligible=true")
    try:
        configured_age = int(cast(Any, audit.get("max_observation_age_days")))
        observation_age = int(cast(Any, audit.get("observation_age_days")))
    except (TypeError, ValueError) as exc:
        raise ValueError("PIT observation freshness ages must be explicit integers") from exc
    if configured_age != PIT_MAX_OBSERVATION_AGE_DAYS:
        raise ValueError("PIT max_observation_age_days must equal the frozen three-day contract")
    if not 0 <= observation_age <= PIT_MAX_OBSERVATION_AGE_DAYS:
        raise ValueError("PIT observation_age_days exceeds the frozen three-day contract")
    oldest_retrieval = _date_key(audit.get("oldest_component_retrieval_date"))
    calculated_age = (pd.Timestamp(as_of_date) - pd.Timestamp(oldest_retrieval)).days
    if calculated_age != observation_age:
        raise ValueError("PIT observation_age_days does not match oldest bundle retrieval")
    if audit.get("missing_bundle_observation_proofs", []) != []:
        raise ValueError("PIT audit is missing complete-bundle observation proofs")
    return {
        "bundle_available_date": bundle_available,
        "oldest_component_retrieval_date": oldest_retrieval,
        "observation_age_days": observation_age,
        "max_observation_age_days": configured_age,
        "latest_observed_vintage_by_source": audit.get("latest_observed_vintage_by_source", {}),
        "missing_observation_sources": audit.get("missing_observation_sources", []),
    }


def _required_state_columns() -> set[str]:
    columns = {"symbol", "as_of_date"}
    for field in PIT_SOURCE_FIELDS:
        columns.add(field)
        columns.update(f"{field}__{suffix}" for suffix in PIT_LINEAGE_SUFFIXES)
    return columns


def _validate_snapshot_frame(frame: pd.DataFrame, *, as_of_date: str) -> pd.DataFrame:
    missing = sorted(_required_state_columns() - set(frame.columns))
    if missing:
        raise ValueError(f"PIT as-of state is missing fields or lineage: {missing}")
    state = frame.loc[:, sorted(_required_state_columns())].copy()
    if state.duplicated("symbol").any():
        raise ValueError(f"PIT as-of state contains duplicate symbols: {as_of_date}")
    dates = _series(state, "as_of_date").map(_date_key)
    if not dates.eq(as_of_date).all():
        raise ValueError(f"PIT state rows are not exact-date as-of rows: {as_of_date}")
    for field in PIT_SOURCE_FIELDS:
        present = _series(state, field).notna()
        for suffix in PIT_LINEAGE_SUFFIXES:
            lineage = _series(state, f"{field}__{suffix}").loc[present].astype(str).str.strip()
            if lineage.eq("").any():
                raise ValueError(f"PIT field {field} lacks {suffix} lineage: {as_of_date}")
        available = _series(state, f"{field}__available_date").loc[present].map(_date_key)
        if available.gt(as_of_date).any():
            raise ValueError(f"PIT field {field} is visible before available_date")
        retrieved = pd.to_datetime(
            _series(state, f"{field}__source_retrieved_at").loc[present],
            errors="coerce",
            utc=True,
        )
        if retrieved.isna().any():
            raise ValueError(f"PIT field {field} has invalid retrieval lineage")
        retrieved_dates = retrieved.dt.tz_convert(None).dt.strftime("%Y%m%d")
        if retrieved_dates.gt(as_of_date).any():
            raise ValueError(f"PIT field {field} uses a future source retrieval")
        bundle_start = (
            _series(state, f"{field}__source_bundle_retrieval_start_date")
            .loc[present]
            .map(_date_key)
        )
        bundle_available = (
            _series(state, f"{field}__source_bundle_available_date").loc[present].map(_date_key)
        )
        if bundle_start.gt(bundle_available).any():
            raise ValueError(f"PIT field {field} has an invalid complete-bundle interval")
        if bundle_available.gt(as_of_date).any():
            raise ValueError(f"PIT field {field} is visible before bundle completion")
    return state


def _namespace_state(state: pd.DataFrame) -> pd.DataFrame:
    rename = {
        column: f"pit_{column}"
        for column in state.columns
        if column not in {"symbol", "as_of_date"}
    }
    return state.rename(columns=rename).drop(columns="as_of_date")


def _aligned_raw_feature(
    frame: pd.DataFrame,
    fields: Sequence[str],
    values: pd.Series,
    *,
    max_report_age_days: int,
) -> tuple[pd.Series, pd.Series]:
    report_periods = pd.concat(
        [
            pd.to_datetime(_series(frame, f"pit_{field}__report_period"), errors="coerce")
            for field in fields
        ],
        axis=1,
    )
    aligned = report_periods.notna().all(axis=1) & report_periods.nunique(axis=1).eq(1)
    report_period = cast(pd.Series, report_periods.iloc[:, 0])
    age_days = (_series(frame, "trade_date") - report_period).dt.days
    fresh = age_days.between(0, max_report_age_days, inclusive="both")
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    observed = aligned & fresh & numeric.notna()
    return numeric.where(observed), observed


def _direct_raw_feature(
    frame: pd.DataFrame,
    field: str,
    *,
    max_report_age_days: int,
) -> tuple[pd.Series, pd.Series]:
    return _aligned_raw_feature(
        frame,
        (field,),
        _series(frame, f"pit_{field}"),
        max_report_age_days=max_report_age_days,
    )


def _rank_feature(
    frame: pd.DataFrame,
    raw: pd.Series,
    observed: pd.Series,
    *,
    financial: pd.Series,
    neutralize_financials: bool,
) -> tuple[pd.Series, pd.Series]:
    comparable = observed & (~financial if neutralize_financials else True)
    ranked = raw.where(comparable).groupby(_series(frame, "trade_date"), sort=False).rank(pct=True)
    if neutralize_financials:
        ranked = ranked.mask(financial, 0.5)
    return ranked, comparable


def _attach_one_snapshot(
    daily: pd.DataFrame,
    state: pd.DataFrame,
    *,
    as_of_date: str,
) -> pd.DataFrame:
    date = pd.Timestamp(as_of_date)
    rows = cast(pd.DataFrame, daily.loc[_series(daily, "trade_date").eq(date)]).copy()
    return rows.merge(_namespace_state(state), on="symbol", how="left", validate="many_to_one")


def _raw_feature_specs(
    frame: pd.DataFrame,
    *,
    max_report_age_days: int,
) -> dict[str, tuple[pd.Series, pd.Series, bool]]:
    def direct(field: str) -> tuple[pd.Series, pd.Series]:
        return _direct_raw_feature(frame, field, max_report_age_days=max_report_age_days)

    assets = pd.to_numeric(_series(frame, "pit_total_assets"), errors="coerce").where(
        lambda values: values > 0
    )
    cfo = pd.to_numeric(_series(frame, "pit_n_cashflow_act"), errors="coerce")
    profit = pd.to_numeric(_series(frame, "pit_n_income_attr_p"), errors="coerce")
    cfo_assets = _aligned_raw_feature(
        frame,
        ("n_cashflow_act", "total_assets"),
        cfo / assets,
        max_report_age_days=max_report_age_days,
    )
    negative_accrual = _aligned_raw_feature(
        frame,
        ("n_income_attr_p", "n_cashflow_act", "total_assets"),
        (cfo - profit) / assets,
        max_report_age_days=max_report_age_days,
    )
    return {
        QUALITY_FEATURES[0]: (*direct("roa"), True),
        QUALITY_FEATURES[1]: (*direct("grossprofit_margin"), True),
        QUALITY_FEATURES[2]: (*cfo_assets, True),
        QUALITY_FEATURES[3]: (*negative_accrual, True),
        GROWTH_FEATURES[0]: (*direct("or_yoy"), False),
        GROWTH_FEATURES[1]: (*direct("netprofit_yoy"), False),
    }


def _attach_ranked_features(
    frame: pd.DataFrame,
    raw_specs: Mapping[str, tuple[pd.Series, pd.Series, bool]],
    *,
    financial: pd.Series,
) -> pd.DataFrame:
    out = frame.copy()
    observed_by_pack: dict[str, list[pd.Series]] = {"quality": [], "growth": []}
    for feature, (raw, observed, neutralize_financials) in raw_specs.items():
        ranked, comparable = _rank_feature(
            out,
            raw,
            observed,
            financial=financial,
            neutralize_financials=neutralize_financials,
        )
        out[feature] = ranked
        out[f"{feature}__observed"] = comparable
        pack = "quality" if feature in QUALITY_FEATURES else "growth"
        observed_by_pack[pack].append(comparable)
    out["pit_financial_sector_neutralized"] = financial
    out["pit_quality_observed_count"] = pd.concat(observed_by_pack["quality"], axis=1).sum(axis=1)
    out["pit_growth_observed_count"] = pd.concat(observed_by_pack["growth"], axis=1).sum(axis=1)
    return out


def _feature_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = (
        _series(frame, "hard_eligible").astype(bool)
        if "hard_eligible" in frame
        else pd.Series(True, index=frame.index)
    )
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.loc[eligible].groupby("trade_date", sort=True):
        quality = pd.to_numeric(_series(group, "pit_quality_observed_count"), errors="coerce")
        growth = pd.to_numeric(_series(group, "pit_growth_observed_count"), errors="coerce")
        joint = quality.ge(2) & growth.ge(1)
        row: dict[str, Any] = {
            "trade_date": trade_date,
            "eligible_rows": len(group),
            "quality_ready_rows": int(quality.ge(2).sum()),
            "growth_ready_rows": int(growth.ge(1).sum()),
            "quality_growth_ready_rows": int(joint.sum()),
            "quality_row_coverage": float(quality.ge(2).mean()),
            "growth_row_coverage": float(growth.ge(1).mean()),
            "quality_growth_row_coverage": float(joint.mean()),
        }
        for feature in (*QUALITY_FEATURES, *GROWTH_FEATURES):
            row[f"{feature}_coverage"] = float(_series(group, feature).notna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_receipt(
    *,
    observations: Mapping[str, Mapping[str, Any]],
    expected_dates: Sequence[str],
    max_report_age_days: int,
) -> dict[str, Any]:
    bundle_available_dates = [str(item["bundle_available_date"]) for item in observations.values()]
    observed_ages = [
        int(item["observation_age_days"])
        for item in observations.values()
        if item.get("observation_age_days") is not None
    ]
    configured_ages = sorted(
        {
            int(item["max_observation_age_days"])
            for item in observations.values()
            if item.get("max_observation_age_days") is not None
        }
    )
    return {
        "schema_version": FUNDAMENTAL_FEATURE_SCHEMA,
        "status": "research_only",
        "production_feature_schema_changed": False,
        "source_loader": "load_pit_fundamentals_as_of_panel_or_exact_date_views",
        "provenance_policy": "require_observed",
        "revision_safe": True,
        "freshness_verified": True,
        "all_as_of_dates_revision_safe_and_fresh": True,
        "bundle_available_date_min": min(bundle_available_dates),
        "bundle_available_date_max": max(bundle_available_dates),
        "maximum_observed_observation_age_days": max(observed_ages, default=None),
        "configured_max_observation_age_days": configured_ages,
        "observation_vintage_by_as_of_date": {
            date: dict(item) for date, item in observations.items()
        },
        "as_of_start": expected_dates[0],
        "as_of_end": expected_dates[-1],
        "max_report_age_days": max_report_age_days,
        "source_fields": list(PIT_SOURCE_FIELDS),
        "required_field_lineage": list(PIT_LINEAGE_SUFFIXES),
        "quality_features": list(QUALITY_FEATURES),
        "growth_features": list(GROWTH_FEATURES),
        "quality_definition": (
            "ROA, gross margin, CFO/assets, and -(net income-CFO)/assets; "
            "same-report-period alignment required for derived ratios"
        ),
        "growth_definition": "PIT provider revenue YoY and net-profit YoY",
        "cross_section_transform": "same-date percentile rank; no fitted global scaler",
        "financial_sector_policy": (
            "quality inputs excluded from observed coverage and assigned neutral 0.5; "
            "growth inputs remain observed when available"
        ),
        "industry_semantics": "current/static classification; disclosed research limitation",
        "industry_classification_point_in_time": False,
        "frozen_historical_financial_universe": False,
        "industry_classification_schema_review_eligible": False,
        "earnings_yield_is_quality": False,
        "price_momentum_is_fundamental_growth": False,
    }


def _snapshots_from_pit_panel(
    pit_panel: Any,
    *,
    expected_dates: Sequence[str],
) -> dict[str, pd.DataFrame]:
    frame, audit = _snapshot_parts(pit_panel)
    if audit.get("provenance_policy") != "require_observed":
        raise ValueError("PIT panel requires provenance_policy=require_observed")
    audit_dates = tuple(sorted(_date_key(value) for value in audit.get("as_of_dates", ())))
    if audit_dates != tuple(expected_dates):
        raise ValueError("PIT panel audit must exactly cover the DailyWatch20 trade calendar")
    observations = audit.get("observation_by_as_of_date")
    if not isinstance(observations, Mapping):
        raise ValueError("PIT panel audit is missing observation_by_as_of_date")
    frame_dates = {_date_key(value) for value in _series(frame, "as_of_date").unique()}
    if frame_dates != set(expected_dates):
        raise ValueError("PIT panel state rows do not cover every requested as-of date")
    snapshots: dict[str, pd.DataFrame] = {}
    for as_of_date in expected_dates:
        state = frame.loc[_series(frame, "as_of_date").map(_date_key).eq(as_of_date)].copy()
        observation = observations.get(as_of_date)
        if not isinstance(observation, Mapping):
            raise ValueError(f"PIT panel lacks observation vintage state: {as_of_date}")
        revision_safe = observation.get("revision_covered") is True
        freshness_verified = observation.get("freshness_verified") is True
        state.attrs["pit_audit"] = {
            "as_of_date": as_of_date,
            "bundle_available_date": observation.get("bundle_available_date"),
            "provenance_policy": "require_observed",
            "revision_safe": revision_safe,
            "freshness_verified": freshness_verified,
            "production_eligible": revision_safe and freshness_verified,
            "oldest_component_retrieval_date": observation.get("oldest_component_retrieval_date"),
            "observation_age_days": observation.get("observation_age_days"),
            "max_observation_age_days": observation.get("max_observation_age_days"),
            "latest_observed_vintage_by_source": observation.get(
                "latest_observed_vintage_by_source", {}
            ),
            "missing_observation_sources": observation.get("missing_observation_sources", []),
        }
        snapshots[as_of_date] = state
    return snapshots


def build_fundamental_feature_panel_from_pit_panel(
    daily_watch20_frame: pd.DataFrame,
    pit_panel: Any,
    *,
    max_report_age_days: int = PIT_MAX_REPORT_AGE_DAYS,
    industry_column: str = "first_industry_name",
) -> FundamentalFeaturePanel:
    """Build features from the formal multi-date loader without ad-hoc forward fills."""

    dates = pd.to_datetime(
        _series(daily_watch20_frame, "trade_date"), errors="raise"
    ).dt.normalize()
    expected = tuple(sorted({_date_key(value) for value in dates}))
    snapshots = _snapshots_from_pit_panel(pit_panel, expected_dates=expected)
    return build_fundamental_feature_panel(
        daily_watch20_frame,
        snapshots,
        max_report_age_days=max_report_age_days,
        industry_column=industry_column,
    )


def build_fundamental_feature_panel(
    daily_watch20_frame: pd.DataFrame,
    pit_snapshots: Mapping[str, Any],
    *,
    max_report_age_days: int = PIT_MAX_REPORT_AGE_DAYS,
    industry_column: str = "first_industry_name",
) -> FundamentalFeaturePanel:
    """Attach controlled quality/growth packs from exact-date formal PIT loader views."""

    if max_report_age_days != PIT_MAX_REPORT_AGE_DAYS:
        raise ValueError("max_report_age_days must equal the frozen 250-day contract")
    missing = sorted({"trade_date", "symbol", industry_column} - set(daily_watch20_frame.columns))
    if missing:
        raise ValueError(f"DailyWatch20 fundamental frame is missing columns: {missing}")
    daily = daily_watch20_frame.copy()
    daily["trade_date"] = pd.to_datetime(
        _series(daily, "trade_date"), errors="raise"
    ).dt.normalize()
    expected_dates = tuple(sorted({_date_key(value) for value in _series(daily, "trade_date")}))
    normalized = {_date_key(key): value for key, value in pit_snapshots.items()}
    if set(normalized) != set(expected_dates):
        raise ValueError("PIT snapshots must exactly cover the DailyWatch20 trade calendar")
    parts: list[pd.DataFrame] = []
    observations: dict[str, dict[str, Any]] = {}
    for as_of_date in expected_dates:
        state, audit = _snapshot_parts(normalized[as_of_date])
        observations[as_of_date] = _validate_snapshot_audit(audit, as_of_date=as_of_date)
        validated = _validate_snapshot_frame(state, as_of_date=as_of_date)
        parts.append(_attach_one_snapshot(daily, validated, as_of_date=as_of_date))
    out = pd.concat(parts, ignore_index=True)
    if "value_yield" in out:
        out["value_yield_pct"] = (
            pd.to_numeric(_series(out, "value_yield"), errors="coerce")
            .groupby(_series(out, "trade_date"), sort=False)
            .rank(pct=True)
        )
    financial = (
        _series(out, industry_column)
        .astype("string")
        .str.contains(_FINANCIAL_INDUSTRY_PATTERN, regex=True, na=False)
    )
    out = _attach_ranked_features(
        out,
        _raw_feature_specs(out, max_report_age_days=max_report_age_days),
        financial=financial,
    )
    out = out.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)
    return FundamentalFeaturePanel(
        frame=out,
        coverage_daily=_feature_coverage(out),
        receipt=_feature_receipt(
            observations=observations,
            expected_dates=expected_dates,
            max_report_age_days=max_report_age_days,
        ),
    )


__all__ = [
    "FUNDAMENTAL_FEATURE_SCHEMA",
    "GROWTH_FEATURES",
    "PIT_LINEAGE_SUFFIXES",
    "PIT_MAX_OBSERVATION_AGE_DAYS",
    "PIT_MAX_REPORT_AGE_DAYS",
    "PIT_SOURCE_FIELDS",
    "QUALITY_FEATURES",
    "FundamentalFeaturePanel",
    "build_fundamental_feature_panel",
    "build_fundamental_feature_panel_from_pit_panel",
]
