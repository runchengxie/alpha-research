"""PIT-safe building blocks for fundamental-state forecasting research."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

FUNDAMENTAL_STATE_SCHEMA = "fundamental_state_forecasting.v1"
TargetTransform = Literal["level", "delta", "pct_change"]


@dataclass(frozen=True)
class FundamentalTargetSpec:
    """Define one future fundamental target from a canonical annual observation."""

    name: str
    source_col: str
    transform: TargetTransform = "level"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("FundamentalTargetSpec.name must be non-empty")
        if not self.source_col.strip():
            raise ValueError("FundamentalTargetSpec.source_col must be non-empty")
        if self.transform not in {"level", "delta", "pct_change"}:
            raise ValueError("FundamentalTargetSpec.transform must be level, delta, or pct_change")


@dataclass(frozen=True)
class FundamentalTargetPanel:
    frame: pd.DataFrame
    audit: dict[str, object]


@dataclass(frozen=True)
class FundamentalScoreSpec:
    """Describe one forecast or valuation input to the cross-sectional score."""

    column: str
    weight: float = 1.0
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not self.column.strip():
            raise ValueError("FundamentalScoreSpec.column must be non-empty")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("FundamentalScoreSpec.weight must be finite and positive")


@dataclass(frozen=True)
class FundamentalForecastRun:
    frame: pd.DataFrame
    audit: dict[str, object]


@dataclass(frozen=True)
class FundamentalPurgeResult:
    frame: pd.DataFrame
    audit: dict[str, object]


def _normalized_dates(series: pd.Series, *, column: str) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"fundamental state requires valid dates in {column}")
    if values.dt.tz is not None:
        values = values.dt.tz_localize(None)
    return values.dt.normalize()


def _nullable_normalized_dates(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    if values.dt.tz is not None:
        values = values.dt.tz_localize(None)
    return values.dt.normalize()


def _numeric(series: pd.Series) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(series, errors="coerce")).replace(
        [np.inf, -np.inf], np.nan
    )


def _validate_target_specs(specs: tuple[FundamentalTargetSpec, ...]) -> None:
    if not specs:
        raise ValueError("fundamental target specs must be non-empty")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("fundamental target names must be unique")


def _target_values(
    current: pd.Series,
    future: pd.Series,
    *,
    transform: TargetTransform,
) -> pd.Series:
    if transform == "level":
        return future
    if transform == "delta":
        return future - current
    valid_base = current.where(current.notna() & np.isfinite(current) & (current > 0))
    return ((future / valid_base) - 1.0).replace([np.inf, -np.inf], np.nan)


def build_annual_fundamental_target_panel(
    frame: pd.DataFrame,
    target_specs: tuple[FundamentalTargetSpec, ...],
    *,
    horizon_years: int = 1,
    symbol_col: str = "symbol",
    report_period_col: str = "report_period",
    available_date_col: str = "available_date",
) -> FundamentalTargetPanel:
    """Attach exact-horizon annual targets without hiding their future availability date.

    The input contract is deliberately strict: one canonical, PIT-audited observation per
    ``(symbol, report_period)``. Revision selection belongs to the data platform, not here.
    """

    specs = tuple(target_specs)
    _validate_target_specs(specs)
    if int(horizon_years) <= 0:
        raise ValueError("horizon_years must be positive")
    required = {symbol_col, report_period_col, available_date_col}
    required.update(spec.source_col for spec in specs)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fundamental state frame missing columns: {missing}")

    out = frame.copy()
    out[symbol_col] = out[symbol_col].astype("string")
    if out[symbol_col].isna().any() or out[symbol_col].str.strip().eq("").any():
        raise ValueError("fundamental state requires non-empty symbols")
    out[symbol_col] = out[symbol_col].astype(str)
    out[report_period_col] = _normalized_dates(out[report_period_col], column=report_period_col)
    out[available_date_col] = _normalized_dates(out[available_date_col], column=available_date_col)
    if out.duplicated([symbol_col, report_period_col]).any():
        raise ValueError("fundamental state input contains duplicate symbol/report_period rows")
    if (out[available_date_col] <= out[report_period_col]).any():
        raise ValueError("annual observations must become available after their report period")

    out["feature_as_of_date"] = out[available_date_col]
    out["target_report_period"] = out[report_period_col] + pd.DateOffset(years=int(horizon_years))

    future_columns = [symbol_col, report_period_col, available_date_col]
    future_columns.extend(sorted({spec.source_col for spec in specs}))
    future = out[future_columns].copy()
    rename = {
        report_period_col: "target_report_period",
        available_date_col: "target_available_date",
    }
    rename.update({spec.source_col: f"__future__{spec.source_col}" for spec in specs})
    future.rename(columns=rename, inplace=True)

    merged = out.merge(
        future,
        how="left",
        on=[symbol_col, "target_report_period"],
        validate="many_to_one",
        sort=False,
    )
    merged["target_available_date"] = pd.to_datetime(
        merged["target_available_date"], errors="coerce"
    ).dt.normalize()
    invalid_availability = merged["target_available_date"].notna() & (
        merged["target_available_date"] <= merged["feature_as_of_date"]
    )
    if invalid_availability.any():
        raise ValueError("future fundamental labels must become available after feature_as_of_date")
    merged["fundamental_label_end_date"] = merged["target_available_date"]

    future_value_columns = sorted({f"__future__{spec.source_col}" for spec in specs})
    for spec in specs:
        current = _numeric(merged[spec.source_col])
        future_values = _numeric(merged[f"__future__{spec.source_col}"])
        merged[spec.name] = _target_values(current, future_values, transform=spec.transform)
    merged.drop(columns=future_value_columns, inplace=True)

    complete = merged["target_available_date"].notna()
    if specs:
        complete &= merged[[spec.name for spec in specs]].notna().all(axis=1)
    audit: dict[str, object] = {
        "schema_version": FUNDAMENTAL_STATE_SCHEMA,
        "input_contract": "one canonical PIT-audited row per symbol/report_period",
        "horizon_years": int(horizon_years),
        "rows": len(merged),
        "complete_label_rows": int(complete.sum()),
        "target_names": [spec.name for spec in specs],
        "label_end_semantics": "target_available_date",
    }
    return FundamentalTargetPanel(merged, audit)


def build_persistence_baseline(
    frame: pd.DataFrame,
    target_spec: FundamentalTargetSpec,
) -> pd.Series:
    """Return the naive no-change benchmark for one target definition."""

    if target_spec.source_col not in frame.columns:
        raise ValueError(f"persistence baseline missing column: {target_spec.source_col}")
    current = _numeric(frame[target_spec.source_col])
    if target_spec.transform == "level":
        return current
    return pd.Series(0.0, index=frame.index, dtype=float)


def build_operating_quality_persistence_targets(
    frame: pd.DataFrame,
    *,
    horizon_years: int = 1,
    symbol_col: str = "symbol",
    report_period_col: str = "report_period",
    available_date_col: str = "available_date",
    roa_col: str = "roa",
    margin_col: str = "gross_margin",
    growth_col: str = "revenue_growth",
    roa_retention: float = 0.8,
    margin_retention: float = 0.9,
) -> pd.DataFrame:
    """Build PIT-auditable targets for whether operating quality persists.

    This is deliberately a one-step operating-state target, not a long-horizon return
    label.  A company is labelled persistent when its next annual observation remains
    profitable, keeps at least the configured fraction of current ROA and margin, and
    continues to grow revenue.  Missing future observations remain unknown (``<NA>``).
    The returned frame carries an audit in ``DataFrame.attrs``.
    """

    if int(horizon_years) <= 0:
        raise ValueError("horizon_years must be positive")
    if not 0 < float(roa_retention) <= 1 or not 0 < float(margin_retention) <= 1:
        raise ValueError("retention thresholds must be in (0, 1]")
    required = {
        symbol_col,
        report_period_col,
        available_date_col,
        roa_col,
        margin_col,
        growth_col,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"quality persistence frame missing columns: {missing}")

    specs = (
        FundamentalTargetSpec(f"future_roa_{horizon_years}y", roa_col, "level"),
        FundamentalTargetSpec(f"future_gross_margin_{horizon_years}y", margin_col, "level"),
        FundamentalTargetSpec(f"future_revenue_growth_{horizon_years}y", growth_col, "level"),
    )
    targets = build_annual_fundamental_target_panel(
        frame,
        specs,
        horizon_years=horizon_years,
        symbol_col=symbol_col,
        report_period_col=report_period_col,
        available_date_col=available_date_col,
    )
    out = targets.frame
    current_roa = _numeric(out[roa_col])
    current_margin = _numeric(out[margin_col])
    future_roa = _numeric(out[f"future_roa_{horizon_years}y"])
    future_margin = _numeric(out[f"future_gross_margin_{horizon_years}y"])
    future_growth = _numeric(out[f"future_revenue_growth_{horizon_years}y"])
    known = future_roa.notna() & future_margin.notna() & future_growth.notna()
    persistent = (
        known
        & current_roa.gt(0)
        & current_margin.gt(0)
        & future_roa.ge(current_roa * float(roa_retention))
        & future_margin.ge(current_margin * float(margin_retention))
        & future_growth.gt(0)
    )
    label = pd.Series(pd.NA, index=out.index, dtype="boolean")
    label.loc[known] = persistent.loc[known]
    out["quality_persistent_1y"] = (
        label if int(horizon_years) == 1 else label.rename(f"quality_persistent_{horizon_years}y")
    )
    out["quality_label_end_date"] = out["fundamental_label_end_date"]
    out.attrs["audit"] = {
        **targets.audit,
        "target_type": "operating_quality_persistence",
        "pit_policy": "future target availability is retained",
        "roa_retention": float(roa_retention),
        "margin_retention": float(margin_retention),
        "known_label_rows": int(known.sum()),
        "persistent_label_rows": int(persistent.sum()),
    }
    return out


def add_cashflow_yield(
    frame: pd.DataFrame,
    *,
    cashflow_col: str = "n_cashflow_act",
    market_cap_col: str = "total_mv",
    output_col: str = "cashflow_yield",
) -> pd.DataFrame:
    """Add a transparent cash-flow-to-market-cap valuation feature.

    The caller must supply a PIT-safe cash-flow measure and same-date market
    capitalization.  Invalid or non-positive market caps remain missing rather
    than being converted into an extreme valuation score.  The input cash-flow
    may be CFO or already-defined FCF; this function does not silently invent a
    capex convention.
    """

    missing = sorted({cashflow_col, market_cap_col} - set(frame.columns))
    if missing:
        raise ValueError(f"cashflow yield frame missing columns: {missing}")
    if not output_col.strip():
        raise ValueError("output_col must be non-empty")
    out = frame.copy()
    cashflow = _numeric(out[cashflow_col])
    market_cap = _numeric(out[market_cap_col])
    valid_market_cap = market_cap.where(market_cap.gt(0))
    out[output_col] = (cashflow / valid_market_cap).replace([np.inf, -np.inf], np.nan)
    return out


def _rank_ic(actual: pd.Series, predicted: pd.Series) -> float:
    actual_rank = actual.rank(method="average")
    predicted_rank = predicted.rank(method="average")
    value = actual_rank.corr(predicted_rank)
    return float(value) if pd.notna(value) else float("nan")


def _learning_target(
    frame: pd.DataFrame,
    target_col: str,
    *,
    date_col: str,
    model_type: str,
    model_params: Mapping[str, Any],
    target_transform: str,
) -> pd.Series:
    """Materialize a training target aligned with the model's ranking objective.

    Fundamental-state forecasts remain measured against the numeric future state.  Ranking
    objectives, however, should train on within-formation relevance rather than arbitrary
    accounting units.  ``auto`` keeps regressors pointwise and maps XGBoost rankers to
    non-negative cross-sectional relevance grades.
    """

    transform = str(target_transform).strip().lower()
    if transform not in {"auto", "raw", "cross_sectional_rank"}:
        raise ValueError("target_transform must be auto, raw, or cross_sectional_rank")
    objective = str(model_params.get("objective", "")).strip().lower()
    is_ranker = model_type == "xgb_ranker" or objective.startswith("rank:")
    if transform == "auto":
        transform = "cross_sectional_rank" if is_ranker else "raw"
    values = _numeric(frame[target_col])
    if transform == "raw":
        return values
    percentile = values.groupby(frame[date_col], sort=False).rank(method="average", pct=True)
    if objective == "rank:ndcg":
        return np.floor(percentile * 31.0).clip(lower=0.0, upper=31.0).astype("Int32")
    return percentile


def _mean_cross_sectional_rank_ic(
    actual: pd.Series,
    predicted: pd.Series,
    groups: pd.Series,
) -> float:
    values: list[float] = []
    for _, index_values in groups.groupby(groups, sort=False).groups.items():
        group_index = pd.Index(index_values)
        if len(group_index) < 2:
            continue
        value = _rank_ic(actual.loc[group_index], predicted.loc[group_index])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def evaluate_fundamental_forecast(
    frame: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    *,
    directional: bool = False,
    date_col: str | None = None,
) -> dict[str, float | int | None]:
    """Evaluate one OOS fundamental forecast with scale and rank metrics."""

    required = {actual_col, predicted_col}
    if date_col is not None:
        required.add(date_col)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"forecast evaluation missing columns: {missing}")
    actual = _numeric(frame[actual_col])
    predicted = _numeric(frame[predicted_col])
    valid = actual.notna() & predicted.notna()
    actual = actual.loc[valid]
    predicted = predicted.loc[valid]
    if actual.empty:
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "rank_ic": float("nan"),
            "direction_accuracy": None if not directional else float("nan"),
        }
    error = predicted - actual
    rank_ic = _rank_ic(actual, predicted)
    if date_col is not None:
        groups = frame.loc[valid, date_col]
        rank_ic = _mean_cross_sectional_rank_ic(actual, predicted, groups)
    direction_accuracy: float | None = None
    if directional:
        direction_accuracy = float((np.sign(predicted) == np.sign(actual)).mean())
    return {
        "count": len(actual),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error.to_numpy(dtype=float))))),
        "rank_ic": rank_ic,
        "direction_accuracy": direction_accuracy,
    }


def _forecast_fundamental_fold(
    data: pd.DataFrame,
    formation: pd.Timestamp,
    *,
    target_spec: FundamentalTargetSpec,
    features: tuple[str, ...],
    models: Mapping[str, Mapping[str, Any]],
    formation_col: str,
    feature_date_col: str,
    label_end_col: str,
    min_train_rows: int,
    min_train_periods: int,
) -> tuple[pd.DataFrame | None, dict[str, object] | None]:
    from .modeling import build_model, fit_model, resolve_model_spec

    test = data.loc[data[formation_col] == formation].copy()
    if test.empty:
        return None, None
    test_cutoff = cast(pd.Timestamp, test[feature_date_col].min()).normalize()
    train = data.loc[(data[formation_col] < formation) & (data[label_end_col] < test_cutoff)].copy()
    train = train.dropna(subset=[*features, target_spec.name])
    train_periods = int(train[formation_col].nunique())
    test_valid = (
        test[list(features)].notna().all(axis=1)
        & test[target_spec.name].notna()
        & test[label_end_col].notna()
    )
    if len(train) < min_train_rows or train_periods < min_train_periods or not test_valid.any():
        return None, None

    prediction_columns = ["pred_persistence", *(f"pred_{name}" for name in models)]
    for column in prediction_columns:
        test[column] = np.nan
    test.loc[test_valid, "pred_persistence"] = build_persistence_baseline(
        test.loc[test_valid], target_spec
    )
    resolved_models: dict[str, str] = {}
    target_transforms: dict[str, str] = {}
    for name, config in models.items():
        model_type, model_params = resolve_model_spec(config)
        model = build_model(model_type, model_params)
        target_transform = str(config.get("target_transform", "auto"))
        fit_data = train.copy()
        fit_data["__learning_target"] = _learning_target(
            fit_data,
            target_spec.name,
            date_col=formation_col,
            model_type=model_type,
            model_params=model_params,
            target_transform=target_transform,
        )
        fit_data = fit_data.dropna(subset=["__learning_target"])
        fit_model(
            model,
            model_type,
            fit_data,
            features=features,
            target_col="__learning_target",
            date_col=formation_col,
        )
        test.loc[test_valid, f"pred_{name}"] = model.predict(test.loc[test_valid, list(features)])
        resolved_models[name] = model_type
        target_transforms[name] = (
            "cross_sectional_rank"
            if str(target_transform).strip().lower() == "auto"
            and (
                model_type == "xgb_ranker"
                or str(model_params.get("objective", "")).startswith("rank:")
            )
            else str(target_transform).strip().lower()
        )

    training_label_end_max = cast(pd.Timestamp, train[label_end_col].max()).normalize()
    audit = {
        "formation": formation.strftime("%Y-%m-%d"),
        "test_cutoff": test_cutoff.strftime("%Y-%m-%d"),
        "training_rows": len(train),
        "training_periods": train_periods,
        "training_label_end_max": training_label_end_max.strftime("%Y-%m-%d"),
        "test_rows": int(test_valid.sum()),
        "models": resolved_models,
        "target_transforms": target_transforms,
    }
    return test, audit


def _forecast_run_audit(
    *,
    target: str,
    features: tuple[str, ...],
    models: Mapping[str, Mapping[str, Any]],
    fold_audit: list[dict[str, object]],
    skipped_folds: int,
    prediction_rows: int,
    min_train_rows: int,
    min_train_periods: int,
) -> dict[str, object]:
    return {
        "schema_version": FUNDAMENTAL_STATE_SCHEMA,
        "target": target,
        "features": list(features),
        "model_names": list(models),
        "leakage_policy": (
            "prior formation and label_end strictly before earliest test feature date"
        ),
        "folds": fold_audit,
        "completed_folds": len(fold_audit),
        "skipped_folds": skipped_folds,
        "prediction_rows": prediction_rows,
        "min_train_rows": min_train_rows,
        "min_train_periods": min_train_periods,
    }


def run_walk_forward_fundamental_forecast(
    frame: pd.DataFrame,
    *,
    target_spec: FundamentalTargetSpec,
    feature_cols: tuple[str, ...],
    model_configs: Mapping[str, Mapping[str, Any]],
    formation_col: str = "report_period",
    feature_date_col: str = "feature_as_of_date",
    label_end_col: str = "fundamental_label_end_date",
    min_train_rows: int = 50,
    min_train_periods: int = 3,
) -> FundamentalForecastRun:
    """Generate leakage-safe expanding-window OOS forecasts by formation period."""

    features = tuple(str(column).strip() for column in feature_cols)
    if not features or any(not column for column in features):
        raise ValueError("feature_cols must contain non-empty names")
    if len(features) != len(set(features)):
        raise ValueError("feature_cols must be unique")
    future_label_columns = {
        target_spec.name,
        "target_available_date",
        "fundamental_label_end_date",
    }
    leaked_features = sorted(future_label_columns.intersection(features))
    if leaked_features:
        raise ValueError(f"future label columns cannot be model features: {leaked_features}")
    if int(min_train_rows) <= 0 or int(min_train_periods) <= 0:
        raise ValueError("minimum training requirements must be positive")
    models = {str(name).strip(): dict(config) for name, config in model_configs.items()}
    if any(not name for name in models) or "persistence" in models:
        raise ValueError("model names must be non-empty and cannot use persistence")

    required = {
        target_spec.source_col,
        target_spec.name,
        formation_col,
        feature_date_col,
        label_end_col,
        *features,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"walk-forward fundamental frame missing columns: {missing}")

    data = frame.copy()
    data[formation_col] = _normalized_dates(data[formation_col], column=formation_col)
    data[feature_date_col] = _normalized_dates(data[feature_date_col], column=feature_date_col)
    data[label_end_col] = _nullable_normalized_dates(data[label_end_col])
    for column in (*features, target_spec.source_col, target_spec.name):
        data[column] = _numeric(data[column])

    pieces: list[pd.DataFrame] = []
    fold_audit: list[dict[str, object]] = []
    skipped_folds = 0
    formations = pd.Index(data[formation_col].dropna().unique()).sort_values()

    for formation in formations:
        test, audit = _forecast_fundamental_fold(
            data,
            cast(pd.Timestamp, formation),
            target_spec=target_spec,
            features=features,
            models=models,
            formation_col=formation_col,
            feature_date_col=feature_date_col,
            label_end_col=label_end_col,
            min_train_rows=int(min_train_rows),
            min_train_periods=int(min_train_periods),
        )
        if test is None or audit is None:
            skipped_folds += 1
            continue
        fold_audit.append(audit)
        pieces.append(test)

    if pieces:
        predictions = pd.concat(pieces, ignore_index=False).sort_values(
            [formation_col], kind="mergesort"
        )
    else:
        predictions = data.iloc[0:0].copy()
        for column in ["pred_persistence", *(f"pred_{name}" for name in models)]:
            predictions[column] = pd.Series(dtype=float)

    audit = _forecast_run_audit(
        target=target_spec.name,
        features=features,
        models=models,
        fold_audit=fold_audit,
        skipped_folds=skipped_folds,
        prediction_rows=len(predictions),
        min_train_rows=int(min_train_rows),
        min_train_periods=int(min_train_periods),
    )
    return FundamentalForecastRun(predictions, audit)


def build_fundamental_forecast_score(
    frame: pd.DataFrame,
    score_specs: tuple[FundamentalScoreSpec, ...],
    *,
    date_col: str = "signal_date",
    score_col: str = "fundamental_score",
) -> pd.DataFrame:
    """Combine forecast and valuation columns into a transparent cross-sectional score."""

    specs = tuple(score_specs)
    if not specs:
        raise ValueError("fundamental score specs must be non-empty")
    required = {date_col}
    required.update(spec.column for spec in specs)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fundamental score frame missing columns: {missing}")

    out = frame.copy()
    out[date_col] = _normalized_dates(out[date_col], column=date_col)
    weighted_sum = pd.Series(0.0, index=out.index, dtype=float)
    observed_weight = pd.Series(0.0, index=out.index, dtype=float)
    for spec in specs:
        values = _numeric(out[spec.column])
        ranked = values.groupby(out[date_col], sort=False).rank(
            method="average",
            pct=True,
            ascending=spec.higher_is_better,
        )
        component_col = f"{score_col}__{spec.column}_pct"
        out[component_col] = ranked
        observed = ranked.notna()
        weighted_sum.loc[observed] += ranked.loc[observed] * spec.weight
        observed_weight.loc[observed] += spec.weight
    out[score_col] = weighted_sum / observed_weight.replace(0.0, np.nan)
    out[f"{score_col}_coverage_weight"] = observed_weight
    out["fundamental_rank"] = out.groupby(date_col, sort=False)[score_col].rank(
        method="min", ascending=False
    )
    out["fundamental_percentile"] = out.groupby(date_col, sort=False)[score_col].rank(
        method="average", pct=True
    )
    return out


def purge_and_embargo_fundamental_rows(
    frame: pd.DataFrame,
    *,
    test_start: object,
    test_end: object,
    embargo_days: int = 0,
    feature_date_col: str = "feature_as_of_date",
    label_end_col: str = "fundamental_label_end_date",
) -> FundamentalPurgeResult:
    """Remove training rows whose label windows touch the test interval or embargo buffer."""

    missing = sorted({feature_date_col, label_end_col} - set(frame.columns))
    if missing:
        raise ValueError(f"fundamental purge frame missing columns: {missing}")
    if int(embargo_days) < 0:
        raise ValueError("embargo_days must be non-negative")
    start = cast(pd.Timestamp, pd.Timestamp(test_start)).normalize()
    end = cast(pd.Timestamp, pd.Timestamp(test_end)).normalize()
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("test_start and test_end must define a valid interval")

    feature_dates = _normalized_dates(frame[feature_date_col], column=feature_date_col)
    label_ends = _normalized_dates(frame[label_end_col], column=label_end_col)
    overlaps = (feature_dates <= end) & (label_ends >= start)
    embargo_end = end + pd.Timedelta(days=int(embargo_days))
    embargoed = (~overlaps) & (feature_dates > end) & (feature_dates <= embargo_end)
    keep = ~(overlaps | embargoed)
    audit: dict[str, object] = {
        "schema_version": FUNDAMENTAL_STATE_SCHEMA,
        "test_start": start.strftime("%Y-%m-%d"),
        "test_end": end.strftime("%Y-%m-%d"),
        "embargo_days": int(embargo_days),
        "input_rows": len(frame),
        "purged_overlap_rows": int(overlaps.sum()),
        "embargoed_rows": int(embargoed.sum()),
        "kept_rows": int(keep.sum()),
    }
    return FundamentalPurgeResult(frame.loc[keep].copy(), audit)


__all__ = [
    "FUNDAMENTAL_STATE_SCHEMA",
    "FundamentalForecastRun",
    "FundamentalPurgeResult",
    "FundamentalScoreSpec",
    "FundamentalTargetPanel",
    "FundamentalTargetSpec",
    "add_cashflow_yield",
    "build_annual_fundamental_target_panel",
    "build_fundamental_forecast_score",
    "build_operating_quality_persistence_targets",
    "build_persistence_baseline",
    "evaluate_fundamental_forecast",
    "purge_and_embargo_fundamental_rows",
    "run_walk_forward_fundamental_forecast",
]
