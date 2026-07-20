"""Alpha-owned rolling out-of-sample scoring and signal diagnostics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from .daily_watch20 import RELATIVE_PERCENTILE_COL
from .daily_watch20_statistics import newey_west_mean_inference


class RollingRanker(Protocol):
    """Minimal ranker capability used by the OOS application service."""

    training_summary: object | None

    def fit(self, frame: pd.DataFrame, *, as_of_date: pd.Timestamp) -> object: ...

    def predict_relative(self, frame: pd.DataFrame) -> pd.DataFrame: ...


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(_series(frame, column), errors="coerce"))


def date_ic(
    group: pd.DataFrame,
    label_col: str,
    *,
    score_column: str = RELATIVE_PERCENTILE_COL,
) -> float:
    """Return one-date Spearman IC over finite score/label pairs."""

    score = _numeric_series(group, score_column)
    label = _numeric_series(group, label_col)
    finite = pd.Series(
        np.isfinite(score.to_numpy(dtype=float)) & np.isfinite(label.to_numpy(dtype=float)),
        index=group.index,
    )
    score_rank = score.loc[finite].rank(method="average")
    label_rank = label.loc[finite].rank(method="average")
    value = cast(float, score_rank.corr(label_rank))
    return float(value) if bool(pd.notna(value)) else np.nan


def cross_section_daily_rows(
    scored: pd.DataFrame,
    *,
    label_col: str,
    return_col: str,
    score_column: str = RELATIVE_PERCENTILE_COL,
) -> pd.DataFrame:
    """Preserve date-level IC and top-bucket label returns for robust inference."""

    rows: list[dict[str, Any]] = []
    for trade_date, group in scored.groupby("trade_date", sort=True):
        ranked = group.sort_values(
            [score_column, "symbol"],
            ascending=[False, True],
            kind="mergesort",
        )
        labels = _numeric_series(ranked, label_col)
        returns = _numeric_series(ranked, return_col)
        top4 = returns.iloc[:4]
        top20 = returns.iloc[:20]
        label_finite = pd.Series(
            np.isfinite(labels.to_numpy(dtype=float)), index=labels.index, dtype=bool
        )
        top4_finite = pd.Series(
            np.isfinite(top4.to_numpy(dtype=float)), index=top4.index, dtype=bool
        )
        top20_finite = pd.Series(
            np.isfinite(top20.to_numpy(dtype=float)), index=top20.index, dtype=bool
        )
        top4_complete = len(top4) == 4 and bool(top4_finite.all())
        top20_complete = len(top20) == 20 and bool(top20_finite.all())
        rows.append(
            {
                "trade_date": cast(pd.Timestamp, pd.Timestamp(str(trade_date))),
                "spearman_ic": date_ic(group, label_col, score_column=score_column),
                "ic_label_candidate_count": len(labels),
                "ic_label_observed_count": int(label_finite.sum()),
                "ic_label_coverage": float(label_finite.mean()),
                "ic_semantics": "conditional_on_finite_future_label",
                "top4_label_observed_count": int(top4_finite.sum()),
                "top4_label_coverage": float(top4_finite.mean()) if len(top4) else 0.0,
                "top4_label_return_complete": top4_complete,
                "top4_label_return_mean": float(top4.mean()) if top4_complete else np.nan,
                "top20_label_observed_count": int(top20_finite.sum()),
                "top20_label_coverage": float(top20_finite.mean()) if len(top20) else 0.0,
                "top20_label_return_complete": top20_complete,
                "top20_label_return_mean": float(top20.mean()) if top20_complete else np.nan,
            }
        )
    return pd.DataFrame(rows)


def inference_fields(
    prefix: str,
    values: pd.Series,
    *,
    minimum_lag: int = 0,
) -> dict[str, Any]:
    inference = newey_west_mean_inference(
        pd.to_numeric(values, errors="coerce"),
        minimum_lag=minimum_lag,
    )
    return {
        f"{prefix}_nw_observations": inference["observations"],
        f"{prefix}_nw_max_lag": inference["max_lag"],
        f"{prefix}_nw_standard_error": inference["standard_error"],
        f"{prefix}_nw_t_stat": inference["t_stat"],
        f"{prefix}_nw_p_value": inference["p_value"],
        f"{prefix}_nw_ci_95_low": inference["ci_95_low"],
        f"{prefix}_nw_ci_95_high": inference["ci_95_high"],
    }


def finite_positive_ratio(values: pd.Series) -> float:
    numeric = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    finite = numeric.loc[np.isfinite(numeric.to_numpy(dtype=float))]
    return float((finite > 0).mean()) if not finite.empty else np.nan


def signal_summary_fields(
    scored: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    label_col: str,
    feature_count: int,
    refit_count: int,
    score_column: str = RELATIVE_PERCENTILE_COL,
) -> dict[str, Any]:
    """Summarize signal quality without importing portfolio accounting code."""

    if "spearman_ic" in daily:
        daily_ic = _numeric_series(daily, "spearman_ic")
    else:
        daily_ic = pd.Series(
            {
                cast(pd.Timestamp, pd.Timestamp(str(date))): date_ic(
                    group, label_col, score_column=score_column
                )
                for date, group in scored.groupby("trade_date", sort=True)
            },
            dtype=float,
        )
    top4_numeric = _numeric_series(daily, "top4_label_return_mean")
    top20_numeric = _numeric_series(daily, "top20_label_return_mean")
    top4_complete = _series(daily, "top4_label_return_complete").astype(bool)
    top20_complete = _series(daily, "top20_label_return_complete").astype(bool)
    top4_inference = top4_numeric if bool(top4_complete.all()) else pd.Series(dtype=float)
    top20_inference = top20_numeric if bool(top20_complete.all()) else pd.Series(dtype=float)
    label_candidate_count = int(_numeric_series(daily, "ic_label_candidate_count").sum())
    label_observed_count = int(_numeric_series(daily, "ic_label_observed_count").sum())
    return {
        "feature_count": feature_count,
        "evaluation_rows": len(scored),
        "evaluation_dates": int(_series(scored, "trade_date").nunique()),
        "refit_count": refit_count,
        "mean_spearman_ic": float(daily_ic.mean()),
        "median_spearman_ic": float(daily_ic.median()),
        "positive_ic_ratio": finite_positive_ratio(daily_ic),
        "ic_semantics": "conditional_on_finite_future_label",
        "ic_label_candidate_count": label_candidate_count,
        "ic_label_observed_count": label_observed_count,
        "ic_label_coverage": (
            label_observed_count / label_candidate_count if label_candidate_count else 0.0
        ),
        "ic_label_coverage_min": float(_numeric_series(daily, "ic_label_coverage").min()),
        "ic_label_complete_date_ratio": float(
            _numeric_series(daily, "ic_label_coverage").eq(1.0).mean()
        ),
        "top4_label_complete_date_ratio": float(top4_complete.mean()),
        "top20_label_complete_date_ratio": float(top20_complete.mean()),
        "top4_label_return_mean": (
            float(top4_numeric.mean()) if bool(top4_complete.all()) else np.nan
        ),
        "top20_label_return_mean": (
            float(top20_numeric.mean()) if bool(top20_complete.all()) else np.nan
        ),
        **inference_fields("mean_spearman_ic", daily_ic, minimum_lag=4),
        **inference_fields("top4_label_return", top4_inference, minimum_lag=4),
        **inference_fields("top20_label_return", top20_inference, minimum_lag=4),
    }


def score_rolling_oos(
    frame: pd.DataFrame,
    *,
    ranker_factory: Any,
    group_features: tuple[str, ...],
    label_col: str,
    return_col: str,
    evaluation_dates: pd.DatetimeIndex,
    rolling_folds: int,
    passthrough_columns: tuple[str, ...] = (),
    embargo_trade_days: int = 0,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit only on eligible rows and emit deterministic rolling OOS scores."""

    if (
        not isinstance(embargo_trade_days, int)
        or isinstance(embargo_trade_days, bool)
        or embargo_trade_days < 0
    ):
        raise ValueError("embargo_trade_days must be a non-negative integer")
    eligible = cast(pd.Series, frame["hard_eligible"]).astype(bool)
    training_frame = cast(pd.DataFrame, frame.loc[eligible]).copy()
    training_dates = pd.DatetimeIndex(
        pd.to_datetime(_series(training_frame, "trade_date").unique())
    ).sort_values()
    scored_parts: list[pd.DataFrame] = []
    refits: list[dict[str, Any]] = []
    for raw_block in np.array_split(evaluation_dates, rolling_folds):
        block_dates = pd.DatetimeIndex(raw_block)
        if block_dates.empty:
            continue
        refit_date = cast(pd.Timestamp, block_dates[0])
        prior_dates = training_dates[training_dates < refit_date]
        required_prior_dates = embargo_trade_days + 1 if embargo_trade_days else 0
        if len(prior_dates) < required_prior_dates:
            raise ValueError(
                "DailyWatch20 OOS has insufficient prior trade dates for "
                f"embargo_trade_days={embargo_trade_days} at {refit_date.date()}"
            )
        fit_as_of_date = (
            cast(pd.Timestamp, prior_dates[-(embargo_trade_days + 1)])
            if embargo_trade_days
            else refit_date
        )
        excluded_dates = tuple(prior_dates[-embargo_trade_days:]) if embargo_trade_days else ()
        ranker: RollingRanker = ranker_factory(group_features)
        ranker.fit(training_frame, as_of_date=fit_as_of_date)
        candidates = training_frame.loc[
            _series(training_frame, "trade_date").isin(block_dates.tolist())
        ]
        scores = ranker.predict_relative(candidates)
        keep = list(
            dict.fromkeys(
                [
                    "trade_date",
                    "symbol",
                    label_col,
                    return_col,
                    "forward_return_1d",
                    "forward_label_start_date",
                    "forward_label_end_date",
                    *passthrough_columns,
                ]
            )
        )
        scored = candidates[keep].merge(
            scores,
            on=["trade_date", "symbol"],
            validate="one_to_one",
        )
        scored["refit_as_of_date"] = refit_date
        scored["model_fit_as_of_date"] = fit_as_of_date
        scored["last_allowed_label_end_date"] = fit_as_of_date
        scored["embargo_trade_days"] = embargo_trade_days
        scored_parts.append(scored)
        summary = ranker.training_summary
        if summary is None:
            raise RuntimeError("DailyWatch20 OOS ranker has no training summary")
        if hasattr(summary, "__dataclass_fields__"):
            refit_summary = asdict(cast(Any, summary))
        elif isinstance(summary, dict):
            refit_summary = dict(summary)
        else:
            raise TypeError("DailyWatch20 training summary must be a dataclass or mapping")
        refit_summary.update(
            {
                "evaluation_refit_date": refit_date,
                "model_fit_as_of_date": fit_as_of_date,
                "last_allowed_label_end_date": fit_as_of_date,
                "embargo_excluded_trade_dates": excluded_dates,
                "embargo_trade_days": embargo_trade_days,
            }
        )
        refits.append(refit_summary)
    if not scored_parts:
        return pd.DataFrame(), refits
    return pd.concat(scored_parts, ignore_index=True), refits


# Compatibility aliases for callers migrating from strategy-pipeline internals.
_date_ic = date_ic
_inference_fields = inference_fields
_finite_positive_ratio = finite_positive_ratio


__all__ = [
    "RollingRanker",
    "cross_section_daily_rows",
    "date_ic",
    "finite_positive_ratio",
    "inference_fields",
    "score_rolling_oos",
    "signal_summary_fields",
]
