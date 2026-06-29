from __future__ import annotations

from typing import Any

SoftCheck = tuple[str, Any, float | None, str]


def _primary_soft_checks(candidate: dict[str, Any], thresholds: Any) -> tuple[SoftCheck, ...]:
    return (
        ("min_eval_ic_ir", candidate["main_eval"]["eval_ic_ir"], thresholds.min_eval_ic_ir, ">="),
        (
            "min_eval_long_short",
            candidate["main_eval"]["eval_long_short"],
            thresholds.min_eval_long_short,
            ">=",
        ),
        (
            "min_walk_forward_test_ic_mean",
            candidate["walk_forward"]["test_ic_mean"],
            thresholds.min_walk_forward_test_ic_mean,
            ">=",
        ),
        (
            "min_final_oos_ic_mean",
            candidate["final_oos"]["ic_mean"],
            thresholds.min_final_oos_ic_mean,
            ">=",
        ),
        (
            "min_final_oos_long_short",
            candidate["final_oos"]["long_short"],
            thresholds.min_final_oos_long_short,
            ">=",
        ),
    )


def _operational_soft_checks(candidate: dict[str, Any], thresholds: Any) -> tuple[SoftCheck, ...]:
    return (
        (
            "max_backtest_drawdown",
            None
            if candidate["backtest"]["max_drawdown"] is None
            else abs(candidate["backtest"]["max_drawdown"]),
            thresholds.max_backtest_drawdown,
            "<=",
        ),
        (
            "max_backtest_avg_turnover",
            candidate["backtest"]["avg_turnover"],
            thresholds.max_backtest_avg_turnover,
            "<=",
        ),
        (
            "max_backtest_avg_cost_drag",
            candidate["backtest"]["avg_cost_drag"],
            thresholds.max_backtest_avg_cost_drag,
            "<=",
        ),
    )


def _cpcv_soft_checks(candidate: dict[str, Any], thresholds: Any) -> tuple[SoftCheck, ...]:
    return (
        (
            "min_cpcv_sharpe_median",
            candidate["cpcv"].get("sharpe_median"),
            thresholds.min_cpcv_sharpe_median,
            ">=",
        ),
        (
            "min_cpcv_sharpe_p25",
            candidate["cpcv"].get("sharpe_p25"),
            thresholds.min_cpcv_sharpe_p25,
            ">=",
        ),
        (
            "min_cpcv_positive_sharpe_ratio",
            candidate["cpcv"].get("positive_sharpe_ratio"),
            thresholds.min_cpcv_positive_sharpe_ratio,
            ">=",
        ),
        (
            "min_cpcv_ic_median",
            candidate["cpcv"].get("ic_median"),
            thresholds.min_cpcv_ic_median,
            ">=",
        ),
        (
            "min_cpcv_long_short_median",
            candidate["cpcv"].get("long_short_median"),
            thresholds.min_cpcv_long_short_median,
            ">=",
        ),
        (
            "max_cpcv_drawdown_p10",
            candidate["cpcv"].get("max_drawdown_p10"),
            thresholds.max_cpcv_drawdown_p10,
            "<=",
        ),
    )


def _evidence_soft_checks(candidate: dict[str, Any], thresholds: Any) -> tuple[SoftCheck, ...]:
    return (
        (
            "max_exposure_screen_breach_count",
            candidate["exposure_screen"].get("breach_count"),
            thresholds.max_exposure_screen_breach_count,
            "<=",
        ),
        (
            "min_dsr",
            candidate["dsr"].get("dsr"),
            thresholds.min_dsr,
            ">=",
        ),
    )


def _soft_threshold_checks(candidate: dict[str, Any], thresholds: Any) -> tuple[SoftCheck, ...]:
    return (
        *_primary_soft_checks(candidate, thresholds),
        *_operational_soft_checks(candidate, thresholds),
        *_cpcv_soft_checks(candidate, thresholds),
        *_evidence_soft_checks(candidate, thresholds),
    )


def _absolute_soft_failures(candidate: dict[str, Any], thresholds: Any) -> list[str]:
    failures: list[str] = []
    checks = _soft_threshold_checks(candidate, thresholds)
    for name, value, threshold, op in checks:
        if threshold is None:
            continue
        if (
            value is None
            or (op == ">=" and value < threshold)
            or (op == "<=" and value > threshold)
        ):
            failures.append(name)
    return failures


def _metric_delta_failure(
    *,
    name: str,
    base: Any,
    cand: Any,
    threshold: float | None,
) -> str | None:
    if threshold is None:
        return None
    if base is None or cand is None or cand - base < threshold:
        return name
    return None


def _delta_soft_failures(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: Any,
) -> list[str]:
    checks = (
        (
            "min_backtest_sharpe_delta",
            baseline["backtest"]["sharpe"],
            candidate["backtest"]["sharpe"],
            thresholds.min_backtest_sharpe_delta,
        ),
        (
            "min_final_oos_sharpe_delta",
            baseline["final_oos"]["sharpe"],
            candidate["final_oos"]["sharpe"],
            thresholds.min_final_oos_sharpe_delta,
        ),
        (
            "min_cpcv_sharpe_median_delta",
            baseline["cpcv"].get("sharpe_median"),
            candidate["cpcv"].get("sharpe_median"),
            thresholds.min_cpcv_sharpe_median_delta,
        ),
        (
            "min_cpcv_sharpe_p25_delta",
            baseline["cpcv"].get("sharpe_p25"),
            candidate["cpcv"].get("sharpe_p25"),
            thresholds.min_cpcv_sharpe_p25_delta,
        ),
    )
    return [
        failure
        for name, base, cand, threshold in checks
        if (
            failure := _metric_delta_failure(
                name=name,
                base=base,
                cand=cand,
                threshold=threshold,
            )
        )
        is not None
    ]


def soft_failures(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: Any,
) -> list[str]:
    return [
        *_absolute_soft_failures(candidate, thresholds),
        *_delta_soft_failures(baseline, candidate, thresholds),
    ]


__all__ = ["soft_failures"]
