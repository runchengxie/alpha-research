import pandas as pd
import pytest

from cstree.alpha.recency_diagnostics import build_recency_diagnostics


def test_build_recency_diagnostics_uses_calendar_windows_and_roles() -> None:
    dates = pd.to_datetime(["2026-01-01", "2026-01-20", "2026-02-01"])
    returns = pd.Series([0.01, -0.02, 0.03], index=dates, name="net_return")
    ic = pd.Series([0.1, -0.2, 0.3], index=dates, name="ic")
    active = pd.Series([0.005, -0.01, 0.02], index=dates, name="active_return")
    turnover = pd.Series([0.2, 0.3, 0.1], index=dates, name="turnover")

    frame = build_recency_diagnostics(
        window_labels=["1m", "1w"],
        ic_series=ic,
        returns=returns,
        active_returns=active,
        turnover=turnover,
        periods_per_year=12.0,
    ).set_index("window")

    one_month = frame.loc["1m"]
    assert one_month["role"] == "watch_signal"
    assert one_month["status"] == "ok"
    assert one_month["ic_count"] == 3
    assert one_month["return_count"] == 3
    assert one_month["total_return"] == pytest.approx((1.01 * 0.98 * 1.03) - 1.0)
    assert one_month["avg_turnover"] == pytest.approx(0.2)

    one_week = frame.loc["1w"]
    assert one_week["role"] == "monitoring_only"
    assert one_week["status"] == "limited_sample"
    assert one_week["ic_count"] == 1
    assert one_week["return_count"] == 1
    assert one_week["total_return"] == pytest.approx(0.03)
