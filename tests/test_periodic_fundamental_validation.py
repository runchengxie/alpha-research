import pandas as pd
import pytest

from alpha_research.fundamental_state import (
    FundamentalTargetSpec,
    build_periodic_fundamental_target_panel,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "period": ["2024-03-31", "2024-06-30"],
            "published": ["2024-04-30", "2024-08-01"],
            "value": [10.0, 12.0],
        }
    )


def _build(frame: pd.DataFrame):
    return build_periodic_fundamental_target_panel(
        frame,
        (FundamentalTargetSpec("growth", "value", "pct_change"),),
        horizon_periods=1,
        symbol_col="ticker",
        report_period_col="period",
        available_date_col="published",
    )


def test_periodic_custom_columns_preserve_input_and_label_availability() -> None:
    frame = _frame()
    before = frame.copy(deep=True)
    result = _build(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert result.frame.loc[0, "growth"] == pytest.approx(0.2)
    assert result.frame.loc[0, "fundamental_label_end_date"] == pd.Timestamp("2024-08-01")
    assert result.audit["complete_label_rows"] == 1


@pytest.mark.parametrize(
    "column,value,error",
    [
        ("ticker", " ", "non-empty symbols"),
        ("period", "not-a-date", "valid dates"),
        ("published", "2024-03-31", "after their report period"),
        ("published", "not-a-date", "valid dates"),
    ],
)
def test_periodic_invalid_observations_are_rejected(column, value, error) -> None:
    frame = _frame()
    frame.loc[0, column] = value
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match=error):
        _build(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_periodic_missing_and_duplicate_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        _build(_frame().drop(columns="value"))
    with pytest.raises(ValueError, match="duplicate"):
        _build(pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True))
