import pandas as pd
import pytest

from alpha_research.forecast_skill_gate import forecast_skill_gate


def history():
    rows = []
    for date in pd.date_range("2018-01-01", periods=8, freq="QS"):
        for number in range(20):
            rows.append(
                {
                    "formation_date": date,
                    "label_end_date": date + pd.Timedelta(days=60),
                    "symbol": str(number),
                    "target": 1.0,
                    "model": 1.0,
                    "control": 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_gate_requires_eight_mature_formations_and_ignores_future_outcomes():
    data = history()
    assert not forecast_skill_gate(data, "2019-11-30", "model", "control")["enabled"]
    result = forecast_skill_gate(data, "2020-01-01", "model", "control")
    assert result["enabled"]
    assert result["formations"] == 8
    assert result["mean_mse_gain"] == pytest.approx(1)
    future = data.iloc[:20].assign(
        formation_date=pd.Timestamp("2020-01-01"),
        label_end_date=pd.Timestamp("2020-04-01"),
        model=10000.0,
    )
    assert (
        forecast_skill_gate(pd.concat([data, future]), "2020-01-01", "model", "control") == result
    )


def test_one_large_win_does_not_override_majority_losses():
    data = history()
    data["model"] = 3.0
    data.loc[data.formation_date.eq(data.formation_date.min()), "control"] = 100.0
    result = forecast_skill_gate(data, "2020-01-01", "model", "control")
    assert result["mean_mse_gain"] > 0
    assert result["win_fraction"] == pytest.approx(1 / 8)
    assert not result["enabled"]


def test_maturity_equality_and_unknown_maturity_exclude_whole_formation():
    data = history()
    data.loc[0, "label_end_date"] = pd.Timestamp("2020-01-01")
    assert forecast_skill_gate(data, "2020-01-01", "model", "control")["formations"] == 7
    data.loc[0, "label_end_date"] = pd.NaT
    assert forecast_skill_gate(data, "2020-01-01", "model", "control")["formations"] == 7


def test_missing_matched_rows_and_duplicate_keys():
    data = history()
    data.loc[0, "model"] = float("nan")
    assert forecast_skill_gate(data, "2020-01-01", "model", "control")["formations"] == 7
    with pytest.raises(ValueError, match="duplicate"):
        forecast_skill_gate(pd.concat([data, data.iloc[:1]]), "2020-01-01", "model", "control")
