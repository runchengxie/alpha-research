from __future__ import annotations

import argparse
import json
from typing import cast

import pandas as pd
import pytest

from alpha_research import dynamic_signal_ensemble as dse


def _ensemble_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-03", periods=8, freq="W-FRI")
    symbols = [f"S{i}" for i in range(6)]
    rows = []
    for d_idx, date in enumerate(dates):
        for s_idx, symbol in enumerate(symbols):
            rank = float(s_idx)
            forward_return = rank / 100.0
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "future_return": forward_return,
                    "signal_a": rank + d_idx * 0.01,
                    "signal_b": rank * 1.01 + d_idx * 0.01,
                    "signal_c": -rank,
                    "size": rank,
                }
            )
    return pd.DataFrame(rows)


def test_direction_calibration_uses_shifted_history_and_inertia():
    raw_rank_ic = pd.DataFrame(
        {
            "factor": [-0.05, -0.04, 0.01, 0.01, 0.03, 0.04],
        },
        index=pd.date_range("2020-01-01", periods=6, freq="D"),
    )
    config = dse.DynamicSignalEnsembleConfig(
        evaluation_window=2,
        flip_mean_threshold=0.02,
        flip_consistency_threshold=0.60,
    )

    directions, report = dse.calibrate_signal_directions(raw_rank_ic, config)

    assert directions.iloc[0, 0] == 1.0
    assert directions.iloc[2, 0] == -1.0
    assert directions.iloc[4, 0] == -1.0
    row = report[(report["date"] == raw_rank_ic.index[4]) & (report["factor"] == "factor")].iloc[0]
    assert row["reason"] == "inertia"
    assert row["rolling_rank_ic_mean"] == pytest.approx(0.01)


def test_dynamic_signal_ensemble_filters_correlated_factors_and_writes_artifacts(tmp_path):
    result = dse.build_dynamic_signal_ensemble(
        _ensemble_data(),
        signal_cols=["signal_a", "signal_b", "signal_c"],
        target_col="future_return",
        risk_cols=["size"],
        config=dse.DynamicSignalEnsembleConfig(
            min_history=2,
            evaluation_window=2,
            covariance_window=2,
            min_icir=None,
            min_long_short_sharpe=None,
            min_stability=None,
            min_coverage_ratio=0.0,
            min_signal_dispersion=0.0,
            min_rank_ic_mean=None,
            min_direction_consistency=0.0,
            correlation_threshold=0.80,
            stock_selection_count=3,
            stock_buffer_count=1,
            risk_penalty_scale=0.05,
        ),
    )

    assert result.summary["no_level2"] is True
    assert result.summary["artifact_type"] == "alpha_research.dynamic_signal_ensemble"
    assert result.summary["rolling_metrics_shifted"] is True
    assert not result.factor_monitor.empty
    assert "correlation_filtered" in set(result.factor_monitor["drop_reason"])
    assert not result.stock_scores.empty

    paths = dse.write_dynamic_ensemble_artifacts(result, output_dir=tmp_path)
    summary = json.loads((tmp_path / "dynamic_signal_ensemble_summary.json").read_text())

    assert (tmp_path / "dynamic_scores.parquet").exists()
    assert summary["files"]["factor_monitor_file"] == paths["factor_monitor_file"]
    assert summary["risk_penalty_enabled"] is True


def test_cli_config_can_build_from_existing_scored_table(tmp_path):
    input_file = tmp_path / "scored.parquet"
    _ensemble_data().to_parquet(input_file)
    config_path = tmp_path / "ensemble.yml"
    config_path.write_text(
        """
dynamic_signal_ensemble:
  input_file: scored.parquet
  signal_cols: [signal_a, signal_b, signal_c]
  target_col: future_return
  risk_cols: [size]
  output_dir: ensemble_out
  config:
    min_history: 2
    evaluation_window: 2
    covariance_window: 2
    min_icir: null
    min_long_short_sharpe: null
    min_stability: null
    min_coverage_ratio: 0.0
    min_signal_dispersion: 0.0
    min_rank_ic_mean: null
    min_direction_consistency: 0.0
    stock_selection_count: 3
""",
        encoding="utf-8",
    )

    dse.run(cast(argparse.Namespace, type("Args", (), {"config": str(config_path), "output_dir": None})()))

    assert (tmp_path / "ensemble_out" / "dynamic_signal_ensemble_summary.json").exists()
