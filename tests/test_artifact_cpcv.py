import argparse
import json
from typing import cast

import pandas as pd
import pytest
import yaml

from alpha_research import artifact_cpcv, cpcv


def _write_artifact_inputs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    periods = pd.DataFrame(
        [
            {
                "rebalance_date": "20200103",
                "entry_date": "20200106",
                "exit_date": "20200110",
                "net_return": 0.02,
                "gross_return": 0.021,
                "turnover": 1.0,
                "total_cost": 0.001,
            },
            {
                "rebalance_date": "20200110",
                "entry_date": "20200113",
                "exit_date": "20200117",
                "net_return": -0.01,
                "gross_return": -0.009,
                "turnover": 0.4,
                "total_cost": 0.001,
            },
            {
                "rebalance_date": "20200117",
                "entry_date": "20200120",
                "exit_date": "20200124",
                "net_return": 0.03,
                "gross_return": 0.031,
                "turnover": 0.5,
                "total_cost": 0.001,
            },
            {
                "rebalance_date": "20200124",
                "entry_date": "20200127",
                "exit_date": "20200131",
                "net_return": 0.01,
                "gross_return": 0.011,
                "turnover": 0.6,
                "total_cost": 0.001,
            },
        ]
    )
    periods.to_csv(run_dir / "backtest_periods.csv", index=False)
    pd.DataFrame([{"rebalance_date": "20200103", "symbol": "AAA", "weight": 1.0}]).to_csv(
        run_dir / "positions_by_rebalance.csv", index=False
    )

    scored_rows = []
    for date_idx, date in enumerate(["20200103", "20200110", "20200117", "20200124"]):
        for symbol_idx, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"]):
            scored_rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "base_score": float(symbol_idx),
                    "volume_heat": float(3 - symbol_idx + date_idx),
                    "future_return": float(symbol_idx - 1) / 100.0,
                }
            )
    pd.DataFrame(scored_rows).to_parquet(run_dir / "signals.parquet")

    benchmark = pd.DataFrame(
        {
            "trade_date": pd.date_range("2020-01-06", "2020-01-31", freq="B"),
            "benchmark_return": [0.001] * 20,
        }
    )
    benchmark.to_csv(run_dir / "benchmark.csv", index=False)
    return run_dir


def _write_config(tmp_path, run_dir):
    config_path = tmp_path / "artifact_cpcv.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "artifact_cpcv": {
                    "run_dir": str(run_dir),
                    "scored_file": str(run_dir / "signals.parquet"),
                    "benchmark_returns_file": str(run_dir / "benchmark.csv"),
                    "score_col": "base_score",
                    "target_col": "future_return",
                    "score_postprocess": {
                        "method": "rank_blend",
                        "columns": ["volume_heat"],
                        "strength": 0.05,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def test_artifact_cpcv_writes_promotion_gate_compatible_reports(tmp_path):
    run_dir = _write_artifact_inputs(tmp_path)
    config_path = _write_config(tmp_path, run_dir)
    out_dir = tmp_path / "cpcv"

    summary = artifact_cpcv.run_artifact_cpcv(
        config_path,
        n_groups=4,
        test_groups=1,
        embargo_days=0,
        out_dir=out_dir,
    )

    assert summary["schema"] == "artifact_cpcv_summary.v1"
    assert summary["path_count"] == 1
    assert summary["valid_path_count"] == 1
    assert summary["positions_rows"] == 1
    assert summary["ic_median"] is not None
    payload = json.loads((out_dir / "cpcv_summary.json").read_text(encoding="utf-8"))
    assert payload["valid_path_count"] == 1
    assert (out_dir / "cpcv_path_metrics.csv").exists()
    assert (out_dir / "cpcv_path_returns.csv").exists()


def test_cpcv_command_dispatches_artifact_configs(tmp_path, capsys):
    run_dir = _write_artifact_inputs(tmp_path)
    config_path = _write_config(tmp_path, run_dir)
    out_dir = tmp_path / "cpcv"

    result = cpcv.run(
        cast(
            argparse.Namespace,
            type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "n_groups": 4,
                    "test_groups": 1,
                    "embargo_days": 0,
                    "out": str(out_dir),
                    "include_final_oos": False,
                    "fail_on_quality": None,
                    "artifacts_root": None,
                },
            )(),
        )
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "artifact_cpcv_summary.v1"
    assert (out_dir / "cpcv_summary.json").exists()


def test_cpcv_pipeline_config_requires_injected_context_builder(tmp_path):
    config_path = tmp_path / "pipeline.yml"
    config_path.write_text("model: {}\n", encoding="utf-8")

    args = type(
        "Args",
        (),
        {
            "config": str(config_path),
            "n_groups": 4,
            "test_groups": 1,
            "embargo_days": 0,
            "out": str(tmp_path / "cpcv"),
            "include_final_oos": False,
            "fail_on_quality": None,
            "artifacts_root": None,
        },
    )()

    with pytest.raises(SystemExit, match="requires a prepare_research_context adapter"):
        cpcv.run(cast(argparse.Namespace, args))
