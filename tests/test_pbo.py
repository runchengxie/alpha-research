import argparse
import json
from typing import cast

import pandas as pd

from alpha_research import pbo


def test_compute_pbo_report_outputs_summary_and_split_rows(tmp_path):
    dates = pd.date_range("2020-01-01", periods=16, freq="D")
    returns = pd.DataFrame(
        {
            "date": dates,
            "candidate_a": [0.01, 0.02, 0.01, -0.01] * 4,
            "candidate_b": [0.02, -0.03, 0.01, 0.00] * 4,
            "candidate_c": [-0.01, 0.00, 0.01, 0.02] * 4,
        }
    )

    report = pbo.compute_pbo_report(returns, n_groups=4, test_groups=2)

    assert report["summary"]["candidate_count"] == 3
    assert report["summary"]["split_count"] == 6
    assert 0.0 <= report["summary"]["pbo"] <= 1.0
    assert 0.0 <= report["summary"]["dsr"] <= 1.0
    assert report["summary"]["dsr_n_trials"] == 3
    assert {row["selected_candidate"] for row in report["rows"]}


def test_pbo_run_writes_summary_and_split_files(tmp_path):
    returns_path = tmp_path / "returns.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=12, freq="D"),
            "a": [0.01, -0.01, 0.02] * 4,
            "b": [-0.01, 0.03, -0.02] * 4,
        }
    ).to_csv(returns_path, index=False)
    out = tmp_path / "pbo"

    assert (
        pbo.run(
            cast(
                argparse.Namespace,
                type(
                    "Args",
                    (),
                    {
                        "returns": str(returns_path),
                        "date_col": "date",
                        "candidate_col": None,
                        "n_groups": 4,
                        "test_groups": 2,
                        "periods_per_year": None,
                        "out": str(out),
                    },
                )(),
            )
        )
        == 0
    )

    payload = json.loads((out / "pbo_summary.json").read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 2
    assert (out / "pbo_splits.csv").exists()
