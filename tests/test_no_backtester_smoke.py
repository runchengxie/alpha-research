from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SMOKE_CODE = r"""
import json
import sys
from pathlib import Path

import pandas as pd

from alpha_research.metrics import daily_ic_series, summarize_ic
from alpha_research.research_dataset import ResearchDataset
from alpha_research.research_model import ResearchModel
from alpha_research.signal_artifact import (
    SIGNAL_CONTRACT_NAME,
    load_signal_metadata,
    read_signal_artifact,
    write_signal_artifact,
)

out_dir = Path(sys.argv[1])
frame = pd.DataFrame(
    {
        "trade_date": pd.to_datetime(
            [
                "2026-01-05",
                "2026-01-05",
                "2026-01-06",
                "2026-01-06",
            ]
        ),
        "symbol": ["600519.SH", "000858.SZ", "600519.SH", "000858.SZ"],
        "f1": [1.0, 2.0, 2.0, 1.0],
        "f2": [0.5, 0.3, 0.7, 0.2],
        "target": [0.01, -0.01, 0.02, -0.02],
        "_segment": ["train", "train", "test", "test"],
        "close": [10.0, 11.0, 10.5, 10.8],
    }
)
dataset = ResearchDataset(
    raw_panel=frame,
    raw_feature_label=frame,
    infer_frame=frame,
    learn_frame=frame,
    backtest_pricing_frame=frame[["trade_date", "symbol", "close"]],
    feature_cols=("f1", "f2"),
    target_col="target",
    train_target_col="target",
)
model = ResearchModel.from_config(
    {"type": "ridge", "params": {"alpha": 0.1}},
    features=["f1", "f2"],
    target_col="target",
)
model.fit(dataset, "train")
pred = model.predict(dataset, "test")
scored = pred.merge(
    dataset.fetch_infer("test", audit=True)[["trade_date", "symbol", "target"]],
    left_on=["signal_date", "symbol"],
    right_on=["trade_date", "symbol"],
    how="left",
)
ic = daily_ic_series(scored, target_col="target", pred_col="raw_pred")
signal_path = out_dir / "signals.parquet"
signals, summary = write_signal_artifact(
    scored,
    signal_path,
    metadata={"model_detail": model.detail(), "ic": summarize_ic(ic)},
    model_version=model.model_version,
    feature_set_id=model.feature_set_id,
    signal_direction=1.0,
    eligible_for_backtest=True,
    eligible_for_live=False,
)
loaded = read_signal_artifact(signal_path)
metadata = load_signal_metadata(signal_path)

for prefix in ("portfolio_backtester", "strategy_pipeline.pipeline"):
    offenders = [
        module_name
        for module_name in sys.modules
        if module_name == prefix or module_name.startswith(prefix + ".")
    ]
    if offenders:
        raise SystemExit("loaded forbidden module(s): " + ", ".join(sorted(offenders)))

print(
    json.dumps(
        {
            "contract": metadata["artifact_type"],
            "rows": int(loaded.shape[0]),
            "summary_rows": int(summary["rows"]),
            "metadata_rows": int(metadata["summary"]["rows"]),
            "has_model_detail": "model_detail" in metadata["metadata"],
            "required_columns": list(signals.columns[:11]),
            "signal_contract": SIGNAL_CONTRACT_NAME,
        },
        sort_keys=True,
    )
)
"""


def test_alpha_can_train_diagnose_and_write_signal_artifact_without_backtester(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, "-c", SMOKE_CODE, str(tmp_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["contract"] == "alpha_research.signals"
    assert payload["signal_contract"] == "alpha_research.signals"
    assert payload["rows"] == 2
    assert payload["summary_rows"] == 2
    assert payload["metadata_rows"] == 2
    assert payload["has_model_detail"] is True
