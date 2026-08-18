from __future__ import annotations

import json

import numpy as np
import pandas as pd
from research_contracts import ArtifactEnvelopeV2, file_sha256, read_artifact_envelope

from alpha_research.style_replica import signal_generator
from alpha_research.style_replica.resvol import compute_resvol_factor
from alpha_research.style_replica.score_a import compute_score_a
from alpha_research.style_replica.score_b import compute_score_b


def test_generate_daily_signals_decorates_and_ranks_scores(monkeypatch) -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    prices = pd.DataFrame(
        {"THEMED": [10.0, 10.5], "UNTHEMED": [20.0, 19.5]},
        index=dates,
    )
    factors = {"placeholder": prices}
    score_a = pd.DataFrame(
        {"THEMED": [0.8, 0.4], "UNTHEMED": [0.9, 0.2]},
        index=dates,
    )
    score_b = pd.DataFrame(
        {"THEMED": [0.3, 0.7], "UNTHEMED": [0.6, 0.9]},
        index=dates,
    )
    industry = pd.DataFrame(
        {
            "symbol": ["THEMED", "UNTHEMED"],
            "industry_name": ["集成电路", "白酒"],
        }
    )

    monkeypatch.setattr(
        signal_generator,
        "compute_all_style_factors",
        lambda *args, **kwargs: factors,
    )
    monkeypatch.setattr(signal_generator, "compute_score_a", lambda factor_map: score_a)
    monkeypatch.setattr(signal_generator, "compute_score_b", lambda factor_map: score_b)

    result = signal_generator.generate_daily_signals(prices, industry_frame=industry)

    first_day = result[result["signal_date"] == "20250102"].set_index("symbol")
    second_day = result[result["signal_date"] == "20250103"].set_index("symbol")
    assert first_day.loc["THEMED", "leg"] == "A"
    assert first_day.loc["UNTHEMED", "leg"] == "B"
    assert first_day.loc["UNTHEMED", "raw_pred"] == 0.9
    assert first_day.loc["UNTHEMED", "rank"] == 1
    assert second_day.loc["THEMED", "raw_pred"] == 0.7
    assert second_day.loc["UNTHEMED", "rank"] == 1
    assert result["model_version"].eq(signal_generator.MODEL_VERSION).all()
    assert result["eligible_for_backtest"].all()
    assert result["eligible_for_live"].all()


def test_scores_allow_optional_intraday_factors_to_be_absent() -> None:
    frame = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]],
        index=pd.date_range("2025-01-02", periods=2),
        columns=pd.Index(["A", "B"]),
    )
    common = {
        "resvol": frame,
        "liquidity": frame,
        "mom20": frame,
        "mom120": frame,
    }
    factors_a = {
        **common,
        "size": frame,
        "beta": frame,
        "industry_mom": frame,
    }
    factors_b = {**common, "vol_convergence": frame}

    assert compute_score_a(factors_a).notna().to_numpy().all()
    assert compute_score_b(factors_b).notna().to_numpy().all()


def test_resvol_uses_observation_level_regression_residuals() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-02", periods=80)
    market = pd.Series(rng.normal(0.0, 0.01, len(dates)), index=dates)
    returns = pd.DataFrame(
        {
            "A": 1.2 * market + rng.normal(0.0, 0.004, len(dates)),
            "B": 0.8 * market + rng.normal(0.0, 0.006, len(dates)),
        },
        index=dates,
    )

    result = compute_resvol_factor(returns, market_returns=market)

    assert result.iloc[:39].isna().all().all()
    assert result.iloc[39:].notna().all().all()
    assert (result.iloc[39:] > 0.0).all().all()


def test_style_replica_write_attaches_readable_v2_envelope(tmp_path, monkeypatch) -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    prices = pd.DataFrame(
        {"THEMED": [10.0, 10.5], "UNTHEMED": [20.0, 19.5]},
        index=dates,
    )
    factors = {"placeholder": prices}
    score_a = pd.DataFrame(
        {"THEMED": [0.8, 0.4], "UNTHEMED": [0.9, 0.2]},
        index=dates,
    )
    score_b = pd.DataFrame(
        {"THEMED": [0.3, 0.7], "UNTHEMED": [0.6, 0.9]},
        index=dates,
    )
    monkeypatch.setattr(
        signal_generator,
        "compute_all_style_factors",
        lambda *args, **kwargs: factors,
    )
    monkeypatch.setattr(signal_generator, "compute_score_a", lambda factor_map: score_a)
    monkeypatch.setattr(signal_generator, "compute_score_b", lambda factor_map: score_b)

    gen = signal_generator.StyleReplicaSignalGenerator()
    signals = gen.generate(prices)
    canonical, _ = gen.write(
        signals,
        tmp_path,
        run_id="run-demo",
        lineage=[("signals.parquet", "c" * 64)],
    )

    meta_path = tmp_path / "signals_style_replica.meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    envelope = read_artifact_envelope(payload, allow_legacy=False)

    assert isinstance(envelope, ArtifactEnvelopeV2)
    assert envelope.run_id == "run-demo"
    assert envelope.artifact_id == "signals_style_replica:run-demo"
    assert envelope.artifact_type == "signals_style_replica.parquet"
    assert envelope.created_at.utcoffset() is not None
    assert envelope.producer.repository == "alpha-research"
    assert envelope.producer.backend == "style_replica"
    assert envelope.content_sha256 == file_sha256(tmp_path / "signals_style_replica.parquet")
    assert len(envelope.lineage) == 1
    assert envelope.lineage[0].artifact_id == "signals.parquet"
    assert not canonical.empty
