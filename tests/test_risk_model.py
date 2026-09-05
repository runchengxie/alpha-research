from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from alpha_research.risk_model import FactorRiskModelEstimate, build_factor_risk_model


def _date(value: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    assert isinstance(result, pd.Timestamp)
    return result


def test_risk_timestamp_rejects_missing_date():
    from alpha_research.risk_model import _comparison_timestamp

    with pytest.raises(ValueError, match="timestamp"):
        _comparison_timestamp(cast(pd.Timestamp, pd.NaT))  # Deliberately invalid runtime input.


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2025-01-02", periods=60, freq="B")
    rng = np.random.default_rng(42)
    exposures = pd.DataFrame(
        {
            "value": [1.0, 0.2, -0.4],
            "momentum": [0.1, 1.1, 0.5],
        },
        index=pd.Index(["000001.SZ", "000002.SZ", "600000.SH"]),
    )
    factor_returns = pd.DataFrame(
        {
            "value": rng.normal(0.0, 0.008, len(dates)),
            "momentum": rng.normal(0.0, 0.010, len(dates)),
        },
        index=dates,
    )
    specific_returns = pd.DataFrame(
        {
            symbol: rng.normal(0.0, 0.012 + i * 0.001, len(dates))
            for i, symbol in enumerate(exposures.index)
        },
        index=dates,
    )
    return exposures, factor_returns, specific_returns


def test_factor_risk_model_builds_asset_covariance_and_receipt() -> None:
    exposures, factor_returns, specific_returns = _inputs()

    estimate = build_factor_risk_model(
        exposures=exposures,
        factor_returns=factor_returns,
        specific_returns=specific_returns,
        as_of=_date("2025-04-01"),
        covariance_shrinkage=0.15,
        min_observations=40,
    )

    covariance = estimate.asset_covariance()
    assert list(covariance.index) == list(exposures.index)
    assert list(covariance.columns) == list(exposures.index)
    assert np.allclose(covariance.to_numpy(), covariance.to_numpy().T)
    assert (np.linalg.eigvalsh(covariance.to_numpy()) > -1e-10).all()
    assert (estimate.specific_risk > 0).all()
    assert estimate.receipt()["schema_version"] == "alpha_research.factor_risk_model.v1"
    assert estimate.receipt()["factor_count"] == 2
    assert estimate.receipt()["asset_count"] == 3


def test_factor_risk_model_rejects_future_observations() -> None:
    exposures, factor_returns, specific_returns = _inputs()

    with pytest.raises(ValueError, match="after as_of"):
        build_factor_risk_model(
            exposures=exposures,
            factor_returns=factor_returns,
            specific_returns=specific_returns,
            as_of=_date("2025-02-01"),
            min_observations=10,
        )


def test_factor_risk_model_rejects_factor_mismatch() -> None:
    exposures, factor_returns, specific_returns = _inputs()

    with pytest.raises(ValueError, match="factor columns"):
        build_factor_risk_model(
            exposures=exposures.rename(columns={"momentum": "size"}),
            factor_returns=factor_returns,
            specific_returns=specific_returns,
            as_of=_date("2025-04-01"),
        )


def test_full_shrinkage_removes_factor_covariance_cross_terms() -> None:
    exposures, factor_returns, specific_returns = _inputs()

    estimate = build_factor_risk_model(
        exposures=exposures,
        factor_returns=factor_returns,
        specific_returns=specific_returns,
        as_of=_date("2025-04-01"),
        covariance_shrinkage=1.0,
    )

    off_diagonal = estimate.factor_covariance.to_numpy().copy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert np.allclose(off_diagonal, 0.0)


def test_manual_estimate_rejects_nonsymmetric_or_indefinite_factor_covariance() -> None:
    exposures, _, _ = _inputs()
    specific_risk = pd.Series(0.1, index=exposures.index)
    base: dict[str, Any] = {
        "as_of": pd.Timestamp("2025-04-01"),
        "exposures": exposures,
        "specific_risk": specific_risk,
        "history_start": pd.Timestamp("2025-01-01"),
        "history_end": pd.Timestamp("2025-03-31"),
        "observations": 60,
        "covariance_shrinkage": 0.0,
    }

    nonsymmetric = FactorRiskModelEstimate(
        factor_covariance=pd.DataFrame(
            [[0.04, 0.02], [0.0, 0.03]],
            index=exposures.columns,
            columns=exposures.columns,
        ),
        **base,
    )
    with pytest.raises(ValueError, match="symmetric"):
        nonsymmetric.validate()

    indefinite = FactorRiskModelEstimate(
        factor_covariance=pd.DataFrame(
            [[0.01, 0.02], [0.02, 0.01]],
            index=exposures.columns,
            columns=exposures.columns,
        ),
        **base,
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        indefinite.validate()


def test_manual_estimate_rejects_history_after_as_of() -> None:
    exposures, _, _ = _inputs()
    estimate = FactorRiskModelEstimate(
        as_of=_date("2025-03-01"),
        exposures=exposures,
        factor_covariance=pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.03]],
            index=exposures.columns,
            columns=exposures.columns,
        ),
        specific_risk=pd.Series(0.1, index=exposures.index),
        history_start=_date("2025-01-01"),
        history_end=_date("2025-03-02"),
        observations=40,
        covariance_shrinkage=0.0,
    )

    with pytest.raises(ValueError, match="history_end"):
        estimate.validate()
