from __future__ import annotations

import pandas as pd
import pytest

from alpha_research.contextual import (
    ExposureSpec,
    FundamentalModifier,
    build_company_exposures,
)


def _stocks() -> pd.DataFrame:
    date = pd.Timestamp("2026-08-28")
    return pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": "BANK",
                "industry": "银行",
                "leverage": 0.90,
                "cash_ratio": 0.10,
            },
            {
                "trade_date": date,
                "symbol": "RE",
                "industry": "房地产",
                "leverage": 0.80,
                "cash_ratio": 0.15,
            },
            {
                "trade_date": date,
                "symbol": "CHEM",
                "industry": "化工",
                "leverage": 0.50,
                "cash_ratio": 0.20,
            },
            {
                "trade_date": date,
                "symbol": "SOFT",
                "industry": "计算机",
                "leverage": 0.20,
                "cash_ratio": 0.60,
            },
            {
                "trade_date": date,
                "symbol": "UNKNOWN",
                "industry": "未映射行业",
                "leverage": 0.40,
                "cash_ratio": 0.40,
            },
        ]
    )


def test_exposure_uses_industry_prior_and_same_date_modifier_then_clips():
    spec = ExposureSpec(
        name="rate_sensitivity",
        industry_prior_map={"银行": 0.4, "房地产": -0.8, "化工": -0.2, "计算机": -0.5},
        fundamental_modifiers=(
            FundamentalModifier(
                field="leverage",
                direction=-1.0,
                weight=0.4,
                normalization="rank_pct",
                missing="ignore_modifier",
            ),
        ),
        unknown_industry="zero_prior",
        version="rate.v1",
    )
    result = build_company_exposures(_stocks(), [spec])
    values = result.set_index("symbol")["exposure_value"].to_dict()

    assert all(-1.0 <= value <= 1.0 for value in values.values())
    assert values["RE"] < -0.8
    assert values["SOFT"] > -0.5
    assert result["exposure_version"].eq("rate.v1").all()
    assert result["source_components"].map(lambda value: value["industry_prior"] is not None).all()


def test_unknown_industry_and_missing_modifier_policies_are_explicit():
    missing_industry = ExposureSpec(
        name="industrial_activity_sensitivity",
        industry_prior_map={"化工": 0.7},
        unknown_industry="missing_exposure",
    )
    result = build_company_exposures(_stocks(), [missing_industry])
    unknown = result.loc[result["symbol"] == "UNKNOWN", "exposure_value"].iloc[0]
    assert pd.isna(unknown)

    missing_modifier = ExposureSpec(
        name="credit_sensitivity",
        industry_prior_map={"银行": 0.6},
        fundamental_modifiers=(
            FundamentalModifier(
                field="missing_field",
                direction=1.0,
                weight=0.5,
                normalization="zscore_clip",
                missing="missing_exposure",
            ),
        ),
        unknown_industry="zero_prior",
    )
    result = build_company_exposures(_stocks(), [missing_modifier])
    assert result["exposure_value"].isna().all()


def test_modifier_normalization_never_uses_other_dates():
    first = _stocks()
    second = _stocks().copy()
    second["trade_date"] = pd.Timestamp("2026-08-29")
    second["leverage"] = [100.0, 200.0, 300.0, 400.0, 500.0]
    frame = pd.concat([first, second], ignore_index=True)
    spec = ExposureSpec(
        name="rate_sensitivity",
        industry_prior_map={
            "银行": 0.0,
            "房地产": 0.0,
            "化工": 0.0,
            "计算机": 0.0,
            "未映射行业": 0.0,
        },
        fundamental_modifiers=(
            FundamentalModifier(
                field="leverage",
                direction=1.0,
                weight=1.0,
                normalization="rank_pct",
                missing="ignore_modifier",
            ),
        ),
        unknown_industry="zero_prior",
    )
    result = build_company_exposures(frame, [spec])
    first_values = (
        result.loc[result["trade_date"] == pd.Timestamp("2026-08-28")]
        .sort_values("symbol")["exposure_value"]
        .tolist()
    )
    second_values = (
        result.loc[result["trade_date"] == pd.Timestamp("2026-08-29")]
        .sort_values("symbol")["exposure_value"]
        .tolist()
    )
    assert first_values != second_values
    assert max(first_values) <= 1.0
    assert max(second_values) <= 1.0


def test_exposure_spec_rejects_invalid_policy_or_bounds():
    with pytest.raises(ValueError, match="unknown_industry"):
        ExposureSpec(name="bad", industry_prior_map={}, unknown_industry="guess")
    with pytest.raises(ValueError, match="clip"):
        ExposureSpec(name="bad", industry_prior_map={}, clip_min=1.0, clip_max=-1.0)
