from __future__ import annotations

from .exposures import ExposureSpec, FundamentalModifier

# Coarse research priors only. They intentionally stop at industry level and remain
# versioned evidence rather than pretending to know a precise sensitivity for every stock.
_RATE_PRIORS = {
    "银行": 0.35,
    "非银金融": 0.30,
    "房地产": -0.80,
    "建筑装饰": -0.45,
    "建筑材料": -0.40,
    "公用事业": -0.35,
    "电力设备": -0.35,
    "电子": -0.35,
    "计算机": -0.50,
    "传媒": -0.40,
    "通信": -0.30,
}

_CREDIT_PRIORS = {
    "银行": 0.55,
    "非银金融": 0.50,
    "房地产": 0.90,
    "建筑装饰": 0.65,
    "建筑材料": 0.55,
    "机械设备": 0.45,
    "汽车": 0.40,
    "家用电器": 0.35,
    "商贸零售": 0.35,
}

_INDUSTRIAL_PRIORS = {
    "钢铁": 0.85,
    "有色金属": 0.80,
    "基础化工": 0.75,
    "机械设备": 0.75,
    "建筑材料": 0.70,
    "建筑装饰": 0.60,
    "煤炭": 0.60,
    "交通运输": 0.45,
    "汽车": 0.45,
    "电子": 0.35,
}

_ENERGY_INPUT_PRIORS = {
    "航空机场": -0.90,
    "基础化工": -0.70,
    "钢铁": -0.65,
    "建筑材料": -0.60,
    "有色金属": -0.55,
    "造纸": -0.50,
    "机械设备": -0.30,
    "汽车": -0.25,
}

_ENERGY_OUTPUT_PRIORS = {
    "煤炭": 0.95,
    "石油石化": 0.90,
    "公用事业": 0.65,
    "电力设备": 0.30,
}


def default_context_exposure_specs() -> tuple[ExposureSpec, ...]:
    return (
        ExposureSpec(
            name="rate_sensitivity",
            industry_prior_map=_RATE_PRIORS,
            fundamental_modifiers=(
                FundamentalModifier(
                    field="leverage",
                    direction=-1.0,
                    weight=0.25,
                    normalization="rank_pct",
                    missing="ignore_modifier",
                ),
            ),
            unknown_industry="zero_prior",
            version="rate.v1",
        ),
        ExposureSpec(
            name="credit_sensitivity",
            industry_prior_map=_CREDIT_PRIORS,
            fundamental_modifiers=(
                FundamentalModifier(
                    field="leverage",
                    direction=1.0,
                    weight=0.20,
                    normalization="rank_pct",
                    missing="ignore_modifier",
                ),
                FundamentalModifier(
                    field="cash_to_assets",
                    direction=-1.0,
                    weight=0.15,
                    normalization="rank_pct",
                    missing="ignore_modifier",
                ),
            ),
            unknown_industry="zero_prior",
            version="credit.v1",
        ),
        ExposureSpec(
            name="industrial_activity_sensitivity",
            industry_prior_map=_INDUSTRIAL_PRIORS,
            unknown_industry="zero_prior",
            version="industrial.v1",
        ),
        ExposureSpec(
            name="energy_input_sensitivity",
            industry_prior_map=_ENERGY_INPUT_PRIORS,
            unknown_industry="zero_prior",
            version="energy-input.v1",
        ),
        ExposureSpec(
            name="energy_output_sensitivity",
            industry_prior_map=_ENERGY_OUTPUT_PRIORS,
            unknown_industry="zero_prior",
            version="energy-output.v1",
        ),
    )
