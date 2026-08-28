from __future__ import annotations

from alpha_research.contextual import (
    ContextInteractionSpec,
    ContextTransformSpec,
    ExposureSpec,
    contextual_feature_set_id,
)


def _specs(industry_map):
    transforms = [
        ContextTransformSpec(
            series_id="rates.shibor_3m",
            transform="change_np",
            window=20,
            minimum_history=21,
            feature_name="ctx__shibor_3m_change20",
        )
    ]
    exposures = [
        ExposureSpec(
            name="rate_sensitivity",
            industry_prior_map=industry_map,
            version="rate.v1",
        )
    ]
    interactions = [
        ContextInteractionSpec(
            context_feature="ctx__shibor_3m_change20",
            exposure_name="rate_sensitivity",
            output_name="ctx__shibor_3m_change20__x__rate_sensitivity",
        )
    ]
    return transforms, exposures, interactions


def test_contextual_feature_set_identity_is_canonical_for_mapping_order():
    first = _specs({"银行": 0.4, "房地产": -0.8})
    second = _specs({"房地产": -0.8, "银行": 0.4})
    assert contextual_feature_set_id(*first) == contextual_feature_set_id(*second)


def test_contextual_feature_set_identity_changes_with_semantics():
    base = _specs({"银行": 0.4, "房地产": -0.8})
    changed_transform = (
        [
            ContextTransformSpec(
                series_id="rates.shibor_3m",
                transform="change_np",
                window=10,
                minimum_history=11,
                feature_name="ctx__shibor_3m_change10",
            )
        ],
        base[1],
        [
            ContextInteractionSpec(
                context_feature="ctx__shibor_3m_change10",
                exposure_name="rate_sensitivity",
                output_name="ctx__shibor_3m_change10__x__rate_sensitivity",
            )
        ],
    )
    changed_exposure = (
        base[0],
        [
            ExposureSpec(
                name="rate_sensitivity",
                industry_prior_map={"银行": 0.4, "房地产": -0.8},
                version="rate.v2",
            )
        ],
        base[2],
    )
    assert contextual_feature_set_id(*base) != contextual_feature_set_id(*changed_transform)
    assert contextual_feature_set_id(*base) != contextual_feature_set_id(*changed_exposure)


def test_contextual_feature_set_id_is_sha256_hex():
    value = contextual_feature_set_id(*_specs({"银行": 0.4}))
    assert len(value) == 64
    assert all(char in "0123456789abcdef" for char in value)
