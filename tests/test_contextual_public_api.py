from __future__ import annotations

import alpha_research
from alpha_research.contextual import default_context_exposure_specs


def test_contextual_core_is_available_from_public_package():
    for name in (
        "ContextTransformSpec",
        "ExposureSpec",
        "FundamentalModifier",
        "ContextInteractionSpec",
        "build_context_features",
        "build_company_exposures",
        "attach_context_as_of",
        "build_context_interactions",
        "contextual_feature_set_id",
    ):
        assert getattr(alpha_research, name) is not None


def test_default_context_exposure_specs_are_versioned_and_stable():
    specs = default_context_exposure_specs()
    assert [spec.name for spec in specs] == [
        "rate_sensitivity",
        "credit_sensitivity",
        "industrial_activity_sensitivity",
        "energy_input_sensitivity",
        "energy_output_sensitivity",
    ]
    assert all(spec.version != "v1" for spec in specs)
    assert all(spec.clip_min == -1.0 and spec.clip_max == 1.0 for spec in specs)
    assert all(spec.unknown_industry == "zero_prior" for spec in specs)
