from __future__ import annotations

from alpha_research.daily_watch20 import DailyWatch20Config, DailyWatch20Ranker


def test_existing_ranker_accepts_contextual_feature_names_without_new_model_type():
    baseline = DailyWatch20Ranker(
        DailyWatch20Config(features=("momentum_20d", "volatility_20d"))
    )
    contextual = DailyWatch20Ranker(
        DailyWatch20Config(
            features=(
                "momentum_20d",
                "volatility_20d",
                "ctx__shibor_3m_change20",
                "ctx__shibor_3m_change20__x__rate_sensitivity",
            )
        )
    )

    assert baseline.model_type == contextual.model_type
    assert baseline.feature_set_id != contextual.feature_set_id
    assert contextual.config.features[-2:] == (
        "ctx__shibor_3m_change20",
        "ctx__shibor_3m_change20__x__rate_sensitivity",
    )
