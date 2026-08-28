from __future__ import annotations

import pandas as pd
import pytest

from alpha_research.contextual import (
    ContextInteractionSpec,
    attach_context_as_of,
    build_context_interactions,
)


def test_context_join_uses_later_of_release_and_retrieval_as_visibility():
    stocks = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-15", "2026-01-16", "2026-01-21"], utc=True),
            "symbol": ["A", "A", "A"],
        }
    )
    context = pd.DataFrame(
        {
            "period_end": pd.to_datetime(["2025-12-31", "2026-01-10"], utc=True),
            "available_at": pd.to_datetime(["2026-01-10", "2026-01-12"], utc=True),
            "source_retrieved_at": pd.to_datetime(["2026-01-10", "2026-01-20"], utc=True),
            "ctx__rate": [1.0, 2.0],
        }
    )

    joined = attach_context_as_of(stocks, context, feature_names=["ctx__rate"])
    assert joined["ctx__rate"].tolist() == [1.0, 1.0, 2.0]
    assert joined["ctx__rate__age_days"].tolist() == pytest.approx([15.0, 16.0, 11.0])


def test_context_join_enforces_feature_age_limit_without_deleting_stock_rows():
    stocks = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-10", "2026-02-20"], utc=True),
            "symbol": ["A", "A"],
        }
    )
    context = pd.DataFrame(
        {
            "period_end": pd.to_datetime(["2025-12-31"], utc=True),
            "available_at": pd.to_datetime(["2026-01-05"], utc=True),
            "source_retrieved_at": pd.to_datetime(["2026-01-05"], utc=True),
            "ctx__state": [3.0],
        }
    )
    joined = attach_context_as_of(
        stocks,
        context,
        feature_names=["ctx__state"],
        series_age_limits={"ctx__state": 30},
    )
    assert joined.iloc[0]["ctx__state"] == pytest.approx(3.0)
    assert pd.isna(joined.iloc[1]["ctx__state"])
    assert len(joined) == 2


def test_interaction_requires_exact_date_exposure_and_preserves_missingness():
    stocks = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-20", "2026-01-21"], utc=True),
            "symbol": ["A", "A"],
        }
    )
    context = pd.DataFrame(
        {
            "period_end": pd.to_datetime(["2025-12-31"], utc=True),
            "available_at": pd.to_datetime(["2026-01-10"], utc=True),
            "source_retrieved_at": pd.to_datetime(["2026-01-10"], utc=True),
            "ctx__shibor_3m_change20": [0.5],
        }
    )
    exposures = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-21"], utc=True),
            "symbol": ["A"],
            "exposure_name": ["rate_sensitivity"],
            "exposure_value": [-0.8],
            "exposure_version": ["rate.v1"],
        }
    )
    spec = ContextInteractionSpec(
        context_feature="ctx__shibor_3m_change20",
        exposure_name="rate_sensitivity",
        output_name="ctx__shibor_3m_change20__x__rate_sensitivity",
    )
    result = build_context_interactions(stocks, context, exposures, [spec])

    assert pd.isna(result.iloc[0][spec.output_name])
    assert result.iloc[1][spec.output_name] == pytest.approx(-0.4)


def test_duplicate_interaction_output_names_are_rejected():
    spec = ContextInteractionSpec(
        context_feature="ctx__a",
        exposure_name="rate_sensitivity",
        output_name="ctx__same",
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_context_interactions(
            pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(["2026-01-01"], utc=True),
                    "symbol": ["A"],
                }
            ),
            pd.DataFrame(
                {
                    "period_end": pd.to_datetime(["2025-12-31"], utc=True),
                    "available_at": pd.to_datetime(["2026-01-01"], utc=True),
                    "source_retrieved_at": pd.to_datetime(["2026-01-01"], utc=True),
                    "ctx__a": [1.0],
                }
            ),
            pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(["2026-01-01"], utc=True),
                    "symbol": ["A"],
                    "exposure_name": ["rate_sensitivity"],
                    "exposure_value": [1.0],
                }
            ),
            [spec, spec],
        )
