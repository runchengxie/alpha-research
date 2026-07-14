from __future__ import annotations

from alpha_research.feature_windows import parse_feature_windows, parse_window_config


def test_parse_feature_windows_extracts_prefixed_numeric_windows() -> None:
    assert parse_feature_windows(
        ["sma_5", "sma_20_diff", "sma_fast", "ret_5"],
        "sma_",
    ) == [5]
    assert parse_feature_windows(
        ["sma_5", "sma_20_diff", "sma_fast", "ret_5"],
        "sma_",
        "_diff",
    ) == [20]


def test_parse_window_config_filters_invalid_and_non_positive_values() -> None:
    assert parse_window_config(["5", 10, 0, -3, "bad", None]) == {5, 10}
