from __future__ import annotations

import pytest

from alpha_research.evaluation_config import (
    normalize_artifact_settings,
    normalize_final_oos,
    normalize_permutation_test,
    normalize_recency_settings,
    normalize_rolling_windows,
    normalize_score_postprocess,
    normalize_signal_settings,
    normalize_walk_forward_permutation,
)


def test_normalize_evaluation_output_settings() -> None:
    assert normalize_rolling_windows({"rolling": {"windows_months": [12, 6]}}) == [6, 12]
    assert normalize_recency_settings({"recency": {"windows": ["1w", "6m"]}}) == [
        "1w",
        "6m",
    ]
    assert normalize_final_oos({"final_oos": {"size": 0.2}}) == {
        "FINAL_OOS_ENABLED": True,
        "FINAL_OOS_SIZE_RAW": 0.2,
    }
    assert normalize_artifact_settings({})["SAVE_ARTIFACTS"] is True


def test_normalize_score_postprocess_defaults_and_neutralize() -> None:
    assert normalize_score_postprocess({}) == {
        "SCORE_POSTPROCESS_ENABLED": False,
        "SCORE_POSTPROCESS_METHOD": "none",
        "SCORE_POSTPROCESS_COLUMNS": [],
        "SCORE_POSTPROCESS_STRENGTH": 1.0,
        "SCORE_POSTPROCESS_MIN_OBS": None,
    }
    settings = normalize_score_postprocess(
        {"score_postprocess": {"method": "neutralize", "columns": ["size"]}}
    )
    assert settings["SCORE_POSTPROCESS_ENABLED"] is True
    assert settings["SCORE_POSTPROCESS_MIN_OBS"] == 5


def test_normalize_score_postprocess_rejects_invalid_config() -> None:
    with pytest.raises(SystemExit, match="columns is required"):
        normalize_score_postprocess(
            {"score_postprocess": {"method": "rank_blend", "enabled": True}}
        )


def test_normalize_signal_settings_defaults_and_aliases() -> None:
    assert normalize_signal_settings({}) == {
        "SIGNAL_DIRECTION_MODE": "fixed",
        "SIGNAL_DIRECTION": 1.0,
        "MIN_ABS_IC_TO_FLIP": 0.0,
    }
    assert (
        normalize_signal_settings(
            {"signal_direction_mode": "cv_ic", "signal_direction": -1, "min_abs_ic_to_flip": 0.2}
        )["SIGNAL_DIRECTION_MODE"]
        == "cv_ic"
    )


def test_normalize_signal_settings_rejects_invalid_values() -> None:
    with pytest.raises(SystemExit, match="signal_direction_mode"):
        normalize_signal_settings({"signal_direction_mode": "random"})
    with pytest.raises(SystemExit, match="cannot be 0"):
        normalize_signal_settings({"signal_direction": 0})


def test_normalize_permutation_and_walk_forward_settings() -> None:
    settings = normalize_permutation_test({"permutation_test": {"enabled": True, "n_runs": 3}})
    assert settings == {
        "PERM_TEST_ENABLED": True,
        "PERM_TEST_RUNS": 3,
        "PERM_TEST_SEED": None,
    }
    assert normalize_walk_forward_permutation(
        {"enabled": True, "n_runs": 5, "seed": 7},
        perm_test_runs=3,
        perm_test_seed=None,
    ) == (True, 5, 7)
