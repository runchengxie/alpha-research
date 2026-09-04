"""Configuration normalization shared by alpha evaluation entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, SupportsInt, cast


def normalize_signal_settings(eval_cfg: Mapping[str, Any]) -> dict[str, Any]:
    signal_direction_mode = str(eval_cfg.get("signal_direction_mode", "fixed")).strip().lower()
    if signal_direction_mode not in {"fixed", "train_ic", "cv_ic"}:
        raise SystemExit("eval.signal_direction_mode must be one of: fixed, train_ic, cv_ic.")

    signal_direction_raw = eval_cfg.get("signal_direction", 1.0)
    signal_direction = float(signal_direction_raw) if signal_direction_raw is not None else 1.0
    if signal_direction == 0:
        raise SystemExit("eval.signal_direction cannot be 0.")

    min_abs_ic_to_flip_raw = eval_cfg.get("min_abs_ic_to_flip", 0.0)
    min_abs_ic_to_flip = (
        float(min_abs_ic_to_flip_raw) if min_abs_ic_to_flip_raw is not None else 0.0
    )
    if min_abs_ic_to_flip < 0:
        raise SystemExit("eval.min_abs_ic_to_flip must be >= 0.")

    return {
        "SIGNAL_DIRECTION_MODE": signal_direction_mode,
        "SIGNAL_DIRECTION": signal_direction,
        "MIN_ABS_IC_TO_FLIP": min_abs_ic_to_flip,
    }


def normalize_permutation_test(eval_cfg: Mapping[str, Any]) -> dict[str, Any]:
    perm_cfg = eval_cfg.get("permutation_test") or {}
    if isinstance(perm_cfg, Mapping):
        enabled = bool(perm_cfg.get("enabled", False))
        runs = int(perm_cfg.get("n_runs", 1))
        seed = perm_cfg.get("seed")
    else:
        enabled = bool(perm_cfg)
        runs = 1
        seed = None
    if seed is not None:
        seed = int(seed)
    if runs < 1:
        enabled = False
    return {
        "PERM_TEST_ENABLED": enabled,
        "PERM_TEST_RUNS": runs,
        "PERM_TEST_SEED": seed,
    }


def normalize_walk_forward_permutation(
    wf_perm_cfg: object,
    *,
    perm_test_runs: int,
    perm_test_seed: object,
) -> tuple[bool, int, object]:
    if isinstance(wf_perm_cfg, Mapping):
        return (
            bool(wf_perm_cfg.get("enabled", False)),
            int(cast(SupportsInt, wf_perm_cfg.get("n_runs", perm_test_runs))),
            wf_perm_cfg.get("seed", perm_test_seed),
        )
    if wf_perm_cfg is None:
        return False, perm_test_runs, perm_test_seed
    return bool(wf_perm_cfg), perm_test_runs, perm_test_seed


def _normalize_text_list(value: object | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_score_postprocess(eval_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize score post-processing options for cross-sectional evaluation."""

    raw_config = eval_cfg.get("score_postprocess")
    if raw_config is None:
        return {
            "SCORE_POSTPROCESS_ENABLED": False,
            "SCORE_POSTPROCESS_METHOD": "none",
            "SCORE_POSTPROCESS_COLUMNS": [],
            "SCORE_POSTPROCESS_STRENGTH": 1.0,
            "SCORE_POSTPROCESS_MIN_OBS": None,
        }
    if not isinstance(raw_config, Mapping):
        raise SystemExit("eval.score_postprocess must be a mapping when provided.")

    method = str(raw_config.get("method", "none")).strip().lower()
    columns = _normalize_text_list(raw_config.get("columns"))
    strength = float(raw_config.get("strength", 1.0))
    min_obs_raw = raw_config.get("min_obs")
    min_obs = int(min_obs_raw) if min_obs_raw is not None else None
    enabled_raw = raw_config.get("enabled")
    enabled = bool(enabled_raw) if enabled_raw is not None else method != "none"

    if method not in {"none", "neutralize", "rank_blend"}:
        raise SystemExit(
            "eval.score_postprocess.method must be one of: none, neutralize, rank_blend."
        )
    if strength < 0 or strength > 1:
        raise SystemExit("eval.score_postprocess.strength must be between 0 and 1.")
    if min_obs is not None and min_obs < 2:
        raise SystemExit("eval.score_postprocess.min_obs must be >= 2.")
    if not enabled or method == "none":
        return {
            "SCORE_POSTPROCESS_ENABLED": enabled,
            "SCORE_POSTPROCESS_METHOD": "none",
            "SCORE_POSTPROCESS_COLUMNS": [],
            "SCORE_POSTPROCESS_STRENGTH": strength,
            "SCORE_POSTPROCESS_MIN_OBS": None,
        }
    if not columns:
        raise SystemExit(f"eval.score_postprocess.columns is required when method={method}.")
    if method == "neutralize":
        required_min_obs = len(columns) + 1
        if min_obs is not None and min_obs < required_min_obs:
            raise SystemExit("eval.score_postprocess.min_obs must be >= len(columns) + 1.")
        min_obs = max(5, required_min_obs) if min_obs is None else min_obs
    else:
        min_obs = None
    return {
        "SCORE_POSTPROCESS_ENABLED": enabled,
        "SCORE_POSTPROCESS_METHOD": method,
        "SCORE_POSTPROCESS_COLUMNS": columns,
        "SCORE_POSTPROCESS_STRENGTH": strength,
        "SCORE_POSTPROCESS_MIN_OBS": min_obs,
    }


__all__ = [
    "normalize_permutation_test",
    "normalize_score_postprocess",
    "normalize_signal_settings",
    "normalize_walk_forward_permutation",
]
