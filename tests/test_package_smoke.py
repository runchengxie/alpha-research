from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import alpha_research as alpha

OWNED_MODULES = (
    "alpha_research.backends",
    "alpha_research.backends.base",
    "alpha_research.backends.native",
    "alpha_research.signal_artifact",
    "alpha_research.benchmarking",
    "alpha_research.compat",
    "alpha_research.cpcv",
    "alpha_research.daily_watch20",
    "alpha_research.daily_watch20_features",
    "alpha_research.dataset",
    "alpha_research.dataset_sampling",
    "alpha_research.date_slices",
    "alpha_research.pbo",
    "alpha_research.overfitting_diagnostics",
    "alpha_research.promotion_gate",
    "alpha_research.promotion_gate_thresholds",
    "alpha_research.recency_diagnostics",
    "alpha_research.feature_evidence",
    "alpha_research.feature_windows",
    "alpha_research.fundamentals",
    "alpha_research.freshness_overlay",
    "alpha_research.metrics",
    "alpha_research.minute_friend_factors",
    "alpha_research.modeling",
    "alpha_research.research_dataset",
    "alpha_research.research_model",
    "alpha_research.return_metrics",
    "alpha_research.risk_model",
    "alpha_research.signal_stability",
    "alpha_research.style_replica",
    "alpha_research.train_eval_contracts",
    "alpha_research.train_eval_diagnostics",
    "alpha_research.train_eval_request_builder",
    "alpha_research.train_eval_result",
    "alpha_research.train_eval_stage",
    "alpha_research.walk_forward_windows",
    "alpha_research.split",
    "alpha_research.transform",
)
FORBIDDEN_RUNTIME_PREFIXES = ("portfolio_backtester", "strategy_pipeline.pipeline")


def test_alpha_package_uses_owner_native_root() -> None:
    package_root = Path(alpha.__file__).resolve().parent

    assert package_root.name == "alpha_research"
    assert (package_root / "__init__.py").is_file()


@pytest.mark.parametrize("module_name", OWNED_MODULES)
def test_owned_alpha_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name


def test_alpha_public_inventory_lists_smoked_modules() -> None:
    public_modules = set(alpha.__all__)

    for module_name in OWNED_MODULES:
        assert module_name.removeprefix("alpha_research.") in public_modules


def test_alpha_root_exports_daily_watch20_feature_api() -> None:
    assert alpha.DAILY_WATCH20_FEATURES
    assert alpha.MINUTE_FEATURES
    assert alpha.DailyWatch20FeatureConfig().forward_days == 5
    assert callable(alpha.build_daily_watch20_feature_frame)


def test_alpha_root_exports_risk_model_api() -> None:
    assert alpha.FACTOR_RISK_MODEL_SCHEMA == "alpha_research.factor_risk_model.v1"
    assert callable(alpha.build_factor_risk_model)


def test_owned_alpha_modules_do_not_load_backtesting_or_pipeline() -> None:
    code = f"""
import importlib
import sys

for module_name in {OWNED_MODULES!r}:
    importlib.import_module(module_name)

for prefix in {FORBIDDEN_RUNTIME_PREFIXES!r}:
    offenders = [
        module_name
        for module_name in sys.modules
        if module_name == prefix or module_name.startswith(prefix + ".")
    ]
    if offenders:
        raise SystemExit("loaded forbidden module(s): " + ", ".join(sorted(offenders)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
