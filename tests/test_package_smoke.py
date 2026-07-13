from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import cstree
from cstree import alpha

OWNED_MODULES = (
    "cstree.alpha.backends",
    "cstree.alpha.backends.base",
    "cstree.alpha.backends.native",
    "cstree.alpha.signal_artifact",
    "cstree.alpha.benchmarking",
    "cstree.alpha.compat",
    "cstree.alpha.cpcv",
    "cstree.alpha.daily_watch20",
    "cstree.alpha.daily_watch20_features",
    "cstree.alpha.dataset",
    "cstree.alpha.dataset_sampling",
    "cstree.alpha.date_slices",
    "cstree.alpha.pbo",
    "cstree.alpha.overfitting_diagnostics",
    "cstree.alpha.promotion_gate",
    "cstree.alpha.promotion_gate_thresholds",
    "cstree.alpha.recency_diagnostics",
    "cstree.alpha.feature_evidence",
    "cstree.alpha.feature_windows",
    "cstree.alpha.fundamentals",
    "cstree.alpha.freshness_overlay",
    "cstree.alpha.metrics",
    "cstree.alpha.modeling",
    "cstree.alpha.research_dataset",
    "cstree.alpha.research_model",
    "cstree.alpha.return_metrics",
    "cstree.alpha.signal_stability",
    "cstree.alpha.style_replica",
    "cstree.alpha.train_eval_contracts",
    "cstree.alpha.train_eval_diagnostics",
    "cstree.alpha.train_eval_request_builder",
    "cstree.alpha.train_eval_result",
    "cstree.alpha.train_eval_stage",
    "cstree.alpha.walk_forward_windows",
    "cstree.alpha.split",
    "cstree.alpha.transform",
)
FORBIDDEN_RUNTIME_PREFIXES = ("cstree.backtesting", "cstree.pipeline")


def test_cstree_namespace_includes_alpha_package_root() -> None:
    namespace_paths = {Path(path).as_posix() for path in cstree.__path__}

    assert any(path.endswith("alpha-research/src/cstree") for path in namespace_paths)


@pytest.mark.parametrize("module_name", OWNED_MODULES)
def test_owned_alpha_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name


def test_alpha_public_inventory_lists_smoked_modules() -> None:
    public_modules = set(alpha.__all__)

    for module_name in OWNED_MODULES:
        assert module_name.removeprefix("cstree.alpha.") in public_modules


def test_alpha_root_exports_daily_watch20_feature_api() -> None:
    assert alpha.DAILY_WATCH20_FEATURES
    assert alpha.MINUTE_FEATURES
    assert alpha.DailyWatch20FeatureConfig().forward_days == 5
    assert callable(alpha.build_daily_watch20_feature_frame)


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
