from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import cstree
from cstree import alpha

OWNED_MODULES = (
    "cstree.alpha.signal_artifact",
    "cstree.alpha.cpcv",
    "cstree.alpha.pbo",
    "cstree.alpha.overfitting_diagnostics",
    "cstree.alpha.feature_evidence",
    "cstree.alpha.modeling",
    "cstree.alpha.train_eval_stage",
)


def test_cstree_namespace_reaches_workspace_siblings() -> None:
    namespace_paths = {Path(path).as_posix() for path in cstree.__path__}

    assert any(path.endswith("alpha-research/src/cstree") for path in namespace_paths)
    assert any(path.endswith("cross-sectional-trees/src/cstree") for path in namespace_paths)
    assert any(path.endswith("portfolio-backtester/src/cstree") for path in namespace_paths)


@pytest.mark.parametrize("module_name", OWNED_MODULES)
def test_owned_alpha_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name


def test_alpha_public_inventory_lists_smoked_modules() -> None:
    public_modules = set(alpha.__all__)

    for module_name in OWNED_MODULES:
        assert module_name.removeprefix("cstree.alpha.") in public_modules
