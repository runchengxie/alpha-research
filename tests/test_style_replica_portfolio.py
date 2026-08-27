from __future__ import annotations

import importlib

import pytest

import alpha_research.style_replica as style_replica


def test_style_replica_alpha_surface_contains_no_final_portfolio_owner() -> None:
    assert not hasattr(style_replica, "StyleReplicaPortfolioConfig")
    assert not hasattr(style_replica, "build_style_replica_positions")
    assert not hasattr(style_replica, "compute_daily_changes")
    assert not hasattr(style_replica, "compute_daily_exposure")
    assert not hasattr(style_replica, "compute_style_exposure_summary")


def test_legacy_style_replica_portfolio_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("alpha_research.style_replica.portfolio")
