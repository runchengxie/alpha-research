from __future__ import annotations

import pytest

from alpha_research.walk_forward import _evaluate_injected_walk_forward_backtest


def test_walk_forward_backtest_requires_injected_evaluator() -> None:
    with pytest.raises(SystemExit, match="walk_forward_backtest_fn"):
        _evaluate_injected_walk_forward_backtest(
            {},
            model_w=object(),
            direction=1.0,
            context={
                "wf_backtest_enabled": True,
                "backtest_topk_fn": lambda *args, **kwargs: None,
                "valid_dates_set": set(),
            },
        )


def test_walk_forward_backtest_uses_injected_evaluator() -> None:
    calls = {}

    def fake_topk(*args, **kwargs):
        return None

    def fake_evaluator(window_meta, **kwargs):
        calls["window_meta"] = window_meta
        calls.update(kwargs)
        return {"ok": True}, {"benchmark": True}, {"active": True}

    result = _evaluate_injected_walk_forward_backtest(
        {"window": 1},
        model_w=object(),
        direction=-1.0,
        context={
            "wf_backtest_enabled": True,
            "walk_forward_backtest_fn": fake_evaluator,
            "backtest_topk_fn": fake_topk,
            "valid_dates_set": {"2024-01-05"},
        },
    )

    assert result == ({"ok": True}, {"benchmark": True}, {"active": True})
    assert calls["window_meta"] == {"window": 1}
    assert calls["direction"] == -1.0
    assert calls["backtest_topk_fn"] is fake_topk
