# Signal Drift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Produce platform-owned signal population drift diagnostics that can later be published to Dashboard research evidence.

**Spec:** `docs/superpowers/specs/2026-09-02-signal-drift-design.md`

- [x] Add tests for identical, shifted, non-finite-filtered, and constant-reference populations.
- [ ] Run focused tests and confirm RED before implementation.
- [x] Implement `SignalDriftReport` and `summarize_signal_drift()` with PSI, KS, mean shift, and std ratio.
- [x] Document the boundary between drift metrics and lifecycle decisions.
- [ ] Run `uv run --extra dev pytest tests/test_signal_drift.py -q`.
- [ ] Run repository lint, format, typecheck, full test, and maintainability gates.
