# Factor Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add stable versioned factor identity and dated evidence summaries suitable for workspace publication and Dashboard inspection.

**Spec:** `docs/superpowers/specs/2026-09-02-factor-catalog-design.md`

- [x] Add tests for round-trip, duplicate versions, invalid dependencies/hash, and non-finite evidence.
- [ ] Run focused tests and confirm RED before implementation.
- [x] Implement `FactorSpec`, `FactorEvidenceSummary`, and `FactorCatalog`.
- [x] Document the RQFactor/Alphalens relationship and ownership boundary.
- [ ] Run `uv run --extra dev pytest tests/test_factor_catalog.py -q`.
- [ ] Run repository lint, format, typecheck, full test, and maintainability gates.
