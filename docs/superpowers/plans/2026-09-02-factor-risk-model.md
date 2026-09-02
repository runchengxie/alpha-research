# Factor Risk Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a PIT-safe factor covariance and specific-risk estimate that emits platform-native pandas results.

**Architecture:** Keep factor construction explicit. The risk-model primitive consumes precomputed exposures/factor returns/specific returns, rejects future observations, and projects `X F X' + D` without importing portfolio code.

**Tech Stack:** Python 3.12, pandas, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-factor-risk-model-design.md`

## Global Constraints

- Do not import `portfolio_backtester` or strategy-pipeline.
- All historical rows must be at or before `as_of`.
- No proprietary provider object or data fetcher is introduced.
- Risk-model and optimizer evidence remain separate.

---

### Task 1: Define risk-model tests

**Files:**
- Create: `tests/test_risk_model.py`

- [x] Add tests for projected covariance, positive specific risk, future-data rejection, factor mismatch, and covariance shrinkage.
- [ ] Run `scripts/dev/run_tests.sh` or `uv run --extra dev pytest tests/test_risk_model.py -q` and confirm RED before implementation.

### Task 2: Implement and export the risk model

**Files:**
- Create: `src/alpha_research/risk_model.py`
- Modify: `src/alpha_research/__init__.py`
- Modify: `tests/test_package_smoke.py`

- [x] Implement `FactorRiskModelEstimate` and `build_factor_risk_model()`.
- [x] Enforce finite data, exact factor/asset identity, explicit `as_of`, minimum observations, and optional diagonal shrinkage.
- [x] Export the API and register the owned module in smoke tests.
- [ ] Run `uv run --extra dev pytest tests/test_risk_model.py tests/test_package_smoke.py -q`.

### Task 3: Document and verify

**Files:**
- Create: `docs/concepts/factor-risk-model.md`

- [x] Document ownership, model equation, PIT behavior, and future integration.
- [ ] Run lint, format, typecheck, full tests, and maintainability gates using `scripts/dev/run_tests.sh`.
