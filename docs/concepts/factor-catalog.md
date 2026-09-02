# Factor catalog

The factor catalog turns reusable factors into versioned research assets instead of leaving identity implicit in Python function names.

Each `FactorSpec` records:

- stable factor id and explicit version;
- owner and frequency;
- declared dependencies;
- PIT semantics;
- formation-universe semantics;
- preprocessing pipeline;
- implementation SHA-256;
- optional description.

`FactorEvidenceSummary` records dated evidence without embedding raw research output. The first summary fields include IC / rank IC, ICIR, turnover, neutralized rank IC, decay horizon, observation count, and lifecycle status (`research`, `candidate`, `production`, `retired`).

A catalog may contain multiple versions of the same factor. Evidence attaches to one exact `(factor_id, version)` and duplicate evidence dates fail closed.

This design is inspired by the factor lifecycle/productization ideas discussed around RQFactor, while keeping PIT semantics and evidence ownership inside this platform. Alphalens Reloaded can later provide differential tear-sheet checks, but does not become the canonical factor identity store.
