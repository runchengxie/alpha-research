# alpha-research

Alpha and factor research package for the research workspace.

This repository owns `cstree.alpha.*`: feature datasets, feature evidence,
modeling helpers, walk-forward/CPCV/PBO diagnostics, signal artifacts, and
dynamic signal ensemble tooling.

Current status: transitional stage-3 split. The package is physically separated
from `cross-sectional-trees`, but some modules still import shared workspace
helpers from `cstree.pipeline`, `cstree.contracts`, `cstree.benchmarking`, and
`cstree.backtesting`. Run it from `research-workspace` with the sibling
submodules checked out until those shared interfaces are extracted.

## Local checks

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev ty check
uv run --extra dev pytest
```

Release/advisory check:

```bash
uv run --extra dev basedpyright
```
