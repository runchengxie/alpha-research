# alpha-research

Alpha and factor research package for the research workspace.

This repository owns `cstree.alpha.*`: feature datasets, feature evidence,
modeling helpers, walk-forward/CPCV/PBO diagnostics, signal artifacts, and
dynamic signal ensemble tooling.

Current status: transitional stage-3 split. The package is physically separated
from `cross-sectional-trees`, and workspace gates prevent runtime imports into
`cstree.pipeline`, `cstree.backtesting`, and strategy-pipeline contract helpers.
Full research runs are still orchestrated by `cross-sectional-trees`, but this
package owns the alpha research layer and should be able to train, diagnose, and
emit signal artifacts without importing portfolio backtesting internals.

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
