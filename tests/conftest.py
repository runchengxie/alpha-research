from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Tests must exercise alpha-research as an independently importable distribution.
# Sibling repositories are integration-test concerns owned by research-workspace.
sys.path.insert(0, str(ROOT / "src"))
