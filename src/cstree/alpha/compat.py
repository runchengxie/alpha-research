from __future__ import annotations

import numpy as np


def ensure_numpy_nan_alias() -> None:
    """Provide the legacy ``np.NaN`` alias expected by pandas_ta."""
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
