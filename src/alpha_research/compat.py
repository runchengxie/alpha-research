from __future__ import annotations

from typing import Any, cast

import numpy as np


def ensure_numpy_nan_alias() -> None:
    """Provide the legacy ``np.NaN`` alias expected by pandas_ta."""
    np_any = cast(Any, np)
    if not hasattr(np, "NaN"):
        np_any.NaN = np.nan
