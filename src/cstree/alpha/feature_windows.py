from __future__ import annotations

from typing import Any


def parse_feature_windows(features: list[str], prefix: str, suffix: str = "") -> list[int]:
    windows: set[int] = set()
    for feature in features:
        if not feature.startswith(prefix):
            continue
        if suffix and not feature.endswith(suffix):
            continue
        end = len(feature) - len(suffix) if suffix else len(feature)
        value = feature[len(prefix) : end]
        if value.isdigit():
            windows.add(int(value))
    return sorted(windows)


def parse_window_config(raw: Any) -> set[int]:
    windows: set[int] = set()
    if isinstance(raw, (list, tuple, set)):
        values = raw
    elif raw is None:
        values = []
    else:
        values = [raw]
    for item in values:
        try:
            window = int(item)
        except (TypeError, ValueError):
            continue
        if window > 0:
            windows.add(window)
    return windows


_parse_window_config = parse_window_config
