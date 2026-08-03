from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml


def _resolve_path(path_text: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if path_text is None:
        return None
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        return (base_dir / candidate).resolve()
    return (Path.cwd() / candidate).resolve()


def _resolve_input_path(
    path_text: str | Path | None, *, base_dir: Path | None = None
) -> Path | None:
    path = _resolve_path(path_text, base_dir=base_dir)
    if path is None or path.exists():
        return path
    candidate = Path(cast("str | Path", path_text)).expanduser()
    if not candidate.is_absolute():
        cwd_path = (Path.cwd() / candidate).resolve()
        if cwd_path.exists():
            return cwd_path
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"YAML file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to parse YAML file: {path} ({exc})") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise SystemExit(f"YAML root must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"JSON file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _set_nested(payload: dict[str, Any], path: str, value: Any) -> None:
    current = payload
    parts = [part for part in str(path).split(".") if part]
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    if parts:
        current[parts[-1]] = value


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _section(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("feature_evidence", config.get("feature_ablation", config))
    if not isinstance(raw, dict):
        raise SystemExit("feature_evidence must be a mapping.")
    return raw


def _families(raw: Any) -> dict[str, list[str]]:
    if isinstance(raw, dict):
        return {
            str(name): [str(item) for item in values]
            for name, values in raw.items()
            if isinstance(values, list)
        }
    if isinstance(raw, list):
        out: dict[str, list[str]] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise SystemExit("feature families list items must be mappings.")
            name = str(item.get("name") or "").strip()
            features = item.get("features")
            if not name or not isinstance(features, list):
                raise SystemExit("Each feature family requires name and features.")
            out[name] = [str(feature) for feature in features]
        return out
    raise SystemExit("feature_evidence.families must be a mapping or list.")


def _features_from_base_config(cfg: dict[str, Any], *, config_dir: Path) -> list[str]:
    base_config_path = _resolve_input_path(cfg.get("base_config"), base_dir=config_dir)
    if base_config_path is None or not base_config_path.exists():
        return []
    base_cfg = _load_yaml(base_config_path)
    features_cfg = cast(
        dict[str, Any],
        base_cfg.get("features") if isinstance(base_cfg.get("features"), dict) else {},
    )
    feature_list = features_cfg.get("list")
    if not isinstance(feature_list, list):
        return []
    return [str(item) for item in feature_list]


def _resolve_feature_list(
    cfg: dict[str, Any],
    *,
    config_dir: Path,
    prefer_base_config: bool,
) -> list[str]:
    features_raw = cfg.get("features")
    if isinstance(features_raw, list):
        return [str(feature) for feature in features_raw]
    if features_raw is not None:
        raise SystemExit("feature_evidence.features must be a list when provided.")

    if prefer_base_config:
        base_features = _features_from_base_config(cfg, config_dir=config_dir)
        if base_features:
            return base_features

    if cfg.get("families"):
        families = _families(cfg.get("families"))
        features = sorted({feature for values in families.values() for feature in values})
        if features:
            return features

    if not prefer_base_config:
        base_features = _features_from_base_config(cfg, config_dir=config_dir)
        if base_features:
            return base_features

    raise SystemExit(
        "feature_evidence.features, feature_evidence.base_config with features.list, "
        "or feature_evidence.families is required."
    )


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name).strip("_")
