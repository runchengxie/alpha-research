from __future__ import annotations

from functools import reduce
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .dynamic_signal_ensemble_types import DynamicSignalEnsembleConfig


def _resolve_path(path_text: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if path_text is None:
        return None
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        return (base_dir / candidate).resolve()
    return (Path.cwd() / candidate).resolve()


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    return pd.read_csv(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Dynamic signal ensemble config must be a mapping: {path}")
    return payload


def _section(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("dynamic_signal_ensemble", config)
    if not isinstance(raw, dict):
        raise SystemExit("dynamic_signal_ensemble must be a mapping.")
    return raw


def _first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def _coerce_date_column(values: pd.Series) -> pd.Series:
    text = values.astype(str)
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce")
    return parsed.dt.normalize()


def _normalize_long_frame(
    frame: pd.DataFrame,
    *,
    date_col: str,
    symbol_col: str,
) -> pd.DataFrame:
    if date_col not in frame.columns:
        raise SystemExit(f"Missing date column for dynamic ensemble: {date_col}")
    if symbol_col not in frame.columns:
        raise SystemExit(f"Missing symbol column for dynamic ensemble: {symbol_col}")
    out = frame.copy()
    out[date_col] = _coerce_date_column(out[date_col])
    out[symbol_col] = out[symbol_col].astype(str)
    return out.dropna(subset=[date_col, symbol_col])


def _pivot_panel(
    data: pd.DataFrame,
    *,
    date_col: str,
    symbol_col: str,
    value_col: str,
) -> pd.DataFrame:
    if value_col not in data.columns:
        raise SystemExit(f"Missing dynamic ensemble value column: {value_col}")
    panel = data.pivot_table(
        index=date_col,
        columns=symbol_col,
        values=value_col,
        aggfunc="last",
    )
    return panel.sort_index().sort_index(axis=1).astype(float)


def _panel_from_long(
    data: pd.DataFrame,
    *,
    date_col: str,
    symbol_col: str,
    columns: list[str],
) -> dict[str, pd.DataFrame]:
    return {
        column: _pivot_panel(data, date_col=date_col, symbol_col=symbol_col, value_col=column)
        for column in columns
    }


def _config_from_mapping(raw: Any) -> DynamicSignalEnsembleConfig:
    if raw is None:
        return DynamicSignalEnsembleConfig()
    if not isinstance(raw, dict):
        raise SystemExit("dynamic ensemble config must be a mapping.")
    allowed = set(DynamicSignalEnsembleConfig.__dataclass_fields__)
    values = {key: value for key, value in raw.items() if key in allowed}
    return DynamicSignalEnsembleConfig(**values)


def _load_regime_features(cfg: dict[str, Any], *, config_dir: Path) -> pd.DataFrame | None:
    path = _resolve_path(cfg.get("regime_features_file"), base_dir=config_dir)
    if path is None:
        return None
    frame = _load_table(path)
    date_col = str(
        cfg.get("regime_date_col") or _first_existing(frame.columns, ("trade_date", "date"))
    )
    if not date_col or date_col not in frame.columns:
        raise SystemExit("Regime features require a trade_date or date column.")
    out = frame.copy()
    out[date_col] = _coerce_date_column(out[date_col])
    out = out.set_index(date_col).sort_index()
    return out.apply(pd.to_numeric, errors="coerce")


def _load_from_signal_files(cfg: dict[str, Any], *, config_dir: Path) -> pd.DataFrame:
    returns_path = _resolve_path(cfg.get("returns_file"), base_dir=config_dir)
    if returns_path is None:
        raise SystemExit("dynamic_signal_ensemble.returns_file is required with signal_files.")
    returns = _load_table(returns_path)
    returns_date_col = str(
        cfg.get("returns_date_col")
        or _first_existing(returns.columns, ("trade_date", "date", "signal_date"))
    )
    returns_symbol_col = str(cfg.get("returns_symbol_col") or "symbol")
    returns_col = str(cfg.get("returns_col") or cfg.get("target_col") or "future_return")
    returns = _normalize_long_frame(
        returns,
        date_col=returns_date_col,
        symbol_col=returns_symbol_col,
    )[[returns_date_col, returns_symbol_col, returns_col]].rename(
        columns={returns_date_col: "trade_date", returns_symbol_col: "symbol"}
    )

    signal_specs = cfg.get("signal_files")
    if not isinstance(signal_specs, list) or not signal_specs:
        raise SystemExit("dynamic_signal_ensemble.signal_files must be a non-empty list.")
    frames: list[pd.DataFrame] = [returns]
    for idx, item in enumerate(signal_specs, start=1):
        spec = {"path": item} if isinstance(item, str) else item
        if not isinstance(spec, dict):
            raise SystemExit(f"signal_files[{idx}] must be a string or mapping.")
        path = _resolve_path(spec.get("path"), base_dir=config_dir)
        if path is None:
            raise SystemExit(f"signal_files[{idx}] is missing path.")
        signal = _load_table(path)
        date_col = str(
            spec.get("date_col")
            or _first_existing(signal.columns, ("signal_date", "trade_date", "date"))
        )
        symbol_col = str(spec.get("symbol_col") or "symbol")
        score_col = str(spec.get("score_col") or cfg.get("score_col") or "signal_backtest")
        name = str(spec.get("name") or path.parent.name or path.stem)
        signal = _normalize_long_frame(signal, date_col=date_col, symbol_col=symbol_col)
        frames.append(
            signal[[date_col, symbol_col, score_col]]
            .rename(columns={date_col: "trade_date", symbol_col: "symbol", score_col: name})
            .drop_duplicates(["trade_date", "symbol"], keep="last")
        )
    return reduce(
        lambda left, right: left.merge(right, on=["trade_date", "symbol"], how="outer"),
        frames,
    )
