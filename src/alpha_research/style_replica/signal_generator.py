"""Signal generator for the StyleReplica research model.

Orchestrates the daily alpha-signal pipeline:
1. Load price data and build the research universe
2. Compute style factors
3. Compute A-leg and B-leg research scores
4. Attach classification/explanation columns
5. Output the canonical signal artifact

Final sleeve quotas, portfolio buffers, replacement limits, overlap handling and
position weights are intentionally outside this module.

Output schema (extends alpha_research.signals):
    signal_date, symbol, raw_pred, signal_eval, signal_backtest,
    signal_direction, rank, model_version, feature_set_id,
    eligible_for_backtest, eligible_for_live
    + score_a, score_b, leg, theme, industry, selected_reason,
      resvol_pct, beta_pct, liquidity_pct, small_size_pct,
      mom20_pct, mom120_pct
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from research_contracts import attach_artifact_envelope_v2, file_sha256

from ..signal_artifact import (
    CANONICAL_SIGNAL_COLUMNS,
    PRODUCER_BACKEND_STYLE_REPLICA,
    SIGNAL_CONTRACT_NAME,
    SIGNAL_SCHEMA_VERSION,
    build_signal_artifact_frame,
    build_signal_envelope_v2,
    default_signal_run_id,
    signal_artifact_summary,
)
from .factors import compute_all_style_factors
from .score_a import compute_score_a
from .score_b import compute_score_b
from .theme_map import build_theme_map, get_theme_label
from .universe import filter_style_replica_universe

MODEL_VERSION = "StyleReplica-A80B20-v0"
FEATURE_SET_ID = "style_replica_v0"

STYLE_REPLICA_SIGNAL_FILE = "signals_style_replica.parquet"
STYLE_REPLICA_META_FILE = "signals_style_replica.meta.json"


@dataclass
class StyleReplicaConfig:
    """Alpha-signal configuration for StyleReplica research."""

    resvol_window: int = 60
    beta_window: int = 60
    liquidity_window: int = 20
    mom_short_window: int = 20
    mom_long_window: int = 120
    model_version: str = MODEL_VERSION
    feature_set_id: str = FEATURE_SET_ID


def _wide_to_long(
    scores: pd.DataFrame,
    *,
    value_col: str = "score",
    date_col: str = "signal_date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Convert a wide score frame to the canonical long layout."""
    long = scores.stack(future_stack=True).reset_index()
    long.columns = [date_col, symbol_col, value_col]
    long[date_col] = pd.to_datetime(long[date_col]).dt.strftime("%Y%m%d")
    long[symbol_col] = long[symbol_col].astype(str)
    return long


def _build_explanation_columns(
    factor_map: dict[str, pd.DataFrame],
    score_date: pd.Timestamp,
) -> dict[str, pd.Series]:
    """Build per-factor percentile explanation columns for a single date."""
    ts_date = pd.Timestamp(score_date)
    explanations: dict[str, pd.Series] = {}
    factor_to_col = {
        "resvol": "resvol_pct",
        "beta": "beta_pct",
        "liquidity": "liquidity_pct",
        "size": "small_size_pct",
        "mom20": "mom20_pct",
        "mom120": "mom120_pct",
    }

    for factor_name, col_name in factor_to_col.items():
        factor_df = factor_map.get(factor_name)
        if factor_df is None or factor_df.empty or ts_date not in factor_df.index:
            continue
        row = factor_df.loc[ts_date]
        explanations[col_name] = row.rank(pct=True, na_option="bottom")
    return explanations


def _filter_price_panel(
    price_panel: pd.DataFrame,
    instruments: pd.DataFrame | None,
) -> pd.DataFrame:
    if instruments is None:
        return price_panel
    return filter_style_replica_universe(price_panel, instruments, price_panel.index[-1])


def _build_classification_series(
    industry_frame: pd.DataFrame | None,
    concept_frame: pd.DataFrame | None,
) -> tuple[pd.Series | None, pd.Series | None]:
    if industry_frame is None or industry_frame.empty:
        return None, None

    theme_series = build_theme_map(industry_frame, concept_frame=concept_frame)
    industry_series = None
    if "industry_name" in industry_frame.columns:
        industry_series = industry_frame.set_index("symbol")["industry_name"]
    return theme_series, industry_series


def _compute_signal_scores(
    price_panel: pd.DataFrame,
    *,
    turnover_panel: pd.DataFrame | None,
    market_cap_panel: pd.DataFrame | None,
    market_returns: pd.Series | None,
    industry_series: pd.Series | None,
) -> pd.DataFrame:
    factors = compute_all_style_factors(
        price_panel,
        turnover_panel=turnover_panel,
        market_cap_panel=market_cap_panel,
        market_returns=market_returns,
        industry_map=industry_series,
    )
    long_a = _wide_to_long(compute_score_a(factors), value_col="score_a")
    long_b = _wide_to_long(compute_score_b(factors), value_col="score_b")
    return long_a.merge(long_b, on=["signal_date", "symbol"], how="outer")


def _assign_leg(row: pd.Series) -> str | None:
    score_a = row.get("score_a")
    score_b = row.get("score_b")
    theme = row.get("theme")
    if pd.isna(score_a) and pd.isna(score_b):
        return None
    if not pd.isna(theme) and not pd.isna(score_a):
        return "A"
    if not pd.isna(score_b):
        return "B"
    return None


def _decorate_signals(
    signals: pd.DataFrame,
    *,
    config: StyleReplicaConfig,
    theme_series: pd.Series | None,
    industry_series: pd.Series | None,
) -> pd.DataFrame:
    signals["model_version"] = config.model_version
    signals["feature_set_id"] = config.feature_set_id
    signals["signal_direction"] = 1.0
    signals["theme"] = (
        signals["symbol"].map(theme_series.to_dict()) if theme_series is not None else None
    )
    signals["industry"] = (
        signals["symbol"].map(industry_series.to_dict()) if industry_series is not None else None
    )

    signals["raw_pred"] = signals[["score_a", "score_b"]].max(axis=1)
    signals["signal_eval"] = signals["raw_pred"]
    signals["signal_backtest"] = signals["raw_pred"]
    signals["leg"] = signals.apply(_assign_leg, axis=1)
    signals["eligible_for_backtest"] = True
    signals["eligible_for_live"] = True
    signals["selected_reason"] = signals.apply(_build_reason, axis=1)
    return signals


def _order_and_rank_signals(signals: pd.DataFrame) -> pd.DataFrame:
    output_cols = [
        "signal_date",
        "symbol",
        "raw_pred",
        "signal_eval",
        "signal_backtest",
        "signal_direction",
        "rank",
        "model_version",
        "feature_set_id",
        "eligible_for_backtest",
        "eligible_for_live",
        "score_a",
        "score_b",
        "leg",
        "theme",
        "industry",
        "selected_reason",
    ]
    available_cols = [column for column in output_cols if column in signals.columns]
    extra_cols = [column for column in signals.columns if column not in available_cols]
    ordered = signals[available_cols + extra_cols].reset_index(drop=True)
    ordered["rank"] = (
        ordered.groupby("signal_date", sort=False)["raw_pred"]
        .rank(ascending=False, method="first", na_option="bottom")
        .astype("Int64")
    )
    return ordered


def generate_daily_signals(
    price_panel: pd.DataFrame,
    *,
    instruments: pd.DataFrame | None = None,
    turnover_panel: pd.DataFrame | None = None,
    market_cap_panel: pd.DataFrame | None = None,
    market_returns: pd.Series | None = None,
    industry_frame: pd.DataFrame | None = None,
    concept_frame: pd.DataFrame | None = None,
    config: StyleReplicaConfig | None = None,
) -> pd.DataFrame:
    """Generate daily StyleReplica research signals for all usable dates."""
    cfg = config or StyleReplicaConfig()
    filtered_prices = _filter_price_panel(price_panel, instruments)
    if filtered_prices.empty:
        return pd.DataFrame()

    theme_series, industry_series = _build_classification_series(
        industry_frame,
        concept_frame,
    )
    signals = _compute_signal_scores(
        filtered_prices,
        turnover_panel=turnover_panel,
        market_cap_panel=market_cap_panel,
        market_returns=market_returns,
        industry_series=industry_series,
    )
    return _order_and_rank_signals(
        _decorate_signals(
            signals,
            config=cfg,
            theme_series=theme_series,
            industry_series=industry_series,
        )
    )


def _build_reason(row: pd.Series) -> str:
    """Build a human-readable research explanation for a single stock."""
    leg = row.get("leg")
    theme = row.get("theme")
    industry = row.get("industry")

    parts = []
    if leg:
        parts.append(f"{leg}腿")
    if theme is not None and not pd.isna(theme):
        parts.append(get_theme_label(str(theme)))
    if industry is not None and not pd.isna(industry):
        parts.append(str(industry))
    return " / ".join(parts) if parts else "未分类"


class StyleReplicaSignalGenerator:
    """Daily alpha-signal generator for the StyleReplica research model."""

    def __init__(self, config: StyleReplicaConfig | None = None):
        self.config = config or StyleReplicaConfig()

    def generate(
        self,
        price_panel: pd.DataFrame,
        *,
        instruments: pd.DataFrame | None = None,
        turnover_panel: pd.DataFrame | None = None,
        market_cap_panel: pd.DataFrame | None = None,
        market_returns: pd.Series | None = None,
        industry_frame: pd.DataFrame | None = None,
        concept_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate daily StyleReplica research signals."""
        return generate_daily_signals(
            price_panel,
            instruments=instruments,
            turnover_panel=turnover_panel,
            market_cap_panel=market_cap_panel,
            market_returns=market_returns,
            industry_frame=industry_frame,
            concept_frame=concept_frame,
            config=self.config,
        )

    def build_canonical_frame(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Normalize signals into the canonical alpha-research artifact format."""
        if signals is None or signals.empty:
            return pd.DataFrame(columns=pd.Index(CANONICAL_SIGNAL_COLUMNS))
        return build_signal_artifact_frame(
            signals,
            model_version=self.config.model_version,
            feature_set_id=self.config.feature_set_id,
            signal_direction=1.0,
            eligible_for_backtest=True,
            eligible_for_live=True,
        )

    def write(
        self,
        signals: pd.DataFrame,
        output_dir: str | Path,
        *,
        extra_metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
        lineage: Sequence[tuple[str, str]] = (),
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Write the signal artifact and alpha-owned metadata."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        canonical = self.build_canonical_frame(signals)
        signal_path = out_dir / STYLE_REPLICA_SIGNAL_FILE
        canonical.to_parquet(signal_path, index=False)

        summary = signal_artifact_summary(canonical, path=signal_path)
        config = {
            "resvol_window": self.config.resvol_window,
            "beta_window": self.config.beta_window,
            "liquidity_window": self.config.liquidity_window,
            "mom_short_window": self.config.mom_short_window,
            "mom_long_window": self.config.mom_long_window,
            "model_version": self.config.model_version,
            "feature_set_id": self.config.feature_set_id,
        }
        meta_payload = {
            "artifact_type": SIGNAL_CONTRACT_NAME,
            "schema_version": SIGNAL_SCHEMA_VERSION,
            "model_version": self.config.model_version,
            "config": config,
            "summary": summary,
            "metadata": dict(extra_metadata or {}),
        }
        resolved_run_id = run_id or default_signal_run_id()
        envelope = build_signal_envelope_v2(
            artifact_id=f"signals_style_replica:{resolved_run_id}",
            artifact_type=STYLE_REPLICA_SIGNAL_FILE,
            run_id=resolved_run_id,
            content_sha256=file_sha256(signal_path),
            configuration=config,
            producer_backend=PRODUCER_BACKEND_STYLE_REPLICA,
            lineage=lineage,
        )
        meta_payload = attach_artifact_envelope_v2(meta_payload, envelope)
        meta_path = out_dir / STYLE_REPLICA_META_FILE
        meta_path.write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return canonical, summary
