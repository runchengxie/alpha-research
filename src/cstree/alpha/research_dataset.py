from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ProcessorProvenance:
    name: str
    fit_scope: str
    applied_to: tuple[str, ...]
    input_columns: tuple[str, ...] = ()
    output_columns: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    leakage_safety: str = "documented"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["applied_to"] = list(self.applied_to)
        payload["input_columns"] = list(self.input_columns)
        payload["output_columns"] = list(self.output_columns)
        return payload


@dataclass(frozen=True)
class ResearchDataset:
    raw_panel: pd.DataFrame
    raw_feature_label: pd.DataFrame
    infer_frame: pd.DataFrame
    learn_frame: pd.DataFrame
    backtest_pricing_frame: pd.DataFrame
    feature_cols: tuple[str, ...]
    target_col: str
    train_target_col: str
    date_col: str = "trade_date"
    symbol_col: str = "symbol"
    processors: tuple[ProcessorProvenance, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def frame_counts(self) -> dict[str, int]:
        return {
            "raw_panel_rows": int(len(self.raw_panel)),
            "raw_feature_label_rows": int(len(self.raw_feature_label)),
            "infer_rows": int(len(self.infer_frame)),
            "learn_rows": int(len(self.learn_frame)),
            "backtest_pricing_rows": int(len(self.backtest_pricing_frame)),
        }

    def fetch_learn(self, segment: str = "all") -> pd.DataFrame:
        frame = self._apply_segment(self.learn_frame, segment)
        return frame.copy()

    def fetch_infer(self, segment: str = "all", *, audit: bool = False) -> pd.DataFrame:
        frame = self._apply_segment(self.infer_frame, segment).copy()
        if not audit:
            drop_cols = [
                col for col in {self.target_col, self.train_target_col} if col in frame.columns
            ]
            if drop_cols:
                frame = frame.drop(columns=drop_cols)
        return frame

    def summary(self) -> dict[str, Any]:
        return {
            **self.frame_counts(),
            "date_col": self.date_col,
            "symbol_col": self.symbol_col,
            "target_col": self.target_col,
            "train_target_col": self.train_target_col,
            "feature_count": len(self.feature_cols),
            "processors": [item.to_dict() for item in self.processors],
            "metadata": self.metadata,
        }

    def _apply_segment(self, frame: pd.DataFrame, segment: str) -> pd.DataFrame:
        if segment in {"", "all", None}:
            return frame
        segment_col = "_segment"
        if segment_col not in frame.columns:
            return frame
        return frame[frame[segment_col].astype(str) == str(segment)]


def _processor_records(
    *,
    features: list[str],
    missing_fill_features: list[str],
    feature_missing_method: str,
    feature_missing_add_indicators: bool,
    winsorize_pct: float | None,
    cs_method: str,
    cs_winsorize_pct: float | None,
    train_target: str,
    target: str,
    train_target_transform: str,
    train_target_group_cols: list[str] | None,
    universe_by_date_applied: bool,
    sample_on_rebalance_dates: bool,
    min_symbols_per_date: int,
) -> tuple[ProcessorProvenance, ...]:
    records: list[ProcessorProvenance] = []
    if universe_by_date_applied:
        records.append(
            ProcessorProvenance(
                name="universe_by_date_filter",
                fit_scope="point_in_time_membership",
                applied_to=("raw_feature_label", "infer", "learn"),
                parameters={"pit_safe": True},
                leakage_safety="pit_filter",
            )
        )
    if missing_fill_features:
        records.append(
            ProcessorProvenance(
                name="feature_missing_fill",
                fit_scope="per_date"
                if feature_missing_method == "cross_sectional_median"
                else "constant",
                applied_to=("infer", "learn"),
                input_columns=tuple(missing_fill_features),
                output_columns=tuple(missing_fill_features),
                parameters={
                    "method": feature_missing_method,
                    "add_indicators": feature_missing_add_indicators,
                },
                leakage_safety="per_date_or_constant",
            )
        )
    if winsorize_pct:
        records.append(
            ProcessorProvenance(
                name="target_winsorize",
                fit_scope="per_date",
                applied_to=("learn",),
                input_columns=(target,),
                output_columns=(target,),
                parameters={"winsorize_pct": winsorize_pct},
                leakage_safety="per_date_target_only",
            )
        )
    if cs_method != "none":
        records.append(
            ProcessorProvenance(
                name="cross_sectional_transform",
                fit_scope="per_date",
                applied_to=("infer", "learn"),
                input_columns=tuple(features),
                output_columns=tuple(features),
                parameters={"method": cs_method, "winsorize_pct": cs_winsorize_pct},
                leakage_safety="per_date",
            )
        )
    if train_target != target:
        group_cols = list(train_target_group_cols or ["trade_date"])
        records.append(
            ProcessorProvenance(
                name="train_target_transform",
                fit_scope="per_group" if group_cols != ["trade_date"] else "per_date",
                applied_to=("learn",),
                input_columns=(target,),
                output_columns=(train_target,),
                parameters={"method": train_target_transform, "group_cols": group_cols},
                leakage_safety="label_only",
            )
        )
    if sample_on_rebalance_dates:
        records.append(
            ProcessorProvenance(
                name="sample_on_rebalance_dates",
                fit_scope="calendar_rule",
                applied_to=("infer", "learn"),
                parameters={"enabled": True},
                leakage_safety="calendar_rule",
            )
        )
    records.append(
        ProcessorProvenance(
            name="min_symbols_per_date_filter",
            fit_scope="per_date_count",
            applied_to=("infer", "learn"),
            parameters={"min_symbols_per_date": min_symbols_per_date},
            leakage_safety="cross_sectional_count",
        )
    )
    return tuple(records)


def build_research_dataset_from_modeling_state(
    *,
    raw_panel: pd.DataFrame,
    modeling_state: dict[str, Any],
    backtest_pricing_frame: pd.DataFrame,
    features: list[str],
    target: str,
    train_target: str,
    missing_fill_features: list[str],
    feature_missing_method: str,
    feature_missing_add_indicators: bool,
    winsorize_pct: float | None,
    cs_method: str,
    cs_winsorize_pct: float | None,
    train_target_transform: str,
    train_target_group_cols: list[str] | None,
    universe_by_date_applied: bool,
    sample_on_rebalance_dates: bool,
    min_symbols_per_date: int,
) -> ResearchDataset:
    processors = _processor_records(
        features=features,
        missing_fill_features=missing_fill_features,
        feature_missing_method=feature_missing_method,
        feature_missing_add_indicators=feature_missing_add_indicators,
        winsorize_pct=winsorize_pct,
        cs_method=cs_method,
        cs_winsorize_pct=cs_winsorize_pct,
        train_target=train_target,
        target=target,
        train_target_transform=train_target_transform,
        train_target_group_cols=train_target_group_cols,
        universe_by_date_applied=universe_by_date_applied,
        sample_on_rebalance_dates=sample_on_rebalance_dates,
        min_symbols_per_date=min_symbols_per_date,
    )
    dropped = modeling_state.get("dropped_date_counts")
    metadata = {
        "valid_dates": int(len(modeling_state.get("valid_dates", []))),
        "dropped_dates": int(len(dropped)) if dropped is not None else 0,
        "min_symbols_per_date": int(min_symbols_per_date),
        "universe_by_date_applied": bool(universe_by_date_applied),
    }
    return ResearchDataset(
        raw_panel=raw_panel,
        raw_feature_label=modeling_state["df_features"],
        infer_frame=modeling_state["df_model_all"],
        learn_frame=modeling_state["df_model_all"],
        backtest_pricing_frame=backtest_pricing_frame,
        feature_cols=tuple(features),
        target_col=target,
        train_target_col=train_target,
        processors=processors,
        metadata=metadata,
    )
