"""Dataset schema helpers for cross-sectional outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class DatasetSchema:
    date_col: str
    instrument_col: str
    price_col: str
    label_col: Optional[str]
    tradable_col: Optional[str]
    feature_cols: list[str]
    extra_cols: list[str] | None = None

    def column_order(self) -> list[str]:
        cols: list[str] = [self.date_col, self.instrument_col, self.price_col]
        cols.extend(self.feature_cols)
        cols.extend(self.extra_cols or [])
        if self.label_col:
            cols.append(self.label_col)
        if self.tradable_col:
            cols.append(self.tradable_col)
        return list(dict.fromkeys(cols))

    def to_dict(self) -> dict:
        return {
            "date_col": self.date_col,
            "instrument_col": self.instrument_col,
            "price_col": self.price_col,
            "label_col": self.label_col,
            "tradable_col": self.tradable_col,
            "feature_cols": list(self.feature_cols),
            "extra_cols": list(self.extra_cols or []),
        }


@dataclass
class Dataset:
    frame: pd.DataFrame
    schema: DatasetSchema

    def validate(self) -> None:
        required = self.schema.column_order()
        missing = [col for col in required if col not in self.frame.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

    def as_frame(self) -> pd.DataFrame:
        return self.frame.copy()

    def as_multiindex(self) -> pd.DataFrame:
        idx_cols = [self.schema.date_col, self.schema.instrument_col]
        return self.frame.set_index(idx_cols).sort_index()


def build_dataset(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    *,
    normalize: bool = True,
) -> Dataset:
    if frame is None:
        out = pd.DataFrame(columns=schema.column_order())
        return Dataset(out, schema)
    required = schema.column_order()
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    out = frame[required].copy()
    if normalize and not out.empty:
        out[schema.date_col] = pd.to_datetime(out[schema.date_col], errors="coerce").dt.normalize()
        out = out[out[schema.date_col].notna()].copy()
        out[schema.instrument_col] = out[schema.instrument_col].astype(str).str.strip()
    return Dataset(out, schema)
