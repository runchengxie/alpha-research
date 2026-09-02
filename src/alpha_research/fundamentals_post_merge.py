"""Post-merge fundamental feature derivations owned by alpha-research."""

from __future__ import annotations

import numpy as np
import pandas as pd


def derive_post_merge_fundamental_features(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    log_market_cap: bool = False,
    market_cap_col: str = "total_mv",
    log_market_cap_col: str = "log_mcap",
    auto_add_log_market_cap: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Derive panel-level fundamental features after owner data has been joined.

    The input frame is mutated in place to preserve the pipeline's current memory behavior.
    The returned feature list is de-duplicated only when a derived log-market-cap feature is
    auto-added.
    """

    features = list(feature_names)
    trade_dates = (
        pd.to_datetime(frame["trade_date"], errors="coerce")
        if "trade_date" in frame.columns
        else None
    )

    if (
        "days_since_report" in features
        and trade_dates is not None
        and "report_trade_date" in frame.columns
    ):
        report_dates = pd.to_datetime(frame["report_trade_date"], errors="coerce")
        frame["days_since_report"] = (trade_dates - report_dates).dt.days

    if (
        "valuation_age_days" in features
        and trade_dates is not None
        and "valuation_trade_date" in frame.columns
    ):
        valuation_dates = pd.to_datetime(frame["valuation_trade_date"], errors="coerce")
        frame["valuation_age_days"] = (trade_dates - valuation_dates).dt.days

    if log_market_cap and market_cap_col in frame.columns:
        market_cap = pd.to_numeric(frame[market_cap_col], errors="coerce")
        frame[log_market_cap_col] = np.where(
            market_cap > 0,
            np.log(market_cap),
            np.nan,
        )
        if auto_add_log_market_cap and log_market_cap_col not in features:
            features = list(dict.fromkeys([*features, log_market_cap_col]))

    return frame, features
