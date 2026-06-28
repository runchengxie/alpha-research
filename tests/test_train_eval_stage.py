from __future__ import annotations

import numpy as np
import pandas as pd
from cstree.pipeline.contracts import TrainEvalData

from cstree.alpha.train_eval_stage import _industry_exposure_columns


def _empty_train_eval_data(
    *,
    passthrough_cols: list[str],
    industry_keep_columns: list[str],
) -> TrainEvalData:
    return TrainEvalData(
        train_df=pd.DataFrame(),
        test_df=pd.DataFrame(),
        test_dates=np.array([]),
        df_features=pd.DataFrame(),
        df_full=pd.DataFrame(),
        df_model_sorted=pd.DataFrame(),
        all_dates=np.array([]),
        all_date_start_rows=np.array([]),
        all_date_end_rows=np.array([]),
        all_date_to_pos={},
        valid_dates_set=set(),
        backtest_pricing_df=pd.DataFrame(),
        benchmark_df=None,
        benchmark_return_series=pd.Series(dtype=float),
        industry_source_df=pd.DataFrame(),
        passthrough_cols=passthrough_cols,
        industry_keep_columns=industry_keep_columns,
        price_passthrough_cols=[],
        bucket_cols=[],
    )


def test_industry_exposure_columns_prioritize_industry_labels_over_pit_metadata() -> None:
    data = _empty_train_eval_data(
        passthrough_cols=[
            "report_period",
            "disclosure_date",
            "available_date",
            "industry_name",
            "industry_system",
        ],
        industry_keep_columns=["industry_name", "industry_code"],
    )

    assert _industry_exposure_columns(data) == [
        "industry_name",
        "industry_code",
        "industry_system",
    ]
