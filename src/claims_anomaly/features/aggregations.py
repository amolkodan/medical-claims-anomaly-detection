from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _rolling_aggregate(
    data_frame: pd.DataFrame,
    entity_column: str,
    datetime_column: str,
    value_columns: Iterable[str],
    window_days: int,
    prefix: str,
) -> pd.DataFrame:
    output = data_frame.copy()
    output[datetime_column] = pd.to_datetime(output[datetime_column], errors="coerce")

    output = output.sort_values([entity_column, datetime_column])
    grouped = output.groupby(entity_column, sort=False)

    for value_column in value_columns:
        series = grouped.apply(
            lambda df: df.set_index(datetime_column)[value_column]
            .rolling(f"{window_days}D", min_periods=1)
            .agg(["mean", "sum", "count"])  
            .reset_index(drop=True)
        )
        series = series.reset_index(level=0, drop=True)
        output[f"{prefix}_{value_column}_mean_{window_days}d"] = series["mean"].astype(float)
        output[f"{prefix}_{value_column}_sum_{window_days}d"] = series["sum"].astype(float)
        output[f"{prefix}_{value_column}_count_{window_days}d"] = series["count"].astype(float)

    output = output.replace([np.inf, -np.inf], np.nan)
    return output


def add_entity_aggregates(
    data_frame: pd.DataFrame,
    provider_column: str,
    member_column: str,
    datetime_column: str,
    numeric_value_columns: Iterable[str],
    provider_window_days: int,
    member_window_days: int,
    include_provider: bool,
    include_member: bool,
) -> pd.DataFrame:
    output = data_frame.copy()

    if include_provider:
        output = _rolling_aggregate(
            output,
            entity_column=provider_column,
            datetime_column=datetime_column,
            value_columns=numeric_value_columns,
            window_days=provider_window_days,
            prefix="provider",
        )

    if include_member:
        output = _rolling_aggregate(
            output,
            entity_column=member_column,
            datetime_column=datetime_column,
            value_columns=numeric_value_columns,
            window_days=member_window_days,
            prefix="member",
        )

    return output
