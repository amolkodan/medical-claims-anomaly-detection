from __future__ import annotations

import pandas as pd


def add_datetime_parts(data_frame: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    output = data_frame.copy()
    output[datetime_column] = pd.to_datetime(output[datetime_column], errors="coerce")
    output[f"{datetime_column}_dow"] = output[datetime_column].dt.dayofweek.astype("Int64")
    output[f"{datetime_column}_month"] = output[datetime_column].dt.month.astype("Int64")
    output[f"{datetime_column}_day"] = output[datetime_column].dt.day.astype("Int64")
    output[f"{datetime_column}_is_weekend"] = output[datetime_column].dt.dayofweek.isin([5, 6]).astype(int)
    return output
