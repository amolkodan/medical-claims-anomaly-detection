from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame


def time_split(
    data_frame: pd.DataFrame,
    datetime_column: str,
    train_end: str,
) -> SplitResult:
    output = data_frame.copy()
    output[datetime_column] = pd.to_datetime(output[datetime_column], errors="coerce")
    cutoff = pd.to_datetime(train_end)
    train_frame = output[output[datetime_column] < cutoff].copy()
    test_frame = output[output[datetime_column] >= cutoff].copy()
    return SplitResult(train_frame=train_frame, test_frame=test_frame)
