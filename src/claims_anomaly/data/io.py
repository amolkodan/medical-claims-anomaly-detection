from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_claims_csv(path: str | Path) -> pd.DataFrame:
    path_obj = Path(path)
    data_frame = pd.read_csv(path_obj)
    return data_frame


def write_csv(data_frame: pd.DataFrame, path: str | Path) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(path_obj, index=False)
