from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    flags: np.ndarray


def threshold_top_fraction(scores: pd.Series, top_fraction: float) -> ThresholdResult:
    if not (0.0 < top_fraction < 1.0):
        raise ValueError("top_fraction must be between 0 and 1")

    n = len(scores)
    k = max(1, int(round(n * top_fraction)))
    threshold = float(np.partition(scores.values, -k)[-k])
    flags = (scores.values >= threshold).astype(int)
    return ThresholdResult(threshold=threshold, flags=flags)
