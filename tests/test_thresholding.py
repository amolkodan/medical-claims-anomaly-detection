import numpy as np
import pandas as pd

from claims_anomaly.pipeline.thresholding import threshold_top_fraction


def test_threshold_top_fraction_flags_count() -> None:
    scores = pd.Series(np.arange(100, dtype=float))
    result = threshold_top_fraction(scores, top_fraction=0.1)
    assert result.flags.sum() in {10, 11}
