from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from claims_anomaly.data.io import read_claims_csv
from claims_anomaly.utils.config import load_yaml
from claims_anomaly.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anomaly scores against suspected fraud labels")
    parser.add_argument("--config", required=True, help="Path to evaluation config YAML")
    parser.add_argument("--input", required=True, help="Path to scored claims CSV")
    parser.add_argument("--output", required=True, help="Path to write evaluation JSON")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "review_rate": float(y_pred.mean()),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    config = load_yaml(args.config)
    df = read_claims_csv(args.input)

    eval_cfg = config.get("evaluation", {})
    score_col = eval_cfg.get("score_column", "anomaly_score")
    label_col = eval_cfg.get("label_column", "is_fraud_suspected")

    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}. Add labels or use the synthetic generator.")

    y_true = df[label_col].astype(int).values
    scores = df[score_col].astype(float).values

    results: Dict[str, object] = {
        "n": int(len(df)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None,
        "average_precision": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None,
        "threshold_grid": [],
    }

    thresholds: List[float] = list(eval_cfg.get("thresholding", {}).get("grid", [0.01, 0.02, 0.05]))
    results["threshold_grid"] = [metrics_at_threshold(y_true, scores, t) for t in thresholds]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
