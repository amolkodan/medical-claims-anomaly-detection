from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from claims_anomaly.data.io import read_claims_csv, write_csv
from claims_anomaly.features.build_features import transform_features
from claims_anomaly.pipeline.thresholding import threshold_top_fraction
from claims_anomaly.utils.config import load_yaml
from claims_anomaly.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score medical claims with trained anomaly model")
    parser.add_argument("--config", required=True, help="Path to scoring config YAML")
    parser.add_argument("--input", required=True, help="Path to input claims CSV")
    parser.add_argument("--model-in", required=True, help="Path to trained model artifact")
    parser.add_argument("--output", required=True, help="Path to write scored claims CSV")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    config = load_yaml(args.config)
    claims = read_claims_csv(args.input)

    artifact = joblib.load(args.model_in)
    features_config = artifact.feature_config

    feature_matrix, enriched = transform_features(claims, features_config, artifact.transformer)

    raw_scores = artifact.model.score_samples(feature_matrix)
    scores = pd.Series(-raw_scores, index=enriched.index)

    score_col = config.get("scoring", {}).get("output_score_column", "anomaly_score")
    enriched[score_col] = scores.astype(float)

    threshold_cfg = config.get("scoring", {}).get("thresholding", {})
    method = threshold_cfg.get("method", "top_fraction")
    flag_col = config.get("scoring", {}).get("output_flag_column", "review_flag")

    if method == "top_fraction":
        top_fraction = float(threshold_cfg.get("top_fraction", 0.02))
        threshold_result = threshold_top_fraction(enriched[score_col], top_fraction=top_fraction)
        enriched[flag_col] = threshold_result.flags
        enriched["review_threshold"] = threshold_result.threshold
    else:
        raise ValueError(f"Unsupported thresholding method: {method}")

    output_path = Path(args.output)
    write_csv(enriched, output_path)


if __name__ == "__main__":
    main()
