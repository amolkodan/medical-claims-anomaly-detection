from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from claims_anomaly.data.io import read_claims_csv
from claims_anomaly.features.build_features import build_features
from claims_anomaly.models.base import ModelArtifact
from claims_anomaly.models.factory import build_model
from claims_anomaly.pipeline.splitting import time_split
from claims_anomaly.utils.config import load_yaml
from claims_anomaly.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anomaly detection model for medical claims")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--input", required=True, help="Path to input claims CSV")
    parser.add_argument("--model-out", required=True, help="Path to write the trained model artifact")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    config = load_yaml(args.config)
    features_config = load_yaml(config["features_config"])

    claims = read_claims_csv(args.input)

    split_cfg = config.get("training", {}).get("split", {})
    if split_cfg.get("type", "time") != "time":
        raise ValueError("Only time split is supported")

    split_result = time_split(
        claims,
        datetime_column=split_cfg["datetime_column"],
        train_end=split_cfg["train_end"],
    )

    feature_output = build_features(split_result.train_frame, features_config)

    model_cfg = config.get("model", {})
    model = build_model(model_cfg["type"], model_cfg.get("params", {}))
    model.fit(feature_output.feature_matrix)

    artifact = ModelArtifact(
        model=model,
        transformer=feature_output.transformer,
        feature_config=features_config,
    )

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_out)


if __name__ == "__main__":
    main()
