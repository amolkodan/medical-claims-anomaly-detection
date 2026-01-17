# Medical Claims Anomaly Detection for Healthcare Fraud Detection

This repository provides an end to end framework to detect anomalous medical claims that may indicate fraud, waste, or abuse. It includes:

- A reproducible feature engineering pipeline for claim, member, provider, and service line level signals
- Multiple anomaly detection models (Isolation Forest and One Class SVM) with consistent training and scoring interfaces
- Thresholding and calibration utilities to convert anomaly scores into review queues
- A synthetic claims generator to create a realistic dataset for development and demos
- CLI workflows for train, score, and evaluate

## Repository layout

- `data/raw` and `data/processed`: local datasets (not committed)
- `configs`: YAML configurations for data, features, and models
- `scripts`: dataset generation and helper scripts
- `src/claims_anomaly`: Python package
- `tests`: unit tests

## Quick start

### 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional deep learning model support:

```bash
pip install -r requirements-ml.txt
```

### 2) Generate synthetic medical claims

```bash
python scripts/generate_synthetic_claims.py --config configs/synthetic_data.yaml --output data/raw/claims.csv
```

### 3) Train an anomaly detection model

```bash
python -m claims_anomaly.pipeline.train \
  --config configs/train_isolation_forest.yaml \
  --input data/raw/claims.csv \
  --model-out models/isolation_forest.joblib
```

### 4) Score claims and create a review queue

```bash
python -m claims_anomaly.pipeline.score \
  --config configs/score.yaml \
  --input data/raw/claims.csv \
  --model-in models/isolation_forest.joblib \
  --output data/processed/scored_claims.csv
```

### 5) Evaluate and select a threshold

```bash
python -m claims_anomaly.pipeline.evaluate \
  --config configs/evaluate.yaml \
  --input data/processed/scored_claims.csv \
  --output data/processed/evaluation.json
```

## Data schema

The pipeline expects a claim level table with one row per claim line or claim header. The synthetic generator produces the following columns:

- `claim_id`, `member_id`, `provider_id`, `service_date`
- `procedure_code`, `diagnosis_code`, `place_of_service`, `billing_type`
- `allowed_amount`, `paid_amount`, `units`, `days_supply`
- `member_age`, `member_gender`

You can extend the schema by updating the feature config and feature builders.

## Modeling approach

Anomaly detection is a practical entry point when labels are sparse or noisy. The included models produce an anomaly score per claim. A downstream threshold converts the score into a review decision.

Recommended operational pattern:

1. Train models on a stable baseline period.
2. Score new claims daily.
3. Select a threshold based on investigation capacity (top N) and drift monitoring.
4. Feed confirmed outcomes back into supervised models when labels mature.

## Security and compliance

This repository is a reference implementation. For production usage:

- Perform HIPAA and security assessments.
- Tokenize or hash identifiers.
- Apply row level and column level access controls.
- Log only non PHI metadata.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
```

## License

MIT
