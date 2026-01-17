# Architecture

## Key objectives

- Detect outlier claim patterns with minimal label dependency
- Support repeatable feature engineering that aligns with fraud investigation workflows
- Enable operational thresholding based on review capacity

## Components

1. Data ingestion
   - CSV reader for demo
   - Extendable to Parquet, warehouse queries, or streaming ingestion

2. Feature engineering
   - Datetime derived features
   - Rolling aggregates by provider and member
   - One hot encoding for high-cardinality codes

3. Modeling
   - Isolation Forest (default)
   - One Class SVM

4. Scoring and decisioning
   - Produces anomaly scores
   - Thresholding generates a review queue

5. Evaluation
   - ROC AUC and Average Precision when labels exist
   - Precision, recall, and review rate at operational thresholds

## Extensibility

- Add claims line level grouping and episode based features
- Add network based features using provider member graphs
- Integrate explainability (permutation importance, SHAP for tree based models)
- Add drift monitoring and alerting
