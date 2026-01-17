from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from claims_anomaly.features.aggregations import add_entity_aggregates
from claims_anomaly.features.time_features import add_datetime_parts


@dataclass(frozen=True)
class FeatureBuildOutput:
    feature_matrix: Any
    transformer: ColumnTransformer
    feature_names: List[str]
    enriched_frame: pd.DataFrame


def build_transformer(
    numeric_columns: List[str],
    categorical_columns: List[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True)),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return transformer


def _extract_feature_names(transformer: ColumnTransformer) -> List[str]:
    feature_names: List[str] = []

    for name, trans, cols in transformer.transformers_:
        if name == "remainder" or trans is None:
            continue

        if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
            onehot = trans.named_steps["onehot"]
            if hasattr(onehot, "get_feature_names_out"):
                cat_names = list(onehot.get_feature_names_out(cols))
            else:
                cat_names = [f"{c}_encoded" for c in cols]
            feature_names.extend(cat_names)
        else:
            feature_names.extend(list(cols))

    return feature_names


def build_features(
    data_frame: pd.DataFrame,
    features_config: Dict[str, Any],
    provider_column: str = "provider_id",
    member_column: str = "member_id",
) -> FeatureBuildOutput:
    datetime_columns = features_config.get("datetime", [])
    if len(datetime_columns) != 1:
        raise ValueError("features_config.datetime must contain exactly one datetime column")

    datetime_column = datetime_columns[0]
    output_frame = add_datetime_parts(data_frame, datetime_column=datetime_column)

    numeric_columns: List[str] = list(features_config.get("numeric", []))
    categorical_columns: List[str] = list(features_config.get("categorical", []))

    agg_cfg = features_config.get("aggregations", {})
    output_frame = add_entity_aggregates(
        output_frame,
        provider_column=provider_column,
        member_column=member_column,
        datetime_column=datetime_column,
        numeric_value_columns=numeric_columns,
        provider_window_days=int(agg_cfg.get("provider_window_days", 30)),
        member_window_days=int(agg_cfg.get("member_window_days", 30)),
        include_provider=bool(agg_cfg.get("include_provider_aggregates", True)),
        include_member=bool(agg_cfg.get("include_member_aggregates", True)),
    )

    for col in output_frame.columns:
        if col.startswith("provider_") or col.startswith("member_") or col.startswith(f"{datetime_column}_"):
            if pd.api.types.is_numeric_dtype(output_frame[col]):
                if col not in numeric_columns:
                    numeric_columns.append(col)

    transformer = build_transformer(numeric_columns=numeric_columns, categorical_columns=categorical_columns)
    feature_matrix = transformer.fit_transform(output_frame)
    feature_names = _extract_feature_names(transformer)

    if hasattr(feature_matrix, "data"):
        if np.isnan(feature_matrix.data).any():
            feature_matrix.data = np.nan_to_num(feature_matrix.data, nan=0.0)

    return FeatureBuildOutput(
        feature_matrix=feature_matrix,
        transformer=transformer,
        feature_names=feature_names,
        enriched_frame=output_frame,
    )


def transform_features(
    data_frame: pd.DataFrame,
    features_config: Dict[str, Any],
    transformer: ColumnTransformer,
    provider_column: str = "provider_id",
    member_column: str = "member_id",
) -> Tuple[Any, pd.DataFrame]:
    datetime_columns = features_config.get("datetime", [])
    datetime_column = datetime_columns[0]

    output_frame = add_datetime_parts(data_frame, datetime_column=datetime_column)

    numeric_columns: List[str] = list(features_config.get("numeric", []))
    agg_cfg = features_config.get("aggregations", {})
    output_frame = add_entity_aggregates(
        output_frame,
        provider_column=provider_column,
        member_column=member_column,
        datetime_column=datetime_column,
        numeric_value_columns=numeric_columns,
        provider_window_days=int(agg_cfg.get("provider_window_days", 30)),
        member_window_days=int(agg_cfg.get("member_window_days", 30)),
        include_provider=bool(agg_cfg.get("include_provider_aggregates", True)),
        include_member=bool(agg_cfg.get("include_member_aggregates", True)),
    )

    feature_matrix = transformer.transform(output_frame)
    return feature_matrix, output_frame
