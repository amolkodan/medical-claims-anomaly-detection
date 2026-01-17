from __future__ import annotations

from typing import Any, Dict

from claims_anomaly.models.isolation_forest import build_isolation_forest
from claims_anomaly.models.one_class_svm import build_one_class_svm


def build_model(model_type: str, params: Dict[str, Any]) -> Any:
    model_type_norm = model_type.strip().lower()
    if model_type_norm == "isolation_forest":
        return build_isolation_forest(params)
    if model_type_norm in {"one_class_svm", "ocsvm"}:
        return build_one_class_svm(params)
    raise ValueError(f"Unsupported model type: {model_type}")
