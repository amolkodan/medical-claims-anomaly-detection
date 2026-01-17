from __future__ import annotations

from typing import Any, Dict

from sklearn.svm import OneClassSVM


def build_one_class_svm(params: Dict[str, Any]) -> OneClassSVM:
    return OneClassSVM(**params)
