from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import IsolationForest


def build_isolation_forest(params: Dict[str, Any]) -> IsolationForest:
    return IsolationForest(**params)
