from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class ModelArtifact:
    model: Any
    transformer: Any
    feature_config: Dict[str, Any]


class AnomalyModel(Protocol):
    def fit(self, x: Any) -> "AnomalyModel":
        ...

    def score_samples(self, x: Any) -> Any:
        ...
