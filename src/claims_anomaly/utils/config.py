from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle) or {}


@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @staticmethod
    def from_cwd() -> "Paths":
        current = Path.cwd().resolve()
        for parent in [current, *current.parents]:
            if (parent / "pyproject.toml").exists():
                return Paths(repo_root=parent)
        return Paths(repo_root=current)
