"""Public predictive-state development census."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_predictive_state_config(path: str | Path) -> Any:
    from .config import load_predictive_state_config as _load

    return _load(path)


def run_predictive_state_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .workflow import run_predictive_state_audit as _run

    return _run(*args, **kwargs)


__all__ = ["load_predictive_state_config", "run_predictive_state_audit"]
