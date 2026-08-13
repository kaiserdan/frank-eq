"""Development-only Shared Predictive Quotient (SPQ0) census."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_spq0_config(path: str | Path) -> Any:
    from .config import load_spq0_config as _load

    return _load(path)


def run_spq0_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .workflow import run_spq0_audit as _run

    return _run(*args, **kwargs)


__all__ = ["load_spq0_config", "run_spq0_audit"]
