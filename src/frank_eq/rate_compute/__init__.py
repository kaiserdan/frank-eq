"""Rate--compute operational-basis development audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import RateComputeRunConfig, load_rate_compute_config


def run_rate_compute_audit(
    config: RateComputeRunConfig,
    *,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Import the heavyweight frozen-model workflow only when it is executed."""

    from .workflow import run_rate_compute_audit as _run

    return _run(config, config_path=config_path, output_dir=output_dir)


__all__ = ["RateComputeRunConfig", "load_rate_compute_config", "run_rate_compute_audit"]
