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


def recover_rate_compute_audit(
    config: RateComputeRunConfig,
    *,
    config_path: str | Path,
    source_run: str | Path,
    recovery_manifest_path: str | Path,
    recovery_manifest_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Import the artifact-only RC0 recovery workflow only when executed."""

    from .workflow import recover_rate_compute_audit as _recover

    return _recover(
        config,
        config_path=config_path,
        source_run=source_run,
        recovery_manifest_path=recovery_manifest_path,
        recovery_manifest_sha256=recovery_manifest_sha256,
        output_dir=output_dir,
    )


__all__ = [
    "RateComputeRunConfig",
    "load_rate_compute_config",
    "recover_rate_compute_audit",
    "run_rate_compute_audit",
]
