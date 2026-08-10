"""Fail-open W&B telemetry for the real Stage-A workflow.

Telemetry is a convenience layer: it must never change scientific outcomes or
fail the workflow. Every call is guarded, so a missing package, missing
credentials, or a network outage degrades to a bounded stderr note and a
failure counter instead of an exception.

Credentials are read from the environment (``WANDB_API_KEY``); the project
identity comes from the frozen configuration file.
"""

from __future__ import annotations

import os
from typing import Any

from frank_eq.real_config import WandBLoggingConfig

_WANDB_FORWARDED_ENV = (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "WANDB_DIR",
    "WANDB_BASE_URL",
)
_OFFLINE_MODES = {"offline", "disabled", "dryrun"}
_MAX_ERROR_NOTES = 10


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested scalar payloads into dotted keys accepted by ``wandb.log``.

    Dictionaries recurse; int, float, bool, and str values are kept; ``None``
    and container values (lists, tuples, sets) are dropped because W&B cannot
    serialize them into a scalar metric row.
    """

    flat: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        elif value is None or isinstance(value, (list, tuple, set)):
            continue
        elif isinstance(value, (int, float, bool, str)):
            flat[name] = value
    return flat


class WandbTelemetry:
    """Lazily initialized W&B sink with a no-op degraded mode."""

    def __init__(
        self,
        config: WandBLoggingConfig,
        *,
        run_name: str,
        job: dict[str, Any] | None = None,
    ):
        self._config = config
        self._run_name = run_name
        self._job = job or {}
        self._run: Any | None = None
        self._attempted = False
        self._failure_count = 0
        self._reason: str | None = None

    @property
    def enabled(self) -> bool:
        self._ensure_run()
        return self._run is not None

    @property
    def failures(self) -> int:
        return self._failure_count

    @property
    def reason(self) -> str | None:
        return self._reason

    def _note(self, message: str) -> None:
        if self._failure_count < _MAX_ERROR_NOTES:
            print(f"frank-eq telemetry: {message}", file=os.sys.stderr)

    def _ensure_run(self) -> None:
        if self._attempted:
            return
        self._attempted = True
        try:
            if not self._config.enabled:
                self._reason = "disabled by logging.wandb.enabled=false"
                return
            mode = os.environ.get("WANDB_MODE")
            if self._config.offline and mode is None:
                os.environ["WANDB_MODE"] = "offline"
                mode = "offline"
            if not os.environ.get("WANDB_API_KEY") and mode not in _OFFLINE_MODES:
                self._reason = "missing WANDB_API_KEY (and WANDB_MODE is not offline)"
                return
            try:
                import wandb  # type: ignore[import-not-found]
            except ImportError:
                self._reason = "wandb package is not installed"
                return
            init_kwargs: dict[str, Any] = {
                "project": self._config.project,
                "name": self._run_name,
                "reinit": True,
                "config": {
                    "run_name": self._run_name,
                    "cluster": self._job.get("cluster"),
                    "slurm_job_id": self._job.get("slurm_job_id"),
                    "source_sha256": self._job.get("source_sha256"),
                    "git_commit": self._job.get("git_commit"),
                },
            }
            if self._config.entity:
                init_kwargs["entity"] = self._config.entity
            if self._config.tags:
                init_kwargs["tags"] = list(self._config.tags)
            self._run = wandb.init(**init_kwargs)
        except Exception as error:  # pragma: no cover - defensive boundary
            self._run = None
            self._failure_count += 1
            self._reason = f"wandb.init failed: {type(error).__name__}: {error}"
            self._note(self._reason)

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        """Log a nested payload; never raises into the workflow."""

        self._ensure_run()
        if self._run is None:
            return
        try:
            self._run.log(_flatten(payload), step=step)
        except Exception as error:  # pragma: no cover - defensive boundary
            self._failure_count += 1
            self._note(f"wandb.log failed: {type(error).__name__}: {error}")

    def finish(self) -> None:
        self._ensure_run()
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as error:  # pragma: no cover - defensive boundary
            self._failure_count += 1
            self._note(f"wandb.finish failed: {type(error).__name__}: {error}")
        finally:
            self._run = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "failures": self._failure_count,
            "reason": self._reason,
        }
