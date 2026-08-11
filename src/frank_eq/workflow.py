"""Auditable end-to-end workflows for the real-checkpoint Stage-A canary."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from frank_eq.data.real import RealBundle, build_real_cache, validate_real_cache
from frank_eq.diagnostics import diagnose_real_cache
from frank_eq.evaluation import Stage0Evaluator
from frank_eq.real_config import RealRunConfig
from frank_eq.telemetry import WandbTelemetry
from frank_eq.training import Stage0Trainer
from frank_eq.utils import atomic_write_json, sha256_file

REAL_STAGE_ORDER = ("cache", "validate", "diagnose", "train", "eval")
STAGEQ_ALLOWED_STAGES = frozenset({"cache", "validate"})


def _timestamp() -> str:
    # ``datetime.UTC`` is unavailable in the Python 3.10 cluster runtimes.
    return dt.datetime.now(dt.timezone.utc).isoformat()  # noqa: UP017


def _environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cluster": os.environ.get("FRANK_EQ_CLUSTER"),
        "source_sha256": os.environ.get("FRANK_EQ_SOURCE_SHA256"),
        "git_commit": os.environ.get("FRANK_EQ_GIT_COMMIT"),
    }
    if torch.cuda.is_available():
        payload["accelerators"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return payload


def parse_real_stages(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        stages = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        stages = tuple(str(item) for item in value)
    if not stages:
        raise ValueError("at least one real Stage-A workflow stage is required")
    unknown = set(stages) - set(REAL_STAGE_ORDER)
    if unknown:
        raise ValueError(f"unknown real Stage-A stages: {sorted(unknown)}")
    indices = [REAL_STAGE_ORDER.index(stage) for stage in stages]
    if indices != sorted(indices):
        raise ValueError(
            "real Stage-A stages must follow cache,validate,diagnose,train,eval order"
        )
    return stages


def infer_protocol_role(config: RealRunConfig, config_path: str | Path) -> str:
    """Infer the fail-closed execution role from registered identity and path.

    Stage-Q configs are intentionally development-only and live under
    ``configs/stageq``. The run-name check preserves the restriction if a
    registered config is copied to a cluster job directory before execution.
    """

    path_parts = {part.lower() for part in Path(config_path).parts}
    run_name = config.run_name.lower()
    if "stageq" in path_parts or run_name.startswith("frank-eq-stageq-"):
        return "stageq"
    return "stagea"


def validate_real_stage_role(
    config: RealRunConfig,
    config_path: str | Path,
    stages: str | list[str] | tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    """Validate stage order and forbid promotional work for Stage Q."""

    selected = parse_real_stages(stages)
    role = infer_protocol_role(config, config_path)
    if role == "stageq":
        forbidden = sorted(set(selected) - STAGEQ_ALLOWED_STAGES)
        if forbidden:
            raise ValueError(
                "Stage-Q development configs permit only cache,validate; "
                f"forbidden stages requested: {forbidden}"
            )
    return role, selected


def _cache_telemetry(cache_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Build a scalar telemetry payload from the cache build summary."""

    payload = {key: value for key, value in summary.items() if key not in {"cache_dir"}}
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        return payload
    try:
        metadata = json.loads(metadata_path.read_text())
        models = metadata.get("extraction", {}).get("models", [])
        payload["branch_kv_reuse"] = sum(
            int(model.get("branch_mode_counts", {}).get("kv_reuse", 0)) for model in models
        )
        payload["branch_exact_prefix_replay"] = sum(
            int(model.get("branch_mode_counts", {}).get("exact_prefix_replay", 0))
            for model in models
        )
    except Exception:
        pass
    return payload


def run_real_stagea(
    config: RealRunConfig,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    stages: str | list[str] | tuple[str, ...] = REAL_STAGE_ORDER,
) -> dict[str, Any]:
    """Run selected stages and preserve scientific failure as a valid job outcome."""

    protocol_role, selected = validate_real_stage_role(config, config_path, stages)
    root = Path(output_dir)
    cache_dir = root / "cache"
    diagnostic_dir = root / "diagnostics"
    train_dir = root / "train"
    eval_dir = root / "eval"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "run_manifest.json"
    status_path = root / "workflow_status.json"
    config_file = Path(config_path)
    telemetry = WandbTelemetry(
        config.logging.wandb,
        run_name=config.run_name,
        job=_environment(),
    )
    manifest = {
        "schema": "frank_eq_real_stagea_manifest_v1",
        "run_name": config.run_name,
        "protocol_role": protocol_role,
        "development_only": protocol_role == "stageq",
        "created_at": _timestamp(),
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "stages": list(selected),
        "output_dir": str(root),
        "environment": _environment(),
        "access_contract": {
            "capture_precedes_operation": True,
            "receiver_tensors_available": False,
            "future_labels_available_at_capture": False,
            "held_sender_updates_public_decoder": False,
            "claim_bearing_test_authorized": protocol_role != "stageq",
        },
    }
    atomic_write_json(manifest_path, manifest)
    telemetry.log(
        {
            "run": {
                "run_name": manifest["run_name"],
                "protocol_role": protocol_role,
                "config_sha256": manifest["config_sha256"],
                "stages": ",".join(selected),
                "python": manifest["environment"].get("python"),
                "torch": manifest["environment"].get("torch"),
                "cuda_available": manifest["environment"].get("cuda_available"),
            }
        }
    )
    status: dict[str, Any] = {
        "schema": "frank_eq_real_stagea_status_v1",
        "state": "running",
        "protocol_role": protocol_role,
        "development_only": protocol_role == "stageq",
        "started_at": _timestamp(),
        "completed_stages": [],
        "current_stage": None,
        "failure": None,
    }
    atomic_write_json(status_path, status)
    stage_outputs: dict[str, Any] = {}
    start = time.time()

    try:
        for stage in selected:
            status["current_stage"] = stage
            status["stage_started_at"] = _timestamp()
            atomic_write_json(status_path, status)
            if stage == "cache":
                stage_outputs[stage] = build_real_cache(config, cache_dir)
                telemetry.log({"cache": _cache_telemetry(cache_dir, stage_outputs[stage])})
            elif stage == "validate":
                stage_outputs[stage] = validate_real_cache(cache_dir)
                telemetry.log(
                    {
                        "cache_validation": {
                            key: value
                            for key, value in stage_outputs[stage].items()
                            if isinstance(value, (int, float, bool, str))
                        }
                    }
                )
            elif stage == "diagnose":
                validate_real_cache(cache_dir)
                stage_outputs[stage] = diagnose_real_cache(cache_dir, diagnostic_dir)
                telemetry.log(
                    {
                        "diagnostic": {
                            "recommendation": stage_outputs[stage]["recommendation"]["code"],
                            "promotional": False,
                        }
                    }
                )
            elif stage == "train":
                validation = validate_real_cache(cache_dir)
                if not validation.get("authorizes_training", False):
                    raise RuntimeError("cache validator did not authorize training")
                bundle = RealBundle.load(cache_dir)
                stage0_config = config.to_stage0_config(bundle.model_hidden_dims)
                atomic_write_json(root / "resolved_stage0_config.json", stage0_config.as_dict())
                trainer = Stage0Trainer(stage0_config, bundle, train_dir, telemetry=telemetry)
                stage_outputs[stage] = trainer.train()
            elif stage == "eval":
                bundle = RealBundle.load(cache_dir)
                stage0_config = config.to_stage0_config(bundle.model_hidden_dims)
                checkpoint = train_dir / "final.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"training checkpoint not found: {checkpoint}")
                evaluator = Stage0Evaluator(
                    stage0_config,
                    bundle,
                    checkpoint_path=checkpoint,
                    output_dir=eval_dir,
                    telemetry=telemetry,
                )
                metrics, decision = evaluator.evaluate()
                stage_outputs[stage] = {"metrics": metrics, "decision": decision}
                telemetry.log(
                    {
                        "workflow": {
                            "scientific_decision": decision.get("decision"),
                            "decision_status": decision.get("status"),
                        }
                    }
                )
            status["completed_stages"].append(stage)
            status["stage_completed_at"] = _timestamp()
            atomic_write_json(status_path, status)

        status.update(
            {
                "state": "completed",
                "current_stage": None,
                "completed_at": _timestamp(),
                "elapsed_seconds": time.time() - start,
                "scientific_decision": (
                    stage_outputs.get("eval", {}).get("decision")
                    if isinstance(stage_outputs.get("eval"), dict)
                    else None
                ),
            }
        )
        atomic_write_json(status_path, status)
        summary = {
            "schema": "frank_eq_real_stagea_run_v1",
            "status": "completed",
            "protocol_role": protocol_role,
            "development_only": protocol_role == "stageq",
            "workflow_integrity_passed": True,
            "manifest": str(manifest_path),
            "workflow_status": str(status_path),
            "stages": stage_outputs,
            "scientific_decision": status.get("scientific_decision"),
            "authorizes_scientific_claim": False,
            "telemetry": telemetry.status(),
        }
        atomic_write_json(root / "run_summary.json", summary)
        return summary
    except Exception as error:
        status.update(
            {
                "state": "failed",
                "failed_at": _timestamp(),
                "elapsed_seconds": time.time() - start,
                "failure": {
                    "stage": status.get("current_stage"),
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        atomic_write_json(status_path, status)
        raise
    finally:
        telemetry.finish()
