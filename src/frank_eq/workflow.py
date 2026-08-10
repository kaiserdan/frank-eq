"""Auditable end-to-end workflows for the real-checkpoint Stage-A canary."""

from __future__ import annotations

import os
import platform
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from frank_eq.data.real import RealBundle, build_real_cache, validate_real_cache
from frank_eq.evaluation import Stage0Evaluator
from frank_eq.real_config import RealRunConfig
from frank_eq.training import Stage0Trainer
from frank_eq.utils import atomic_write_json, sha256_file

REAL_STAGE_ORDER = ("cache", "validate", "train", "eval")


def _timestamp() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


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
        raise ValueError("real Stage-A stages must follow cache,validate,train,eval order")
    return stages


def run_real_stagea(
    config: RealRunConfig,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    stages: str | list[str] | tuple[str, ...] = REAL_STAGE_ORDER,
) -> dict[str, Any]:
    """Run selected stages and preserve scientific failure as a valid job outcome."""

    selected = parse_real_stages(stages)
    root = Path(output_dir)
    cache_dir = root / "cache"
    train_dir = root / "train"
    eval_dir = root / "eval"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "run_manifest.json"
    status_path = root / "workflow_status.json"
    config_file = Path(config_path)
    manifest = {
        "schema": "frank_eq_real_stagea_manifest_v1",
        "run_name": config.run_name,
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
        },
    }
    atomic_write_json(manifest_path, manifest)
    status: dict[str, Any] = {
        "schema": "frank_eq_real_stagea_status_v1",
        "state": "running",
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
            elif stage == "validate":
                stage_outputs[stage] = validate_real_cache(cache_dir)
            elif stage == "train":
                validation = validate_real_cache(cache_dir)
                if not validation.get("authorizes_training", False):
                    raise RuntimeError("cache validator did not authorize training")
                bundle = RealBundle.load(cache_dir)
                stage0_config = config.to_stage0_config(bundle.model_hidden_dims)
                atomic_write_json(root / "resolved_stage0_config.json", stage0_config.as_dict())
                trainer = Stage0Trainer(stage0_config, bundle, train_dir)
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
                )
                metrics, decision = evaluator.evaluate()
                stage_outputs[stage] = {"metrics": metrics, "decision": decision}
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
            "workflow_integrity_passed": True,
            "manifest": str(manifest_path),
            "workflow_status": str(status_path),
            "stages": stage_outputs,
            "scientific_decision": status.get("scientific_decision"),
            "authorizes_scientific_claim": False,
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
