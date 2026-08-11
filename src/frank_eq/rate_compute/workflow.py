"""Development-only rate--compute and public-basis audit workflow."""

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

from frank_eq.data.real_panel import RealPanel, generate_real_panel
from frank_eq.real_config import RealPanelConfig
from frank_eq.telemetry import WandbTelemetry
from frank_eq.utils import atomic_write_json, sha256_file

from .config import RateComputeRunConfig
from .decision import evaluate_rate_compute
from .records import calibrate_records, capture_records

RATE_COMPUTE_ALLOWED_STAGES = ("audit",)


def _timestamp() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


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


def parse_rate_compute_stages(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        stages = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        stages = tuple(str(item) for item in value)
    if stages != RATE_COMPUTE_ALLOWED_STAGES:
        raise ValueError("rate--compute RC0 permits exactly one stage: audit")
    return stages


def _build_panel(config: RateComputeRunConfig, n_entities: int) -> RealPanel:
    panel_config = RealPanelConfig(
        n_worlds=config.panel.worlds_per_complexity,
        n_entities=n_entities,
        n_operations=config.panel.n_target_operations,
        n_renderers=config.panel.n_renderers,
        train_fraction=config.panel.train_fraction,
        validation_fraction=min(
            0.20, max(0.05, (1.0 - config.panel.train_fraction) / 2.0)
        ),
        operation_holdout_fraction=0.25,
        oracle_smoothing=config.panel.oracle_smoothing,
        min_operation_positive_fraction=config.panel.min_operation_positive_fraction,
        max_operation_positive_fraction=config.panel.max_operation_positive_fraction,
        max_generation_attempts=config.panel.max_generation_attempts,
        seed=config.panel.seed + 1009 * n_entities,
    )
    return generate_real_panel(panel_config)


def _development_split(
    config: RateComputeRunConfig,
    n_entities: int,
) -> dict[int, str]:
    """Create a deterministic 70/30 train/validation split with no test role."""

    rng = np.random.default_rng(config.panel.seed + 17 + 1009 * n_entities)
    world_ids = np.arange(config.panel.worlds_per_complexity, dtype=np.int64)
    rng.shuffle(world_ids)
    train_count = int(round(config.panel.train_fraction * len(world_ids)))
    if train_count < 2 or len(world_ids) - train_count < 2:
        raise ValueError("rate--compute split needs at least two train and validation worlds")
    train = set(int(value) for value in world_ids[:train_count])
    return {
        int(world_id): ("train" if int(world_id) in train else "validation")
        for world_id in range(config.panel.worlds_per_complexity)
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _write_panels(root: Path, panels: dict[int, RealPanel]) -> dict[str, str]:
    result: dict[str, str] = {}
    for n_entities, panel in sorted(panels.items()):
        path = root / "panels" / f"n{n_entities}.json"
        atomic_write_json(path, panel.to_dict())
        result[str(n_entities)] = str(path)
    return result


def _artifact_manifest(root: Path, relative_paths: list[str]) -> dict[str, Any]:
    files = {
        relative: sha256_file(root / relative)
        for relative in relative_paths
        if (root / relative).is_file()
    }
    return {
        "schema": "frank_eq_rate_compute_artifact_manifest_v1",
        "files": files,
    }


def run_rate_compute_audit(
    config: RateComputeRunConfig,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    stages: str | list[str] | tuple[str, ...] = RATE_COMPUTE_ALLOWED_STAGES,
) -> dict[str, Any]:
    """Run the frozen RC0 development audit and preserve negative decisions."""

    selected = parse_rate_compute_stages(stages)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_file = Path(config_path)
    manifest_path = root / "run_manifest.json"
    status_path = root / "workflow_status.json"
    telemetry = WandbTelemetry(
        config.logging.wandb,
        run_name=config.run_name,
        job=_environment(),
    )
    run_manifest = {
        "schema": "frank_eq_rate_compute_run_manifest_v1",
        "run_name": config.run_name,
        "protocol_role": "rate_compute_development",
        "development_only": True,
        "created_at": _timestamp(),
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "stages": list(selected),
        "output_dir": str(root),
        "environment": _environment(),
        "access_contract": {
            "state_precedes_operation_reveal": True,
            "literal_kv_reuse": True,
            "exact_replay_fallback": False,
            "test_worlds_available": False,
            "held_sender_loaded": False,
            "receiver_tensors_available": False,
            "claim_bearing_role": False,
        },
    }
    atomic_write_json(manifest_path, run_manifest)
    status: dict[str, Any] = {
        "schema": "frank_eq_rate_compute_status_v1",
        "state": "running",
        "current_stage": "audit",
        "completed_stages": [],
        "started_at": _timestamp(),
        "failure": None,
    }
    atomic_write_json(status_path, status)
    started = time.time()

    try:
        panels = {
            n_entities: _build_panel(config, n_entities)
            for n_entities in config.panel.entity_counts
        }
        splits = {
            n_entities: _development_split(config, n_entities)
            for n_entities in config.panel.entity_counts
        }
        panel_paths = _write_panels(root, panels)
        split_payload = {
            str(n_entities): {
                "train_world_ids": sorted(
                    world_id for world_id, role in split.items() if role == "train"
                ),
                "validation_world_ids": sorted(
                    world_id for world_id, role in split.items() if role == "validation"
                ),
                "test_world_ids": [],
            }
            for n_entities, split in splits.items()
        }
        atomic_write_json(root / "development_splits.json", split_payload)
        telemetry.log(
            {
                "audit": {
                    "models": len(config.models),
                    "entity_counts": len(config.panel.entity_counts),
                    "worlds_per_complexity": config.panel.worlds_per_complexity,
                    "target_operations": config.panel.n_target_operations,
                }
            }
        )

        records, model_metadata = capture_records(config, panels, splits, telemetry)
        atomic_write_json(root / "models.json", model_metadata)
        _write_jsonl(root / "records_raw.jsonl", records)

        calibration = calibrate_records(records, config)
        atomic_write_json(root / "calibration.json", calibration)
        _write_jsonl(root / "records_calibrated.jsonl", records)

        metrics, decision, compiled, direct_selection = evaluate_rate_compute(
            records, panels, config
        )
        _write_jsonl(root / "compiled_predictions.jsonl", compiled)
        atomic_write_json(root / "direct_protocol_selection.json", direct_selection)
        atomic_write_json(root / "metrics.json", metrics)
        atomic_write_json(root / "decision.json", decision)

        status.update(
            {
                "state": "completed",
                "current_stage": None,
                "completed_stages": ["audit"],
                "completed_at": _timestamp(),
                "elapsed_seconds": time.time() - started,
                "scientific_decision": decision,
            }
        )
        atomic_write_json(status_path, status)
        artifact_paths = [
            "run_manifest.json",
            "workflow_status.json",
            "development_splits.json",
            "models.json",
            "records_raw.jsonl",
            "calibration.json",
            "records_calibrated.jsonl",
            "compiled_predictions.jsonl",
            "direct_protocol_selection.json",
            "metrics.json",
            "decision.json",
            *[str(Path(path).relative_to(root)) for path in panel_paths.values()],
        ]
        artifact_manifest = _artifact_manifest(root, artifact_paths)
        atomic_write_json(root / "artifact_manifest.json", artifact_manifest)
        telemetry.log(
            {
                "decision": {
                    "status": decision["status"],
                    "diagnosis": decision["diagnosis"],
                    "basis_passed": decision["checks"]["basis_contract"]["passed"],
                    "compiled_direct_passed": decision["checks"][
                        "compiled_over_train_selected_direct"
                    ]["passed"],
                }
            }
        )
        summary = {
            "schema": "frank_eq_rate_compute_run_v1",
            "status": "completed",
            "workflow_integrity_passed": True,
            "development_only": True,
            "root": str(root),
            "records": len(records),
            "compiled_predictions": len(compiled),
            "decision": decision,
            "artifact_manifest": str(root / "artifact_manifest.json"),
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
                "elapsed_seconds": time.time() - started,
                "failure": {
                    "stage": "audit",
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        atomic_write_json(status_path, status)
        raise
    finally:
        telemetry.finish()
