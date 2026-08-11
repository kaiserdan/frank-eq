"""Development-only rate--compute and public-basis audit workflow."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shutil
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
RATE_COMPUTE_ACCESS_CONTRACT = {
    "state_precedes_operation_reveal": True,
    "literal_kv_reuse": True,
    "exact_replay_fallback": False,
    "test_worlds_available": False,
    "held_sender_loaded": False,
    "receiver_tensors_available": False,
    "claim_bearing_role": False,
}
RATE_COMPUTE_RECOVERY_INPUTS = (
    "config.yaml",
    "run_manifest.json",
    "workflow_status.json",
    "development_splits.json",
    "panels/n4.json",
    "panels/n6.json",
    "models.json",
    "records_raw.jsonl",
    "calibration.json",
    "records_calibrated.jsonl",
)
RATE_COMPUTE_RECOVERY_COPIES = (
    "config.yaml",
    "development_splits.json",
    "panels/n4.json",
    "panels/n6.json",
    "models.json",
    "records_raw.jsonl",
    "calibration.json",
    "records_calibrated.jsonl",
)


def _timestamp() -> str:
    # ``datetime.UTC`` is unavailable in the Python 3.10 Olivia runtime.
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
        "git_dirty": os.environ.get("FRANK_EQ_GIT_DIRTY"),
        "runtime_image": os.environ.get("FRANK_EQ_RUNTIME_IMAGE"),
        "runtime_image_sha256": os.environ.get("FRANK_EQ_IMAGE_SHA256"),
        "hf_home": os.environ.get("HF_HOME"),
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


def build_rate_compute_panels(config: RateComputeRunConfig) -> dict[int, RealPanel]:
    """Build every frozen RC0 complexity panel from the registered config."""

    return {
        n_entities: _build_panel(config, n_entities)
        for n_entities in config.panel.entity_counts
    }


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL in {path.name} at line {line_number}"
                ) from error
    return rows


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
    config_snapshot = root / "config.yaml"
    shutil.copyfile(config_file, config_snapshot)
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
        "config_snapshot": config_snapshot.name,
        "config_sha256": sha256_file(config_file),
        "stages": list(selected),
        "output_dir": str(root),
        "environment": _environment(),
        "access_contract": RATE_COMPUTE_ACCESS_CONTRACT,
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
        panels = build_rate_compute_panels(config)
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
        artifact_paths = [
            "config.yaml",
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
            "run_summary.json",
            *[str(Path(path).relative_to(root)) for path in panel_paths.values()],
        ]
        artifact_manifest = _artifact_manifest(root, artifact_paths)
        atomic_write_json(root / "artifact_manifest.json", artifact_manifest)
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


def _validate_recovery_inputs(
    config: RateComputeRunConfig,
    *,
    config_path: Path,
    source_run: Path,
    recovery_manifest_path: Path,
    recovery_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fail closed before reusing a completed RC0 capture."""

    if sha256_file(recovery_manifest_path) != recovery_manifest_sha256:
        raise ValueError("recovery input manifest SHA-256 mismatch")
    recovery_input = json.loads(recovery_manifest_path.read_text())
    if recovery_input.get("schema") != "frank_eq_rate_compute_recovery_input_v1":
        raise ValueError("unsupported rate--compute recovery input schema")
    if recovery_input.get("source_remote_run_root") != str(source_run):
        raise ValueError("recovery source path differs from the frozen input manifest")
    registered = recovery_input.get("files")
    if not isinstance(registered, dict) or set(registered) != set(
        RATE_COMPUTE_RECOVERY_INPUTS
    ):
        raise ValueError("recovery input manifest does not cover the exact reusable files")
    for relative, expected_hash in registered.items():
        path = source_run / relative
        if not path.is_file():
            raise FileNotFoundError(f"recovery input is missing {relative}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"recovery input hash mismatch for {relative}")

    forbidden = (
        "compiled_predictions.jsonl",
        "direct_protocol_selection.json",
        "metrics.json",
        "decision.json",
        "artifact_manifest.json",
        "run_summary.json",
    )
    if any((source_run / relative).exists() for relative in forbidden):
        raise ValueError("recovery source already contains post-calibration outcomes")

    source_manifest = json.loads((source_run / "run_manifest.json").read_text())
    source_status = json.loads((source_run / "workflow_status.json").read_text())
    if (
        source_manifest.get("development_only") is not True
        or source_manifest.get("stages") != ["audit"]
        or source_manifest.get("access_contract") != RATE_COMPUTE_ACCESS_CONTRACT
    ):
        raise ValueError("recovery source does not satisfy the frozen RC0 access contract")
    if (
        source_status.get("state") != "failed"
        or source_status.get("current_stage") != "audit"
        or source_status.get("completed_stages") != []
        or not source_status.get("failure")
    ):
        raise ValueError("recovery source is not a failed pre-decision audit")

    source_config_hash = sha256_file(source_run / "config.yaml")
    if (
        source_config_hash != sha256_file(config_path)
        or source_config_hash != source_manifest.get("config_sha256")
        or source_config_hash != recovery_input.get("source_config_sha256")
    ):
        raise ValueError("recovery config differs from the original frozen audit")

    models = json.loads((source_run / "models.json").read_text())
    expected_models = [(item.model_id, item.revision) for item in config.models]
    observed_models = [
        (item.get("model_id"), item.get("revision_observed")) for item in models
    ]
    if observed_models != expected_models:
        raise ValueError("recovery model roster or revision differs from the frozen audit")
    expected_records = 0
    for item in models:
        records = int(item.get("records", -1))
        branch = item.get("branch_execution", {})
        if (
            records < 1
            or branch.get("mode") != "kv_reuse"
            or branch.get("kv_cloned_response_branches") != records
            or branch.get("exact_prefix_continuity_checks") != records
            or branch.get("exact_replay_response_branches") != 0
            or branch.get("allow_exact_replay_fallback") is not False
            or branch.get("configured_branch_batch_size")
            != config.capture.branch_batch_size
            or branch.get("max_observed_batch_size")
            != config.capture.branch_batch_size
            or branch.get("exclusive_cache_batching") is not True
        ):
            raise ValueError("recovery model metadata violates exclusive cloned-KV capture")
        expected_records += records

    calibrated_records = _read_jsonl(source_run / "records_calibrated.jsonl")
    raw_count = sum(1 for _ in (source_run / "records_raw.jsonl").open(encoding="utf-8"))
    if raw_count != expected_records or len(calibrated_records) != expected_records:
        raise ValueError("recovery record counts differ from captured branch accounting")
    for row in calibrated_records:
        if "calibrated_probability" not in row or "prior_probability" not in row:
            raise ValueError("recovery calibrated records are incomplete")
    return recovery_input, source_manifest, source_status


def recover_rate_compute_audit(
    config: RateComputeRunConfig,
    *,
    config_path: str | Path,
    source_run: str | Path,
    recovery_manifest_path: str | Path,
    recovery_manifest_sha256: str,
    output_dir: str | Path,
    stages: str | list[str] | tuple[str, ...] = RATE_COMPUTE_ALLOWED_STAGES,
) -> dict[str, Any]:
    """Finish deterministic RC0 scoring from a hash-frozen completed capture."""

    selected = parse_rate_compute_stages(stages)
    source = Path(source_run)
    root = Path(output_dir)
    config_file = Path(config_path)
    recovery_file = Path(recovery_manifest_path)
    if source.resolve() == root.resolve():
        raise ValueError("recovery output must not overwrite the failed source run")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("recovery output directory must be empty")

    recovery_input, source_manifest, source_status = _validate_recovery_inputs(
        config,
        config_path=config_file,
        source_run=source,
        recovery_manifest_path=recovery_file,
        recovery_manifest_sha256=recovery_manifest_sha256,
    )
    for relative in RATE_COMPUTE_RECOVERY_COPIES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)

    provenance = {
        "schema": "frank_eq_rate_compute_recovery_provenance_v1",
        "recovery_input_sha256": recovery_manifest_sha256,
        "source_cluster": recovery_input.get("source_cluster"),
        "source_job_name": recovery_input.get("source_job_name"),
        "source_slurm_job_id": recovery_input.get("source_slurm_job_id"),
        "source_remote_run_root": str(source),
        "source_archive_sha256": recovery_input.get("source_archive_sha256"),
        "source_config_sha256": recovery_input.get("source_config_sha256"),
        "source_files": recovery_input["files"],
        "source_failure": source_status["failure"],
        "source_workflow_state": source_status["state"],
        "source_run_manifest_schema": source_manifest.get("schema"),
        "capture_reused": True,
        "capture_executed_in_recovery": False,
        "calibration_reused": True,
        "post_calibration_outcomes_preexisting": False,
    }
    provenance_path = root / "recovery_provenance.json"
    atomic_write_json(provenance_path, provenance)

    run_manifest = {
        "schema": "frank_eq_rate_compute_run_manifest_v1",
        "run_name": config.run_name,
        "protocol_role": "rate_compute_development",
        "development_only": True,
        "created_at": _timestamp(),
        "config_path": str(config_file),
        "config_snapshot": "config.yaml",
        "config_sha256": sha256_file(root / "config.yaml"),
        "stages": list(selected),
        "output_dir": str(root),
        "environment": _environment(),
        "access_contract": RATE_COMPUTE_ACCESS_CONTRACT,
        "recovery": {
            "artifact_only": True,
            "model_capture_executed": False,
            "source_job_name": recovery_input.get("source_job_name"),
            "source_slurm_job_id": recovery_input.get("source_slurm_job_id"),
            "source_archive_sha256": recovery_input.get("source_archive_sha256"),
            "recovery_input_sha256": recovery_manifest_sha256,
            "recovery_provenance": provenance_path.name,
            "recovery_provenance_sha256": sha256_file(provenance_path),
        },
    }
    atomic_write_json(root / "run_manifest.json", run_manifest)
    status: dict[str, Any] = {
        "schema": "frank_eq_rate_compute_status_v1",
        "state": "running",
        "current_stage": "audit",
        "completed_stages": [],
        "started_at": _timestamp(),
        "failure": None,
        "recovery": True,
    }
    atomic_write_json(root / "workflow_status.json", status)
    started = time.time()

    try:
        panels = {
            n_entities: RealPanel.from_dict(
                json.loads((root / f"panels/n{n_entities}.json").read_text())
            )
            for n_entities in config.panel.entity_counts
        }
        records = _read_jsonl(root / "records_calibrated.jsonl")
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
        atomic_write_json(root / "workflow_status.json", status)
        summary = {
            "schema": "frank_eq_rate_compute_run_v1",
            "status": "completed",
            "workflow_integrity_passed": True,
            "development_only": True,
            "artifact_only_recovery": True,
            "model_capture_executed": False,
            "root": str(root),
            "records": len(records),
            "compiled_predictions": len(compiled),
            "decision": decision,
            "artifact_manifest": str(root / "artifact_manifest.json"),
            "authorizes_scientific_claim": False,
            "telemetry": {
                "enabled": False,
                "reason": "artifact_only_recovery",
            },
        }
        atomic_write_json(root / "run_summary.json", summary)
        artifact_paths = [
            "config.yaml",
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
            "run_summary.json",
            "recovery_provenance.json",
            *[f"panels/n{n_entities}.json" for n_entities in config.panel.entity_counts],
        ]
        atomic_write_json(
            root / "artifact_manifest.json",
            _artifact_manifest(root, artifact_paths),
        )
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
        atomic_write_json(root / "workflow_status.json", status)
        raise
