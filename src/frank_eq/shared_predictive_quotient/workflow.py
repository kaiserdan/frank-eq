"""Immutable development-only SPQ0 plan and audit workflow."""

from __future__ import annotations

import datetime as dt
import os
import platform
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from frank_eq.telemetry import WandbTelemetry
from frank_eq.utils import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file

from .capture import _write_npz, capture_model, load_capture
from .checkpoints import preflight_checkpoint_contract
from .config import SPQRunConfig
from .evaluation import deterministic_prediction_digest, evaluate_all_models, gate_decision
from .panel import build_panels

SPQ0_ALLOWED_STAGES = ("audit",)
SPQ0_CONFIG_PATH = "configs/spq0/real_olivia_spq0.yaml"
SPQ0_PLAN_PATH = "configs/spq0/inspected_plan.json"
SPQ0_REGISTRATION_PATH = "configs/spq0/registration.json"
SPQ0_PROTOCOL_PATH = "docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md"
SPQ0_RUNTIME_IMAGE = "/cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif"
SPQ0_RUNTIME_IMAGE_SHA256 = "a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1"


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
        "git_dirty": os.environ.get("FRANK_EQ_GIT_DIRTY"),
        "runtime_image": os.environ.get("FRANK_EQ_RUNTIME_IMAGE"),
        "runtime_image_sha256": os.environ.get("FRANK_EQ_IMAGE_SHA256"),
    }
    if torch.cuda.is_available():
        payload["accelerators"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return payload


def parse_spq0_stages(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        stages = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        stages = tuple(str(item) for item in value)
    if stages != SPQ0_ALLOWED_STAGES:
        raise ValueError("SPQ0 permits exactly one development stage: audit")
    return stages


def _repo_root(config_path: str | Path) -> Path:
    source = Path(config_path).resolve()
    try:
        return source.parents[2]
    except IndexError as error:
        raise ValueError("SPQ0 config path is not inside a repository") from error


def build_spq0_plan(
    config: SPQRunConfig,
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    """Build the model-free deterministic plan bound to config and protocol hashes."""

    config.validate()
    source = Path(config_path)
    root = _repo_root(source)
    protocol = root / SPQ0_PROTOCOL_PATH
    if not protocol.is_file():
        raise FileNotFoundError(f"SPQ0 protocol not found: {protocol}")
    systems, basis = config.build_systems_and_basis()
    role_counts = {
        "calibration": (
            config.systems.fit_systems
            * len(config.panel.roles.calibration.lengths)
            * config.panel.roles.calibration.histories_per_system_length
        ),
        "selection": (
            config.systems.fit_systems
            * len(config.panel.roles.selection.lengths)
            * config.panel.roles.selection.histories_per_system_length
        ),
        "validation": (
            len(systems)
            * len(config.panel.roles.validation.lengths)
            * config.panel.roles.validation.histories_per_system_length
        ),
    }
    prefixes_per_model = (
        role_counts["calibration"] * len(config.panel.fit_renderers)
        + role_counts["selection"] * len(config.panel.fit_renderers)
        + role_counts["validation"] * len(config.panel.validation_renderers)
    )
    tests_per_prefix = len(basis.public_tests) + len(basis.target_tests)
    system_payload = [system.to_dict() for system in systems]
    basis_payload = basis.to_dict()
    active_revisions = {
        model.model_id: {
            "hf_id": model.hf_id,
            "family": model.family,
            "revision": model.revision,
        }
        for model in config.models
    }
    reserved = [
        {
            "model_id": model.model_id,
            "hf_id": model.hf_id,
            "family": model.family,
            "revision": model.revision,
            "access": "reserved_unopened",
        }
        for model in config.reserved_unopened_models
    ]
    payload: dict[str, Any] = {
        "schema": "frank_eq_spq0_plan_v1",
        "protocol_version": config.protocol_version,
        "config_path": SPQ0_CONFIG_PATH,
        "config_sha256": sha256_file(source),
        "protocol_path": SPQ0_PROTOCOL_PATH,
        "protocol_sha256": sha256_file(protocol),
        "development_only": True,
        "models": active_revisions,
        "active_checkpoint_revision_registry_sha256": sha256_bytes(
            canonical_json_bytes(active_revisions)
        ),
        "reserved_unopened_models": reserved,
        "reserved_checkpoint_non_access_contract_sha256": sha256_bytes(
            canonical_json_bytes(reserved)
        ),
        "systems": {
            "fit": config.systems.fit_systems,
            "validation_only": config.systems.validation_only_systems,
            "family_sha256": sha256_bytes(canonical_json_bytes(system_payload)),
            "validation_only_system_ids": [
                system.system_id for system in systems if system.role == "validation_only"
            ],
        },
        "public_basis": {
            "exact_rank": basis.exact_rank,
            "rank_grid": config.semantic_encoder.rank_grid,
            "public_tests": len(basis.public_tests),
            "target_tests": len(basis.target_tests),
            "core_condition_numbers": dict(basis.core_condition_numbers),
            "maximum_target_l1": basis.maximum_target_l1,
            "maximum_exact_executor_error": basis.maximum_exact_executor_error,
            "basis_sha256": sha256_bytes(canonical_json_bytes(basis_payload)),
        },
        "panel": {
            "roles": role_counts,
            "test_histories": 0,
            "fit_renderers": config.panel.fit_renderers,
            "validation_renderers": config.panel.validation_renderers,
            "validation_only_length": config.panel.validation_only_length,
        },
        "capture": {
            "surfaces": config.capture.surfaces,
            "normalized_depths": config.capture.normalized_depths,
            "literal_kv_reuse": True,
            "exact_replay_fallback": False,
            "prefixes_per_model": prefixes_per_model,
            "future_tests_per_prefix": tests_per_prefix,
            "post_reveal_query_branches_per_model": prefixes_per_model
            * tests_per_prefix,
            "categorical_candidates_per_query": len(
                config.probability_protocol.bins
            ),
            "primary_packet_post_capture_source_queries": 0,
            "runtime_queries_are_development_tomography": True,
        },
        "composition": {
            "target_local_reader_count": len(config.models),
            "target_readers_frozen_before_source_evaluation": True,
            "ordered_cross_family_pairs": len(config.models) * (len(config.models) - 1),
            "pair_specific_mapper_count": 0,
            "behavioral_residual_promotional": False,
        },
        "rate_compute": {
            "primary_packet_rank": basis.exact_rank,
            "primary_quantization_bits_per_coordinate": 4,
            "primary_payload_bits": 4 * basis.exact_rank,
            "amortized_future_query_counts": config.evaluation.amortized_future_query_counts,
            "primary_amortized_query_count": config.gates.amortized_query_count_for_primary_utility,
            "direct_one_query_advantage_is_not_conjunctive": True,
        },
        "runtime": {
            "cluster": "olivia",
            "profile": "full",
            "stages": ["audit"],
            "image": SPQ0_RUNTIME_IMAGE,
            "image_sha256": SPQ0_RUNTIME_IMAGE_SHA256,
            "active_checkpoint_file_hash_preflight_required": True,
            "clean_committed_source_required": True,
        },
        "access": {
            "future_test_revealed_before_capture": False,
            "test_role": False,
            "held_sender": False,
            "receiver": False,
            "reserved_checkpoint_snapshot_resolution_attempts": 0,
            "reserved_checkpoint_files_opened": 0,
            "reserved_checkpoint_model_loads": 0,
        },
    }
    payload["plan_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def write_spq0_plan(path: str | Path, plan: Mapping[str, Any]) -> None:
    expected = dict(plan)
    observed = expected.pop("plan_sha256", None)
    if observed != sha256_bytes(canonical_json_bytes(expected)):
        raise ValueError("SPQ0 plan has an invalid internal SHA-256")
    atomic_write_json(path, dict(plan))


def _artifact_manifest(root: Path, relative_paths: Iterable[str]) -> dict[str, Any]:
    return {
        "schema": "frank_eq_spq0_artifact_manifest_v1",
        "files": {
            relative: sha256_file(root / relative)
            for relative in sorted(set(relative_paths))
            if (root / relative).is_file()
        },
    }


def _write_array_groups(
    root: Path,
    directory: str,
    groups: Mapping[str, Mapping[str, np.ndarray]],
    *,
    schema: str,
) -> tuple[dict[str, Any], list[str]]:
    entries: dict[str, Any] = {}
    paths: list[str] = []
    for group in sorted(groups):
        path = root / directory / f"{group}.npz"
        digest = _write_npz(path, dict(groups[group]))
        relative = str(path.relative_to(root))
        entries[group] = {
            "path": relative,
            "sha256": digest,
            "arrays": sorted(groups[group]),
        }
        paths.append(relative)
    manifest_path = root / f"{directory}_manifest.json"
    atomic_write_json(manifest_path, {"schema": schema, "entries": entries})
    paths.append(str(manifest_path.relative_to(root)))
    return entries, paths


def run_spq0_audit(
    config: SPQRunConfig,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    stages: str | Iterable[str] = SPQ0_ALLOWED_STAGES,
    inspected_plan: Mapping[str, Any] | None = None,
    inspected_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the one authorized SPQ0 audit and independently recompute its result."""

    selected_stages = parse_spq0_stages(stages)
    config.validate()
    config_file = Path(config_path)
    root_repo = _repo_root(config_file)
    expected_plan = build_spq0_plan(config, config_path=config_file)
    if inspected_plan is None or dict(inspected_plan) != expected_plan:
        raise ValueError("committed inspected SPQ0 plan differs from the frozen implementation")
    if inspected_plan_sha256 != expected_plan["plan_sha256"]:
        raise ValueError("inspected SPQ0 plan SHA-256 differs from the frozen plan")
    environment = _environment()
    if environment.get("runtime_image_sha256") != SPQ0_RUNTIME_IMAGE_SHA256:
        raise RuntimeError("SPQ0 requires the exact registered runtime-image SHA-256")
    if environment.get("git_dirty") != "false":
        raise RuntimeError("SPQ0 execution requires a clean committed source tree")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_file, root / "config.yaml")
    shutil.copyfile(root_repo / SPQ0_PROTOCOL_PATH, root / "protocol.md")
    shutil.copyfile(root_repo / SPQ0_REGISTRATION_PATH, root / "registration.json")
    atomic_write_json(root / "dry_run_plan.json", expected_plan)
    telemetry = WandbTelemetry(config.logging.wandb, run_name=config.run_name, job=environment)
    run_manifest = {
        "schema": "frank_eq_spq0_run_manifest_v1",
        "run_name": config.run_name,
        "protocol_version": config.protocol_version,
        "development_only": True,
        "created_at": _timestamp(),
        "config_path": str(config_file),
        "config_sha256": sha256_file(root / "config.yaml"),
        "protocol_sha256": sha256_file(root / "protocol.md"),
        "registration_sha256": sha256_file(root / "registration.json"),
        "plan_sha256": expected_plan["plan_sha256"],
        "stages": list(selected_stages),
        "environment": environment,
        "access_contract": {
            "state_precedes_future_test": True,
            "literal_cloned_kv_reuse": True,
            "exact_replay_fallback": False,
            "calibration_selection_validation_only": True,
            "test_role": False,
            "held_sender": False,
            "receiver": False,
            "reserved_checkpoints_unopened": True,
        },
    }
    atomic_write_json(root / "run_manifest.json", run_manifest)
    status: dict[str, Any] = {
        "schema": "frank_eq_spq0_status_v1",
        "state": "running",
        "current_stage": "checkpoint_preflight",
        "completed_stages": [],
        "started_at": _timestamp(),
        "failure": None,
    }
    atomic_write_json(root / "workflow_status.json", status)
    started = time.time()
    try:
        checkpoint_preflight = preflight_checkpoint_contract(
            config,
            output_path=root / "checkpoint_preflight.json",
        )
        status["current_stage"] = "audit"
        status["checkpoint_preflight_completed"] = True
        atomic_write_json(root / "workflow_status.json", status)

        systems, basis = config.build_systems_and_basis()
        panels = build_panels(config, systems, basis)
        atomic_write_json(
            root / "systems.json",
            {
                "schema": "frank_eq_spq0_system_family_v1",
                "systems": [system.to_dict() for system in systems],
            },
        )
        atomic_write_json(root / "public_basis.json", basis.to_dict())
        panel_paths: list[str] = []
        for role, panel in panels.items():
            relative = f"panels/{role}.json"
            atomic_write_json(root / relative, panel.to_dict())
            panel_paths.append(relative)
        atomic_write_json(
            root / "panel_manifest.json",
            {
                "schema": "frank_eq_spq0_panel_manifest_v1",
                "roles": {
                    role: {
                        "path": f"panels/{role}.json",
                        "sha256": sha256_file(root / f"panels/{role}.json"),
                        "histories": len(panel.histories),
                        "systems": list(panel.system_ids),
                        "lengths": list(panel.lengths),
                        "renderers": list(panel.renderers),
                    }
                    for role, panel in panels.items()
                },
                "test_histories": 0,
            },
        )
        atomic_write_json(
            root / "models.json",
            {
                "schema": "frank_eq_spq0_models_v1",
                "active_founders": [
                    {
                        "model_id": model.model_id,
                        "hf_id": model.hf_id,
                        "family": model.family,
                        "revision": model.revision,
                    }
                    for model in config.models
                ],
                "reserved_unopened": checkpoint_preflight["reserved_unopened"],
            },
        )

        capture_entries: dict[str, Any] = {}
        for model in config.models:
            capture_entries[model.model_id] = capture_model(
                config,
                model,
                systems,
                basis,
                panels,
                root,
                telemetry,
            )
        atomic_write_json(
            root / "capture_manifest.json",
            {"schema": "frank_eq_spq0_capture_manifest_v1", "entries": capture_entries},
        )
        captures = {
            model_id: load_capture(root, entry)
            for model_id, entry in sorted(capture_entries.items())
        }
        metrics, training, predictions, checkpoints = evaluate_all_models(
            config,
            systems,
            basis,
            captures,
        )
        decision = gate_decision(config, metrics)
        atomic_write_json(root / "training_summary.json", training)
        _, checkpoint_paths = _write_array_groups(
            root,
            "checkpoints",
            checkpoints,
            schema="frank_eq_spq0_checkpoint_manifest_v1",
        )
        _, prediction_paths = _write_array_groups(
            root,
            "predictions",
            predictions,
            schema="frank_eq_spq0_predictions_manifest_v1",
        )
        metrics["prediction_digest_sha256"] = deterministic_prediction_digest(predictions)
        atomic_write_json(root / "metrics.json", metrics)
        atomic_write_json(root / "decision.json", decision)
        rate_compute = {
            "schema": "frank_eq_spq0_rate_compute_v1",
            "primary_packet": {
                "rank": basis.exact_rank,
                "bits_per_coordinate": 4,
                "payload_bits": 4 * basis.exact_rank,
                "framing_bits": None,
                "source_post_capture_queries": 0,
                "consumer": "frozen target-local reader",
            },
            "development_tomography": {
                "future_test_queries_per_prefix": len(basis.public_tests)
                + len(basis.target_tests),
                "categorical_candidates_per_query": len(
                    config.probability_protocol.bins
                ),
                "counts_as_primary_packet": False,
            },
            "amortized_frontier": {
                model_id: row["amortized_utility"]
                for model_id, row in metrics["models"].items()
            },
            "primary_amortized_query_count": config.gates.amortized_query_count_for_primary_utility,
            "one_direct_query_is_not_a_conjunctive_gate": True,
        }
        atomic_write_json(root / "rate_compute.json", rate_compute)
        status.update(
            {
                "state": "completed",
                "current_stage": None,
                "completed_stages": ["checkpoint_preflight", "audit"],
                "completed_at": _timestamp(),
                "elapsed_seconds": time.time() - started,
                "scientific_decision": decision,
            }
        )
        atomic_write_json(root / "workflow_status.json", status)
        summary = {
            "schema": "frank_eq_spq0_run_v1",
            "status": "completed",
            "workflow_integrity_passed": True,
            "development_only": True,
            "root": str(root),
            "models": [model.model_id for model in config.models],
            "reserved_models_loaded": 0,
            "systems": len(systems),
            "validation_only_systems": sum(
                system.role == "validation_only" for system in systems
            ),
            "test_histories": 0,
            "ordered_cross_family_pairs": len(metrics["cross_family_composition"]),
            "pair_specific_mappers": 0,
            "decision": decision,
            "authorization": decision["authorization"],
            "telemetry": telemetry.status(),
        }
        atomic_write_json(root / "run_summary.json", summary)
        capture_paths = [
            entry[key]
            for entry in capture_entries.values()
            for key in ("metadata", "array")
        ]
        artifact_paths = [
            "config.yaml",
            "protocol.md",
            "registration.json",
            "dry_run_plan.json",
            "run_manifest.json",
            "workflow_status.json",
            "checkpoint_preflight.json",
            "systems.json",
            "public_basis.json",
            "panel_manifest.json",
            *panel_paths,
            "models.json",
            "capture_manifest.json",
            *capture_paths,
            "training_summary.json",
            *checkpoint_paths,
            *prediction_paths,
            "metrics.json",
            "decision.json",
            "rate_compute.json",
            "run_summary.json",
        ]
        atomic_write_json(
            root / "artifact_manifest.json",
            _artifact_manifest(root, artifact_paths),
        )
        from .verify import verify_spq0_run

        verification = verify_spq0_run(
            root,
            config_path=config_file,
            write_verification=True,
        )
        if not verification["passed"]:
            raise RuntimeError("independent SPQ0 verification failed")
        telemetry.log(
            {"decision": {"status": decision["status"], "diagnosis": decision["diagnosis"]}}
        )
        return summary
    except Exception as error:
        status.update(
            {
                "state": "failed",
                "failed_at": _timestamp(),
                "elapsed_seconds": time.time() - started,
                "failure": {
                    "stage": status.get("current_stage"),
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        atomic_write_json(root / "workflow_status.json", status)
        raise
    finally:
        telemetry.finish()
