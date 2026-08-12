"""Fail-closed staged execution for the frozen Stage-A v3 representation run."""

from __future__ import annotations

import datetime as dt
import gc
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

from frank_eq.rate_compute.backend import RateComputeModelAdapter
from frank_eq.real_config import WandBLoggingConfig
from frank_eq.telemetry import WandbTelemetry
from frank_eq.utils import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

from .access import StageAV3AccessController
from .baselines import unified_operation_descriptors
from .capture import (
    build_capture_config,
    build_model_spec,
    capture_panel_shard,
    load_capture_shard,
    write_capture_shard,
)
from .config import StageAV3Config, StageAV3ModelSpec
from .controls import apply_train_controls, fit_train_controls
from .evaluation import evaluate_stagea_v3
from .packet import encode_typed_edge_packet
from .panel import V3Panel, generate_v3_panel
from .predictions import (
    V3PredictionBundle,
    assemble_prediction_bundle,
    load_prediction_bundle,
    write_prediction_bundle_artifacts,
)
from .training import (
    predict_basis_ensemble,
    predict_continuous_ensemble,
    train_basis_predictor,
    train_continuous_quotient,
)

STAGEA_V3_STAGE_ORDER = (
    "prepare",
    "founder_fit",
    "freeze",
    "held_onboard",
    "evaluate",
)
_OLIVIA_RUNTIME_IMAGE = "/cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif"
_OLIVIA_RUNTIME_IMAGE_SHA256 = "a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1"


def _timestamp() -> str:
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
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        "cluster": os.environ.get("FRANK_EQ_CLUSTER"),
        "source_sha256": os.environ.get("FRANK_EQ_SOURCE_SHA256"),
        "git_commit": os.environ.get("FRANK_EQ_GIT_COMMIT"),
        "git_dirty": os.environ.get("FRANK_EQ_GIT_DIRTY"),
        "project_version": os.environ.get("PROJECT_VERSION"),
        "runtime_image": os.environ.get("FRANK_EQ_RUNTIME_IMAGE"),
        "runtime_image_sha256": os.environ.get("FRANK_EQ_IMAGE_SHA256"),
        "hf_home": os.environ.get("HF_HOME"),
    }
    if torch.cuda.is_available():
        payload["accelerators"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return payload


def _validate_official_environment(environment: dict[str, Any]) -> None:
    """Fail before any task panel exists when immutable Olivia provenance is absent."""

    required_sha_fields = {"source_sha256": 64, "git_commit": 40}
    for key, expected_length in required_sha_fields.items():
        value = environment.get(key)
        if (
            not isinstance(value, str)
            or len(value) != expected_length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"official Stage-A v3 run requires a valid {key}")
    if environment.get("cluster") != "olivia":
        raise RuntimeError("official Stage-A v3 execution is registered only for Olivia")
    if environment.get("git_dirty") != "false":
        raise RuntimeError("official Stage-A v3 execution requires a clean immutable commit")
    if environment.get("runtime_image") != _OLIVIA_RUNTIME_IMAGE:
        raise RuntimeError("official Stage-A v3 runtime image path differs from the freeze")
    if environment.get("runtime_image_sha256") != _OLIVIA_RUNTIME_IMAGE_SHA256:
        raise RuntimeError("official Stage-A v3 runtime image hash differs from the freeze")
    if not str(environment.get("slurm_job_id") or "").isdigit():
        raise RuntimeError("official Stage-A v3 execution requires a Slurm job identity")
    if not environment.get("project_version"):
        raise RuntimeError("official Stage-A v3 execution requires an immutable project version")
    if not environment.get("cuda_available") or environment.get("cuda_device_count") != 1:
        raise RuntimeError("official Stage-A v3 execution requires exactly one visible CUDA GPU")


def parse_stagea_v3_stages(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        stages = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        stages = tuple(str(item) for item in value)
    if stages != STAGEA_V3_STAGE_ORDER:
        raise ValueError(
            "Stage-A v3 permits only the complete frozen sequence: "
            + ",".join(STAGEA_V3_STAGE_ORDER)
        )
    return stages


def build_stagea_v3_plan(
    config: StageAV3Config,
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    """Build a content-addressed plan without instantiating any panel or model."""

    panel = config.section("panel")
    compiler = config.section("compiler")
    rows: dict[str, Any] = {}
    totals = {"prefix_forwards": 0, "logical_source_queries": 0}
    for role, role_config in panel["roles"].items():
        renderers = 3 if role == "test" else 2
        rows[role] = {}
        for entity_count in panel["entity_counts"]:
            prefixes_per_model = int(role_config["worlds_per_complexity"]) * renderers
            logical_queries_per_prefix = entity_count * (entity_count - 1) + 3 * int(
                panel["n_target_operations"]
            )
            rows[role][str(entity_count)] = {
                "worlds": int(role_config["worlds_per_complexity"]),
                "renderers": renderers,
                "prefixes_per_model": prefixes_per_model,
                "logical_queries_per_prefix": logical_queries_per_prefix,
                "logical_queries_per_model": prefixes_per_model * logical_queries_per_prefix,
            }
            totals["prefix_forwards"] += prefixes_per_model * len(config.models)
            totals["logical_source_queries"] += (
                prefixes_per_model * logical_queries_per_prefix * len(config.models)
            )
    repository = _repository_root()
    config_source = Path(config_path).resolve()
    try:
        stable_config_path = config_source.relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError("Stage-A v3 config must be inside the repository") from error
    if config_source != config.source_path.resolve():
        raise ValueError("plan config path differs from the loaded frozen config")
    implementation_files = _implementation_files(repository, config_source)
    implementation_hashes = {
        path.relative_to(repository).as_posix(): sha256_file(path)
        for path in implementation_files
        if path.is_file()
    }
    payload: dict[str, Any] = {
        "schema": "frank_eq_stagea_v3_dry_run_plan_v1",
        "protocol_version": config.protocol_version,
        "run_name": config.run_name,
        "config_path": stable_config_path,
        "config_sha256": config.config_sha256,
        "implementation_files": implementation_hashes,
        "implementation_tree_sha256": sha256_bytes(canonical_json_bytes(implementation_hashes)),
        "stage_order": list(STAGEA_V3_STAGE_ORDER),
        "model_order": [model.model_id for model in config.models],
        "model_revisions": {model.model_id: model.revision for model in config.models},
        "held_model_task_opened": False,
        "test_panel_instantiated": False,
        "rows": rows,
        "totals": totals,
        "compiler": {
            "channels": compiler["channels"],
            "seeds": compiler["seeds"],
            "model_dim": compiler["model_dim"],
            "attention_blocks": compiler["attention_blocks"],
        },
        "test_access": {
            "count": 1,
            "requires_founder_freeze": True,
            "requires_held_freeze": True,
        },
        "protected_authorization": config.section("authorization"),
    }
    payload["plan_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def write_stagea_v3_plan(path: str | Path, plan: dict[str, Any]) -> str:
    atomic_write_json(path, plan)
    return sha256_file(path)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _implementation_files(repository: Path, config_source: Path) -> list[Path]:
    """Return the complete code and protocol surface bound by the run plan."""

    files = set((repository / "src/frank_eq").rglob("*.py"))
    files.update(
        {
            repository / "docs/20_STAGEA_V3_PROTOCOL.md",
            repository / "configs/stagea_v3/registration.json",
            config_source,
            repository / "olivia/quickstart.sh",
            repository / "olivia/stagea_v3.slurm",
        }
    )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Stage-A v3 implementation surface is incomplete: "
            + ", ".join(str(path) for path in sorted(missing))
        )
    return sorted(files)


def _write_panel(path: Path, panel: V3Panel) -> str:
    atomic_write_json(path, panel.to_dict())
    return sha256_file(path)


def _load_panel(path: Path, *, expected_sha256: str | None = None) -> V3Panel:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"Stage-A v3 panel hash mismatch: {path}")
    return V3Panel.from_dict(json.loads(path.read_text()))


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_hashes(root: Path, paths: list[Path]) -> dict[str, str]:
    return {_relative(root, path): sha256_file(path) for path in sorted(paths)}


def _source_contract(root: Path, config: StageAV3Config) -> tuple[Path, dict[str, Any]]:
    repository = _repository_root()
    source_files = _implementation_files(repository, config.source_path.resolve())
    files = {
        path.resolve().relative_to(repository).as_posix(): sha256_file(path)
        for path in source_files
    }
    payload = {
        "schema": "frank_eq_stagea_v3_implementation_manifest_v1",
        "config_sha256": config.config_sha256,
        "source_archive_sha256": os.environ.get("FRANK_EQ_SOURCE_SHA256"),
        "git_commit": os.environ.get("FRANK_EQ_GIT_COMMIT"),
        "git_dirty": os.environ.get("FRANK_EQ_GIT_DIRTY"),
        "runtime_image": os.environ.get("FRANK_EQ_RUNTIME_IMAGE"),
        "runtime_image_sha256": os.environ.get("FRANK_EQ_IMAGE_SHA256"),
        "files": files,
        "implementation_tree_sha256": sha256_bytes(canonical_json_bytes(files)),
    }
    path = root / "implementation_manifest.json"
    atomic_write_json(path, payload)
    return path, payload


def _write_role_panels(
    root: Path,
    config: StageAV3Config,
    role: str,
    *,
    test_access_grant: dict[str, Any] | None = None,
) -> tuple[dict[int, V3Panel], Path, dict[str, str]]:
    panels: dict[int, V3Panel] = {}
    files: dict[str, str] = {}
    for entity_count in config.section("panel")["entity_counts"]:
        panel = generate_v3_panel(
            config,
            role,
            entity_count,
            test_access_grant=test_access_grant,
        )
        path = root / "panels" / f"{role}_n{entity_count}.json"
        digest = _write_panel(path, panel)
        panels[entity_count] = panel
        files[_relative(root, path)] = digest
    manifest_path = root / f"{role}_panel_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema": "frank_eq_stagea_v3_panel_manifest_v1",
            "config_sha256": config.config_sha256,
            "role": role,
            "files": files,
            "operation_registry_sha256": {
                str(entity_count): panel.operation_registry_sha256
                for entity_count, panel in panels.items()
            },
        },
    )
    return panels, manifest_path, files


def _capture_path(root: Path, role: str, model_id: str, entity_count: int) -> Path:
    return root / "captures" / role / f"{model_id}-n{entity_count}.pt"


def _capture_model_roles(
    root: Path,
    config: StageAV3Config,
    model: StageAV3ModelSpec,
    panels_by_role: dict[str, dict[int, V3Panel]],
    telemetry: WandbTelemetry,
) -> dict[tuple[str, int], dict[str, Any]]:
    print(f"stagea-v3 loading frozen model {model.model_id} at {model.revision}", flush=True)
    adapter = RateComputeModelAdapter(
        build_model_spec(model),
        build_capture_config(config),
    )
    result: dict[tuple[str, int], dict[str, Any]] = {}
    try:
        for role, panels in panels_by_role.items():
            for entity_count, panel in sorted(panels.items()):
                started = time.time()
                shard = capture_panel_shard(
                    config,
                    model,
                    panel,
                    adapter=adapter,
                )
                path = _capture_path(root, role, model.model_id, entity_count)
                digest = write_capture_shard(path, shard)
                result[(role, entity_count)] = {
                    "path": _relative(root, path),
                    "sha256": digest,
                    "summary": shard.capture_summary,
                    "elapsed_seconds": time.time() - started,
                }
                telemetry.log(
                    {
                        "capture": {
                            "model_id": model.model_id,
                            "role": role,
                            "entity_count": entity_count,
                            "rows": shard.rows,
                            "elapsed_seconds": time.time() - started,
                        }
                    }
                )
                del shard
    finally:
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _write_operation_descriptors(
    root: Path,
    panels: dict[int, V3Panel],
) -> tuple[dict[int, torch.Tensor], dict[str, str], list[Path]]:
    descriptors: dict[int, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    paths: list[Path] = []
    for entity_count, panel in sorted(panels.items()):
        values = unified_operation_descriptors(
            panel.panel.operations,
            entity_count=entity_count,
        )
        path = root / "operation_descriptors" / f"n{entity_count}.json"
        atomic_write_json(
            path,
            {
                "schema": "frank_eq_stagea_v3_operation_descriptors_v1",
                "entity_count": entity_count,
                "operation_registry_sha256": panel.operation_registry_sha256,
                "values": values.tolist(),
            },
        )
        descriptors[entity_count] = torch.from_numpy(values).float()
        hashes[f"n{entity_count}"] = sha256_file(path)
        paths.append(path)
    return descriptors, hashes, paths


def _compact_training(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "history"}


def _fit_model(
    root: Path,
    config: StageAV3Config,
    model: StageAV3ModelSpec,
    capture_index: dict[tuple[str, int], dict[str, Any]],
    panels: dict[int, V3Panel],
    descriptors: dict[int, torch.Tensor],
    descriptor_hashes: dict[str, str],
    telemetry: WandbTelemetry,
) -> tuple[dict[str, Any], list[Path], dict[str, Any]]:
    train_shards = {
        entity_count: load_capture_shard(
            root / capture_index[("train", entity_count)]["path"],
            expected_sha256=capture_index[("train", entity_count)]["sha256"],
        )
        for entity_count in config.section("panel")["entity_counts"]
    }
    validation_shards = {
        entity_count: load_capture_shard(
            root / capture_index[("validation", entity_count)]["path"],
            expected_sha256=capture_index[("validation", entity_count)]["sha256"],
        )
        for entity_count in config.section("panel")["entity_counts"]
    }
    capture_hashes = {row["path"]: row["sha256"] for row in capture_index.values()}
    registry: dict[str, Any] = {
        "model_id": model.model_id,
        "model_role": model.role,
        "semantic": [],
        "behavioral": [],
        "token_id": [],
        "final_token": [],
        "continuous": [],
    }
    paths: list[Path] = []
    summary: dict[str, Any] = {"model_id": model.model_id, "fits": []}
    onboarding = model.role == "held"
    for registered_seed in config.section("compiler")["seeds"]:
        jobs = (
            ("activation", "semantic", "semantic"),
            ("activation", "behavioral", "behavioral"),
            ("token_id", "semantic", "token_id"),
            ("final_token", "semantic", "final_token"),
        )
        for kind, channel, registry_key in jobs:
            path = (
                root
                / "checkpoints"
                / model.model_id
                / f"{registry_key}-{channel}-seed{registered_seed}.pt"
            )
            metadata = train_basis_predictor(
                config,
                train_shards=train_shards,
                validation_shards=validation_shards,
                kind=kind,
                channel=channel,
                registered_seed=registered_seed,
                checkpoint_path=path,
                onboarding=onboarding,
                capture_sha256=capture_hashes,
            )
            entry = {
                "path": _relative(root, path),
                "sha256": sha256_file(path),
                "metadata": _compact_training(metadata),
            }
            registry[registry_key].append(entry)
            summary["fits"].append(entry["metadata"])
            paths.append(path)
            telemetry.log(
                {
                    "fit": {
                        "model_id": model.model_id,
                        "kind": kind,
                        "channel": channel,
                        "seed": registered_seed,
                        "best_metric": metadata["best_selection_metric"],
                        "best_epoch": metadata["best_epoch"],
                    }
                }
            )

        continuous_path = (
            root / "checkpoints" / model.model_id / f"continuous-semantic-seed{registered_seed}.pt"
        )
        continuous_metadata = train_continuous_quotient(
            config,
            train_shards=train_shards,
            validation_shards=validation_shards,
            operation_descriptors=descriptors,
            registered_seed=registered_seed,
            checkpoint_path=continuous_path,
            onboarding=onboarding,
            capture_sha256=capture_hashes,
            descriptor_sha256=descriptor_hashes,
        )
        continuous_entry = {
            "path": _relative(root, continuous_path),
            "sha256": sha256_file(continuous_path),
            "metadata": _compact_training(continuous_metadata),
        }
        registry["continuous"].append(continuous_entry)
        summary["fits"].append(continuous_entry["metadata"])
        paths.append(continuous_path)
        telemetry.log(
            {
                "fit": {
                    "model_id": model.model_id,
                    "kind": "historical_continuous_quotient",
                    "seed": registered_seed,
                    "best_metric": continuous_metadata["best_selection_metric"],
                    "best_epoch": continuous_metadata["best_epoch"],
                }
            }
        )

    controls = fit_train_controls(
        config,
        model_id=model.model_id,
        train_shards=train_shards,
        panels=panels,
        capture_sha256=capture_hashes,
    )
    controls_path = root / "controls" / f"{model.model_id}.json"
    atomic_write_json(controls_path, controls)
    registry["controls"] = {
        "path": _relative(root, controls_path),
        "sha256": sha256_file(controls_path),
    }
    paths.append(controls_path)
    del train_shards, validation_shards
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return registry, paths, summary


def _checkpoint_paths(root: Path, registry: dict[str, Any], key: str) -> list[Path]:
    return [root / entry["path"] for entry in registry[key]]


def _checkpoint_hashes(registry: dict[str, Any], key: str) -> dict[str, str]:
    return {Path(entry["path"]).name: str(entry["sha256"]) for entry in registry[key]}


def _quantize_identity(probabilities: np.ndarray, entity_count: int) -> np.ndarray:
    return np.stack(
        [
            encode_typed_edge_packet(row, entity_count=entity_count, bits=4).probabilities()
            for row in probabilities
        ]
    )


def _synchronize_accelerator() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _token_slot_flops_estimate(
    config: StageAV3Config,
    shard: Any,
    *,
    ensemble_size: int,
) -> int:
    """Estimate dense inference FLOPs; one multiply-add counts as two FLOPs."""

    compiler = config.section("compiler")
    batch = int(shard.rows)
    depths = int(shard.residuals.shape[1])
    tokens = int(shard.residuals.shape[2])
    input_width = int(shard.residuals.shape[3])
    coordinates = int(shard.coordinate_count)
    model_dim = int(compiler["model_dim"])
    feedforward_dim = int(compiler["feedforward_dim"])
    blocks = int(compiler["attention_blocks"])
    memory = depths * tokens
    projection = 2 * batch * depths * tokens * input_width * model_dim
    position = 2 * tokens * (model_dim + model_dim * model_dim)
    per_block = batch * (
        2 * model_dim * model_dim * (2 * coordinates + 2 * memory)
        + 4 * coordinates * memory * model_dim
        + 4 * coordinates * model_dim * feedforward_dim
    )
    output = 2 * batch * coordinates * model_dim
    return int(ensemble_size * (projection + position + blocks * per_block + output))


def _dense_parameter_flops_estimate(
    parameter_count: int,
    output_rows: int,
    *,
    ensemble_size: int,
) -> int:
    """Conservative dense parameter-application proxy for control architectures."""

    return int(2 * parameter_count * output_rows * ensemble_size)


def _predict_model(
    root: Path,
    config: StageAV3Config,
    model: StageAV3ModelSpec,
    registry: dict[str, Any],
    capture_index: dict[tuple[str, int], dict[str, Any]],
    test_panels: dict[int, V3Panel],
    descriptors: dict[int, torch.Tensor],
    controller: StageAV3AccessController,
) -> tuple[list[V3PredictionBundle], dict[tuple[str, int], dict[str, np.ndarray]], list[Path]]:
    controls_path = root / registry["controls"]["path"]
    if sha256_file(controls_path) != registry["controls"]["sha256"]:
        raise ValueError("train-control artifact hash mismatch before evaluation")
    controls_artifact = json.loads(controls_path.read_text())
    bundles: list[V3PredictionBundle] = []
    identity: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    artifact_paths: list[Path] = []
    for entity_count in config.section("panel")["entity_counts"]:
        train_entry = capture_index[("train", entity_count)]
        train_shard = load_capture_shard(
            root / train_entry["path"], expected_sha256=train_entry["sha256"]
        )
        test_entry = capture_index[("test", entity_count)]
        controller.record_test_file_open(test_entry["path"])
        test_shard = load_capture_shard(
            root / test_entry["path"], expected_sha256=test_entry["sha256"]
        )

        inference_wall_seconds: dict[str, float] = {}
        _synchronize_accelerator()
        inference_started = time.perf_counter()
        semantic, semantic_logits, semantic_meta = predict_basis_ensemble(
            config,
            _checkpoint_paths(root, registry, "semantic"),
            test_shard,
            checkpoint_sha256=_checkpoint_hashes(registry, "semantic"),
        )
        _synchronize_accelerator()
        inference_wall_seconds["semantic"] = time.perf_counter() - inference_started

        _synchronize_accelerator()
        inference_started = time.perf_counter()
        train_semantic, _, _ = predict_basis_ensemble(
            config,
            _checkpoint_paths(root, registry, "semantic"),
            train_shard,
            checkpoint_sha256=_checkpoint_hashes(registry, "semantic"),
        )
        _synchronize_accelerator()
        inference_wall_seconds["semantic_train_identity"] = time.perf_counter() - inference_started

        _synchronize_accelerator()
        inference_started = time.perf_counter()
        behavioral, behavioral_logits, behavioral_meta = predict_basis_ensemble(
            config,
            _checkpoint_paths(root, registry, "behavioral"),
            test_shard,
            checkpoint_sha256=_checkpoint_hashes(registry, "behavioral"),
        )
        _synchronize_accelerator()
        inference_wall_seconds["behavioral"] = time.perf_counter() - inference_started

        _synchronize_accelerator()
        inference_started = time.perf_counter()
        token, token_logits, token_meta = predict_basis_ensemble(
            config,
            _checkpoint_paths(root, registry, "token_id"),
            test_shard,
            checkpoint_sha256=_checkpoint_hashes(registry, "token_id"),
        )
        _synchronize_accelerator()
        inference_wall_seconds["token_id"] = time.perf_counter() - inference_started

        _synchronize_accelerator()
        inference_started = time.perf_counter()
        final_token, final_logits, final_meta = predict_basis_ensemble(
            config,
            _checkpoint_paths(root, registry, "final_token"),
            test_shard,
            checkpoint_sha256=_checkpoint_hashes(registry, "final_token"),
        )
        _synchronize_accelerator()
        inference_wall_seconds["final_token"] = time.perf_counter() - inference_started

        _synchronize_accelerator()
        inference_started = time.perf_counter()
        continuous, continuous_logits, continuous_meta = predict_continuous_ensemble(
            config,
            _checkpoint_paths(root, registry, "continuous"),
            test_shard,
            descriptors[entity_count],
            checkpoint_sha256=_checkpoint_hashes(registry, "continuous"),
        )
        _synchronize_accelerator()
        inference_wall_seconds["continuous"] = time.perf_counter() - inference_started

        controls_started = time.perf_counter()
        controls = apply_train_controls(
            config,
            controls_artifact,
            test_shard,
            test_panels[entity_count],
        )
        inference_wall_seconds["train_controls"] = time.perf_counter() - controls_started
        compiler_parameters = {
            "semantic": int(semantic_meta[0]["parameter_count"]),
            "behavioral": int(behavioral_meta[0]["parameter_count"]),
            "token_id": int(token_meta[0]["parameter_count"]),
            "final_token": int(final_meta[0]["parameter_count"]),
            "continuous": int(continuous_meta[0]["parameter_count"]),
        }
        ensemble_size = len(config.section("compiler")["seeds"])
        token_slot_flops = _token_slot_flops_estimate(
            config,
            test_shard,
            ensemble_size=ensemble_size,
        )
        compiler_flops = {
            "semantic": token_slot_flops,
            "behavioral": token_slot_flops,
            "token_id": token_slot_flops,
            "final_token": _dense_parameter_flops_estimate(
                compiler_parameters["final_token"],
                test_shard.rows,
                ensemble_size=ensemble_size,
            ),
            "continuous": _dense_parameter_flops_estimate(
                compiler_parameters["continuous"],
                test_shard.rows * int(test_shard.operation_targets.shape[1]),
                ensemble_size=ensemble_size,
            ),
        }
        bundle = assemble_prediction_bundle(
            config,
            shard=test_shard,
            panel=test_panels[entity_count],
            semantic_primary=semantic.numpy(),
            behavioral_primary=behavioral.numpy(),
            token_primary=token.numpy(),
            final_token_primary=final_token.numpy(),
            continuous_primary=continuous.numpy(),
            semantic_seed_logits=[value.numpy() for value in semantic_logits],
            behavioral_seed_logits=[value.numpy() for value in behavioral_logits],
            token_seed_logits=[value.numpy() for value in token_logits],
            final_token_seed_logits=[value.numpy() for value in final_logits],
            continuous_seed_logits=[value.numpy() for value in continuous_logits],
            controls=controls,
            compiler_compute={
                "parameter_count": compiler_parameters,
                "flops_estimate": compiler_flops,
                "inference_wall_seconds": inference_wall_seconds,
                "ensemble_size": ensemble_size,
                "flops_convention": (
                    "multiply_add_equals_two; token-slot estimate follows dense "
                    "projection/attention/feedforward tensor shapes including padded "
                    "tokens; final-token and continuous controls use a declared dense "
                    "parameter-application proxy"
                ),
            },
        )
        array_path = root / "predictions" / f"{model.model_id}-n{entity_count}.npz"
        metadata_path = root / "predictions" / f"{model.model_id}-n{entity_count}.json"
        write_prediction_bundle_artifacts(
            array_path,
            metadata_path,
            bundle,
            config_sha256=config.config_sha256,
        )
        controller.record_test_file_open(_relative(root, array_path))
        controller.record_test_file_open(_relative(root, metadata_path))
        restored = load_prediction_bundle(
            array_path,
            metadata_path,
            config_sha256=config.config_sha256,
        )
        bundles.append(restored)
        artifact_paths.extend([array_path, metadata_path])

        identity[(model.model_id, entity_count)] = {
            "probabilities": _quantize_identity(train_semantic.numpy(), entity_count),
            "world_ids": train_shard.world_ids.numpy().astype(np.int64),
            "renderer_ids": train_shard.renderer_ids.numpy().astype(np.int64),
        }
        del train_shard, test_shard
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return bundles, identity, artifact_paths


def _write_identity_basis(
    root: Path,
    identity: dict[tuple[str, int], dict[str, np.ndarray]],
) -> tuple[Path, dict[str, str]]:
    files: dict[str, str] = {}
    for (model_id, entity_count), values in sorted(identity.items()):
        path = root / "identity_train_basis" / f"{model_id}-n{entity_count}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **values)
        os.replace(temporary, path)
        files[_relative(root, path)] = sha256_file(path)
    manifest_path = root / "identity_train_basis_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema": "frank_eq_stagea_v3_identity_train_basis_v1",
            "fit_role": "train",
            "files": files,
        },
    )
    return manifest_path, files


def _test_artifact_paths(root: Path, config: StageAV3Config) -> list[str]:
    paths = [
        f"panels/test_n{entity_count}.json"
        for entity_count in config.section("panel")["entity_counts"]
    ]
    paths.append("test_panel_manifest.json")
    for model in config.models:
        for entity_count in config.section("panel")["entity_counts"]:
            paths.extend(
                [
                    f"captures/test/{model.model_id}-n{entity_count}.pt",
                    f"predictions/{model.model_id}-n{entity_count}.npz",
                    f"predictions/{model.model_id}-n{entity_count}.json",
                ]
            )
    return paths


def _integrity_checks(
    root: Path,
    config: StageAV3Config,
    capture_registry: dict[str, dict[tuple[str, int], dict[str, Any]]],
    checkpoint_registry: dict[str, dict[str, Any]],
    access_ledger: dict[str, Any],
    bundles: list[V3PredictionBundle],
) -> dict[str, bool]:
    summaries = [
        row["summary"] for model_rows in capture_registry.values() for row in model_rows.values()
    ]
    branch_valid = all(
        row.get("exact_replay_response_branches") == 0
        and row.get("allow_exact_replay_fallback") is False
        and row.get("exclusive_cache_batching") is True
        and row.get("primary_compiler_post_capture_source_queries") == 0
        and row.get("behavioral_probability_storage_epsilon") == 1e-7
        and isinstance(row.get("behavioral_probability_clamp_count"), int)
        and row.get("behavioral_probability_clamp_count") >= 0
        and row.get("logical_post_capture_source_queries")
        == row.get("kv_cloned_response_branches")
        == row.get("exact_prefix_continuity_checks")
        for row in summaries
    )
    revisions = {row["model_id"]: row["revision_observed"] for row in summaries}
    revision_valid = all(revisions.get(model.model_id) == model.revision for model in config.models)
    expected_seeds = config.section("compiler")["seeds"]
    checkpoints_valid = all(
        all(
            [entry["metadata"]["registered_seed"] for entry in registry[key]] == expected_seeds
            for key in ("semantic", "behavioral", "token_id", "final_token", "continuous")
        )
        for registry in checkpoint_registry.values()
    )
    registered = set(access_ledger.get("registered_test_files", []))
    open_rows = access_ledger.get("test_file_opens", [])
    opened = {row["path"] for row in open_rows}
    opened_hashes_valid = all(
        (root / row["path"]).is_file() and sha256_file(root / row["path"]) == row.get("sha256")
        for row in open_rows
    )
    compute_complete = all(
        set(bundle.compute.get("parameter_count", {}))
        == {"semantic", "behavioral", "token_id", "final_token", "continuous"}
        and set(bundle.compute.get("flops_estimate", {}))
        == {"semantic", "behavioral", "token_id", "final_token", "continuous"}
        and set(bundle.compute.get("inference_wall_seconds", {}))
        >= {"semantic", "behavioral", "token_id", "final_token", "continuous"}
        and "executor_wall_seconds" in bundle.compute
        for bundle in bundles
    )
    authorization = config.section("authorization")
    protected_closed = not any(
        authorization[key]
        for key in (
            "receiver_execution_authorized",
            "new_receiver_world_access_authorized",
            "scientific_claim_authorized",
            "paper_claim_authorized",
        )
    )
    return {
        "config_snapshot_hash": sha256_file(root / "config.yaml") == config.config_sha256,
        "exclusive_kv_and_prefix_continuity": branch_valid,
        "model_revisions_exact": revision_valid,
        "checkpoint_seed_registry_complete": checkpoints_valid,
        "test_access_consumed_once": access_ledger.get("test_access_count") == 1,
        "test_files_registered_and_opened": (
            bool(registered) and registered == opened and opened_hashes_valid
        ),
        "founder_freeze_present": (root / "freeze_manifest.json").is_file(),
        "held_freeze_present": (root / "held_onboarding_manifest.json").is_file(),
        "protected_authorizations_closed": protected_closed,
        "required_baselines_complete": all(
            len(bundle.semantic_basis) == 15 and len(bundle.operations) == 18 for bundle in bundles
        ),
        "consumer_compute_declared": compute_complete,
    }


def run_stagea_v3(
    config: StageAV3Config,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    stages: str | list[str] | tuple[str, ...] = STAGEA_V3_STAGE_ORDER,
    dry_run_plan: dict[str, Any] | None = None,
    inspected_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Execute the one authorized representation workflow and preserve any miss."""

    selected_stages = parse_stagea_v3_stages(stages)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Stage-A v3 output directory must be empty")
    config_source = Path(config_path).resolve()
    shutil.copyfile(config_source, root / "config.yaml")
    repository = _repository_root()
    shutil.copyfile(repository / "docs/20_STAGEA_V3_PROTOCOL.md", root / "protocol.md")
    shutil.copyfile(
        repository / "configs/stagea_v3/registration.json",
        root / "registration.json",
    )
    expected_plan = build_stagea_v3_plan(config, config_path=config_source)
    plan = dry_run_plan or expected_plan
    if canonical_json_bytes(plan) != canonical_json_bytes(expected_plan):
        raise ValueError("dry-run plan differs from the current config/implementation plan")
    if not inspected_plan_sha256 or plan.get("plan_sha256") != inspected_plan_sha256:
        raise ValueError("run requires the exact human-inspected dry-run plan SHA-256")
    write_stagea_v3_plan(root / "dry_run_plan.json", plan)
    environment = _environment()
    _validate_official_environment(environment)
    implementation_path, implementation = _source_contract(root, config)
    run_manifest = {
        "schema": "frank_eq_stagea_v3_run_manifest_v1",
        "run_name": config.run_name,
        "protocol_version": config.protocol_version,
        "config_sha256": config.config_sha256,
        "created_at": _timestamp(),
        "stages": list(selected_stages),
        "output_dir": str(root),
        "environment": environment,
        "implementation_manifest_sha256": sha256_file(implementation_path),
        "dry_run_plan_sha256": sha256_file(root / "dry_run_plan.json"),
        "inspected_plan_sha256": inspected_plan_sha256,
        "representation_only": True,
        "receiver_execution_authorized": False,
        "test_access_limit": 1,
    }
    atomic_write_json(root / "run_manifest.json", run_manifest)
    status: dict[str, Any] = {
        "schema": "frank_eq_stagea_v3_workflow_status_v1",
        "state": "running",
        "current_stage": "prepare",
        "completed_stages": [],
        "started_at": _timestamp(),
        "failure": None,
        "test_access_consumed": False,
    }
    atomic_write_json(root / "workflow_status.json", status)
    controller = StageAV3AccessController(root, config_sha256=config.config_sha256)
    controller.initialize()
    logging = config.section("logging")["wandb"]
    telemetry = WandbTelemetry(
        WandBLoggingConfig(**logging),
        run_name=config.run_name,
        job={
            "cluster": environment.get("cluster"),
            "slurm_job_id": environment.get("slurm_job_id"),
            "source_sha256": environment.get("source_sha256"),
            "git_commit": environment.get("git_commit"),
        },
    )
    started = time.time()
    capture_registry: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    checkpoint_registry: dict[str, dict[str, Any]] = {}
    training_summaries: dict[str, Any] = {}
    all_artifact_paths: list[Path] = [
        root / name
        for name in (
            "config.yaml",
            "protocol.md",
            "registration.json",
            "dry_run_plan.json",
            "implementation_manifest.json",
            "run_manifest.json",
            "workflow_status.json",
            "access_ledger.json",
        )
    ]
    try:
        print("stagea-v3 stage=prepare generating train/validation panels", flush=True)
        train_panels, train_manifest, train_panel_files = _write_role_panels(root, config, "train")
        validation_panels, validation_manifest, validation_panel_files = _write_role_panels(
            root, config, "validation"
        )
        all_artifact_paths.extend(
            [
                train_manifest,
                validation_manifest,
                *[root / path for path in train_panel_files],
                *[root / path for path in validation_panel_files],
            ]
        )
        descriptors, descriptor_hashes, descriptor_paths = _write_operation_descriptors(
            root, train_panels
        )
        all_artifact_paths.extend(descriptor_paths)
        controller.advance("founder_fit")
        status.update({"current_stage": "founder_fit", "completed_stages": ["prepare"]})
        atomic_write_json(root / "workflow_status.json", status)

        founder_paths: list[Path] = []
        for model in config.founder_models:
            capture_index = _capture_model_roles(
                root,
                config,
                model,
                {"train": train_panels, "validation": validation_panels},
                telemetry,
            )
            capture_registry[model.model_id] = capture_index
            founder_paths.extend(root / row["path"] for row in capture_index.values())
            registry, paths, summary = _fit_model(
                root,
                config,
                model,
                capture_index,
                train_panels,
                descriptors,
                descriptor_hashes,
                telemetry,
            )
            checkpoint_registry[model.model_id] = registry
            training_summaries[model.model_id] = summary
            founder_paths.extend(paths)

        founder_checkpoint_manifest = root / "founder_checkpoints_manifest.json"
        atomic_write_json(
            founder_checkpoint_manifest,
            {
                "schema": "frank_eq_stagea_v3_checkpoint_manifest_v1",
                "config_sha256": config.config_sha256,
                "model_role": "founder",
                "models": {
                    model.model_id: checkpoint_registry[model.model_id]
                    for model in config.founder_models
                },
            },
        )
        founder_paths.append(founder_checkpoint_manifest)
        freeze_artifacts = _artifact_hashes(
            root,
            [
                root / "config.yaml",
                root / "protocol.md",
                root / "registration.json",
                root / "dry_run_plan.json",
                implementation_path,
                train_manifest,
                validation_manifest,
                *descriptor_paths,
                *founder_paths,
            ],
        )
        atomic_write_json(
            root / "freeze_manifest.json",
            {
                "schema": "frank_eq_stagea_v3_freeze_v1",
                "status": "frozen",
                "config_sha256": config.config_sha256,
                "frozen_at": _timestamp(),
                "founder_models": [model.model_id for model in config.founder_models],
                "test_files_existing": [],
                "artifacts": freeze_artifacts,
            },
        )
        all_artifact_paths.extend([*founder_paths, root / "freeze_manifest.json"])
        controller.advance("freeze")
        controller.advance("held_onboard")
        status.update(
            {
                "current_stage": "held_onboard",
                "completed_stages": ["prepare", "founder_fit", "freeze"],
            }
        )
        atomic_write_json(root / "workflow_status.json", status)

        held = config.held_model
        held_capture = _capture_model_roles(
            root,
            config,
            held,
            {"train": train_panels, "validation": validation_panels},
            telemetry,
        )
        capture_registry[held.model_id] = held_capture
        held_paths = [root / row["path"] for row in held_capture.values()]
        held_registry, paths, held_summary = _fit_model(
            root,
            config,
            held,
            held_capture,
            train_panels,
            descriptors,
            descriptor_hashes,
            telemetry,
        )
        checkpoint_registry[held.model_id] = held_registry
        training_summaries[held.model_id] = held_summary
        held_paths.extend(paths)
        held_checkpoint_manifest = root / "held_checkpoints_manifest.json"
        atomic_write_json(
            held_checkpoint_manifest,
            {
                "schema": "frank_eq_stagea_v3_checkpoint_manifest_v1",
                "config_sha256": config.config_sha256,
                "model_role": "held",
                "models": {held.model_id: held_registry},
            },
        )
        held_paths.append(held_checkpoint_manifest)
        atomic_write_json(
            root / "held_onboarding_manifest.json",
            {
                "schema": "frank_eq_stagea_v3_held_onboarding_v1",
                "status": "frozen",
                "config_sha256": config.config_sha256,
                "frozen_at": _timestamp(),
                "held_model": held.model_id,
                "founder_freeze_sha256": sha256_file(root / "freeze_manifest.json"),
                "test_files_existing": [],
                "artifacts": _artifact_hashes(
                    root,
                    [root / "freeze_manifest.json", *held_paths],
                ),
            },
        )
        all_artifact_paths.extend([*held_paths, root / "held_onboarding_manifest.json"])

        test_files = _test_artifact_paths(root, config)
        grant = controller.assert_can_create_test(test_files)
        status.update(
            {
                "current_stage": "evaluate",
                "completed_stages": [
                    "prepare",
                    "founder_fit",
                    "freeze",
                    "held_onboard",
                ],
                "test_access_consumed": True,
            }
        )
        atomic_write_json(root / "workflow_status.json", status)
        print("stagea-v3 test access consumed; generating the registered test panel", flush=True)
        test_panels, test_manifest, test_panel_files = _write_role_panels(
            root,
            config,
            "test",
            test_access_grant=grant,
        )
        controller.record_test_file_open(_relative(root, test_manifest))
        for path in test_panel_files:
            controller.record_test_file_open(path)
        test_panels = {
            entity_count: _load_panel(
                root / f"panels/test_n{entity_count}.json",
                expected_sha256=test_panel_files[f"panels/test_n{entity_count}.json"],
            )
            for entity_count in config.section("panel")["entity_counts"]
        }
        all_artifact_paths.extend([test_manifest, *[root / path for path in test_panel_files]])

        for model in config.models:
            test_capture = _capture_model_roles(
                root,
                config,
                model,
                {"test": test_panels},
                telemetry,
            )
            capture_registry[model.model_id].update(test_capture)
            all_artifact_paths.extend(root / row["path"] for row in test_capture.values())

        all_bundles: list[V3PredictionBundle] = []
        identity_train_basis: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        prediction_paths: list[Path] = []
        for model in config.models:
            bundles, identity, paths = _predict_model(
                root,
                config,
                model,
                checkpoint_registry[model.model_id],
                capture_registry[model.model_id],
                test_panels,
                descriptors,
                controller,
            )
            all_bundles.extend(bundles)
            identity_train_basis.update(identity)
            prediction_paths.extend(paths)
        all_artifact_paths.extend(prediction_paths)

        identity_manifest, identity_files = _write_identity_basis(root, identity_train_basis)
        all_artifact_paths.extend([identity_manifest, *[root / path for path in identity_files]])
        compiler_manifest = root / "compiler_checkpoints_manifest.json"
        atomic_write_json(
            compiler_manifest,
            {
                "schema": "frank_eq_stagea_v3_checkpoint_manifest_v1",
                "config_sha256": config.config_sha256,
                "models": checkpoint_registry,
            },
        )
        training_summary_path = root / "training_summary.json"
        atomic_write_json(
            training_summary_path,
            {
                "schema": "frank_eq_stagea_v3_training_summary_v1",
                "config_sha256": config.config_sha256,
                "models": training_summaries,
            },
        )
        baseline_manifest = root / "baseline_manifest.json"
        atomic_write_json(
            baseline_manifest,
            {
                "schema": "frank_eq_stagea_v3_baseline_manifest_v1",
                "required": config.section("baselines")["required"],
                "control_artifacts": {
                    model_id: registry["controls"]
                    for model_id, registry in checkpoint_registry.items()
                },
                "conditions_complete": all(
                    len(bundle.semantic_basis) == 15 and len(bundle.operations) == 18
                    for bundle in all_bundles
                ),
            },
        )
        prediction_manifest = root / "predictions_manifest.json"
        prediction_entries = {
            f"{bundle.model_id}|{bundle.entity_count}": {
                "array": f"predictions/{bundle.model_id}-n{bundle.entity_count}.npz",
                "array_sha256": sha256_file(
                    root / f"predictions/{bundle.model_id}-n{bundle.entity_count}.npz"
                ),
                "metadata": f"predictions/{bundle.model_id}-n{bundle.entity_count}.json",
                "metadata_sha256": sha256_file(
                    root / f"predictions/{bundle.model_id}-n{bundle.entity_count}.json"
                ),
            }
            for bundle in all_bundles
        }
        atomic_write_json(
            prediction_manifest,
            {
                "schema": "frank_eq_stagea_v3_predictions_manifest_v1",
                "config_sha256": config.config_sha256,
                "entries": prediction_entries,
            },
        )
        model_rows = [
            {
                "model_id": model.model_id,
                "role": model.role,
                "revision_requested": model.revision,
                "revision_observed": next(
                    row["summary"]["revision_observed"]
                    for row in capture_registry[model.model_id].values()
                ),
                "capture_groups": {
                    f"{role}|{entity_count}": row
                    for (role, entity_count), row in sorted(
                        capture_registry[model.model_id].items()
                    )
                },
            }
            for model in config.models
        ]
        models_path = root / "models.json"
        atomic_write_json(models_path, model_rows)
        capture_validation = root / "capture_validation.json"
        access_ledger = controller.read()
        integrity = _integrity_checks(
            root,
            config,
            capture_registry,
            checkpoint_registry,
            access_ledger,
            all_bundles,
        )
        atomic_write_json(
            capture_validation,
            {
                "schema": "frank_eq_stagea_v3_capture_validation_v1",
                "checks": integrity,
                "models": model_rows,
            },
        )
        metrics, decision, rate_compute = evaluate_stagea_v3(
            config,
            bundles=all_bundles,
            panels=test_panels,
            train_identity_basis=identity_train_basis,
            integrity_checks=integrity,
        )
        atomic_write_json(root / "metrics.json", metrics)
        atomic_write_json(root / "decision.json", decision)
        atomic_write_json(root / "rate_compute.json", rate_compute)

        required_now = [
            compiler_manifest,
            training_summary_path,
            baseline_manifest,
            prediction_manifest,
            models_path,
            capture_validation,
            root / "metrics.json",
            root / "decision.json",
            root / "rate_compute.json",
        ]
        all_artifact_paths.extend(required_now)
        status.update(
            {
                "state": "completed",
                "current_stage": None,
                "completed_stages": list(STAGEA_V3_STAGE_ORDER),
                "completed_at": _timestamp(),
                "elapsed_seconds": time.time() - started,
                "scientific_decision": decision,
            }
        )
        atomic_write_json(root / "workflow_status.json", status)
        telemetry.log(
            {
                "decision": {
                    "status": decision["status"],
                    "diagnosis": decision["diagnosis"],
                }
            }
        )
        run_summary = {
            "schema": "frank_eq_stagea_v3_run_summary_v1",
            "status": "completed",
            "workflow_integrity_passed": all(integrity.values()),
            "root": str(root),
            "protocol_version": config.protocol_version,
            "decision": decision,
            "representation_only": True,
            "telemetry": telemetry.status(),
            "elapsed_seconds": time.time() - started,
        }
        atomic_write_json(root / "run_summary.json", run_summary)
        all_artifact_paths.append(root / "run_summary.json")
        artifact_manifest = {
            "schema": "frank_eq_stagea_v3_artifact_manifest_v1",
            "files": _artifact_hashes(
                root,
                list(
                    {path.resolve(): path for path in all_artifact_paths if path.is_file()}.values()
                ),
            ),
        }
        atomic_write_json(root / "artifact_manifest.json", artifact_manifest)

        from .verify import verify_stagea_v3_run

        audit = verify_stagea_v3_run(
            root,
            config_path=config_source,
            write_audit=True,
            require_existing_audit=False,
        )
        if not audit["passed"]:
            raise RuntimeError("independent Stage-A v3 audit failed")
        artifact_manifest["files"]["independent_audit.json"] = sha256_file(
            root / "independent_audit.json"
        )
        atomic_write_json(root / "artifact_manifest.json", artifact_manifest)
        run_summary["independent_audit"] = audit
        run_summary["artifact_manifest"] = "artifact_manifest.json"
        atomic_write_json(root / "run_summary.json", run_summary)
        artifact_manifest["files"]["run_summary.json"] = sha256_file(root / "run_summary.json")
        atomic_write_json(root / "artifact_manifest.json", artifact_manifest)
        return run_summary
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
                "test_access_consumed": controller.read().get("test_access_count") == 1,
            }
        )
        atomic_write_json(root / "workflow_status.json", status)
        raise
    finally:
        telemetry.finish()
