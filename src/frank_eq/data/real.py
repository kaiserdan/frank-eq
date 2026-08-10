"""Real-checkpoint Stage-A bundle construction, serialization, and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from frank_eq.config import DataConfig
from frank_eq.contracts import (
    FutureBranchRecord,
    FutureSignatureRecord,
    StateCaptureRecord,
    validate_world_split_integrity,
)
from frank_eq.real_config import RealRunConfig
from frank_eq.schemas import OperationDefinition, SplitManifest
from frank_eq.utils import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file

from .hf_backend import HFModelAdapter
from .real_panel import RealPanel, generate_real_panel
from .split import build_split_manifest, validate_split_manifest


@dataclass(slots=True)
class RealBundle:
    """Trainer-compatible arrays plus claim-boundary metadata for real checkpoints."""

    world_ids: np.ndarray
    model_ids: np.ndarray
    renderer_ids: np.ndarray
    hidden: np.ndarray
    hidden_mask: np.ndarray
    facts: np.ndarray
    residual: np.ndarray
    signatures: np.ndarray
    model_signatures: np.ndarray
    operation_descriptors: np.ndarray
    operations: list[OperationDefinition]
    split: SplitManifest
    model_hidden_dims: list[int]
    n_layers: int
    model_names: list[str]
    task_family: str = "relational_graph_future_operations_v1"
    scope: str = "real frozen-LLM future-defined causal-state Stage A"

    @property
    def n_views(self) -> int:
        return int(self.world_ids.shape[0])

    @property
    def max_hidden_dim(self) -> int:
        return int(self.hidden.shape[-1])

    def indices_for(
        self,
        *,
        world_ids: tuple[int, ...] | list[int] | np.ndarray,
        model_ids: tuple[int, ...] | list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        mask = np.isin(self.world_ids, np.asarray(world_ids, dtype=np.int64))
        if model_ids is not None:
            mask &= np.isin(self.model_ids, np.asarray(model_ids, dtype=np.int64))
        return np.flatnonzero(mask)

    def save(
        self,
        directory: str | Path,
        *,
        panel: RealPanel,
        records: list[FutureSignatureRecord],
        extraction_metadata: dict[str, Any],
        real_config: RealRunConfig,
    ) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "dataset.npz",
            world_ids=self.world_ids,
            model_ids=self.model_ids,
            renderer_ids=self.renderer_ids,
            hidden=self.hidden,
            hidden_mask=self.hidden_mask,
            facts=self.facts,
            residual=self.residual,
            signatures=self.signatures,
            model_signatures=self.model_signatures,
            operation_descriptors=self.operation_descriptors,
        )
        atomic_write_json(target / "panel.json", panel.to_dict())
        with (target / "future_signature_records.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n")
        metadata = {
            "schema": "frank_eq_real_bundle_v1",
            "task_family": self.task_family,
            "scope": self.scope,
            "model_hidden_dims": self.model_hidden_dims,
            "model_names": self.model_names,
            "n_layers": self.n_layers,
            "operations": [operation.to_dict() for operation in self.operations],
            "split": self.split.to_dict(),
            "real_config": real_config.as_dict(),
            "extraction": extraction_metadata,
            "panel_sha256": sha256_file(target / "panel.json"),
            "records_sha256": sha256_file(target / "future_signature_records.jsonl"),
            "dataset_sha256": sha256_file(target / "dataset.npz"),
        }
        atomic_write_json(target / "metadata.json", metadata)

    @classmethod
    def load(cls, directory: str | Path) -> "RealBundle":
        source = Path(directory)
        required = ("dataset.npz", "metadata.json", "panel.json", "future_signature_records.jsonl")
        missing = [name for name in required if not (source / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete real bundle {source}; missing {missing}")
        arrays = np.load(source / "dataset.npz")
        metadata = json.loads((source / "metadata.json").read_text())
        if metadata.get("schema") != "frank_eq_real_bundle_v1":
            raise ValueError("unsupported real-bundle schema")
        return cls(
            world_ids=arrays["world_ids"],
            model_ids=arrays["model_ids"],
            renderer_ids=arrays["renderer_ids"],
            hidden=arrays["hidden"],
            hidden_mask=arrays["hidden_mask"],
            facts=arrays["facts"],
            residual=arrays["residual"],
            signatures=arrays["signatures"],
            model_signatures=arrays["model_signatures"],
            operation_descriptors=arrays["operation_descriptors"],
            operations=[OperationDefinition.from_dict(item) for item in metadata["operations"]],
            split=SplitManifest.from_dict(metadata["split"]),
            model_hidden_dims=[int(value) for value in metadata["model_hidden_dims"]],
            n_layers=int(metadata["n_layers"]),
            model_names=[str(value) for value in metadata["model_names"]],
            task_family=str(metadata["task_family"]),
            scope=str(metadata["scope"]),
        )


def _record_from_dict(payload: dict[str, Any]) -> FutureSignatureRecord:
    capture_payload = payload["capture"]
    capture = StateCaptureRecord(
        state_id=str(capture_payload["state_id"]),
        world_id=str(capture_payload["world_id"]),
        model_id=str(capture_payload["model_id"]),
        renderer_id=str(capture_payload["renderer_id"]),
        split=str(capture_payload["split"]),
        prefix_sha256=str(capture_payload["prefix_sha256"]),
        hidden_artifact_sha256=str(capture_payload["hidden_artifact_sha256"]),
        captured_before_operation=bool(capture_payload["captured_before_operation"]),
        capture_step=int(capture_payload["capture_step"]),
    )
    branches = tuple(
        FutureBranchRecord(
            state_id=str(item["state_id"]),
            operation_id=str(item["operation_id"]),
            operation_descriptor_sha256=str(item["operation_descriptor_sha256"]),
            outcome_probabilities=tuple(float(value) for value in item["outcome_probabilities"]),
            branch_seed=int(item["branch_seed"]),
            operation_reveal_step=int(item["operation_reveal_step"]),
        )
        for item in payload["branches"]
    )
    return FutureSignatureRecord(capture=capture, branches=branches)


def load_future_signature_records(directory: str | Path) -> list[FutureSignatureRecord]:
    path = Path(directory) / "future_signature_records.jsonl"
    return [
        _record_from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line
    ]


def _split_by_world(split: SplitManifest) -> dict[int, str]:
    result: dict[int, str] = {}
    for name, ids in (
        ("train", split.train_world_ids),
        ("validation", split.validation_world_ids),
        ("test", split.test_world_ids),
    ):
        for world_id in ids:
            result[int(world_id)] = name
    return result


def build_real_cache(config: RealRunConfig, output_dir: str | Path) -> dict[str, Any]:
    """Extract all founder and held-sender views without training a quotient model."""

    config.validate()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    panel = generate_real_panel(config.panel)
    operations = [row.definition for row in panel.operations]
    placeholder_data = DataConfig(
        n_worlds=config.panel.n_worlds,
        n_founder_models=config.founder_count,
        include_held_model=True,
        n_renderers=config.panel.n_renderers,
        n_layers=len(config.capture.normalized_depths),
        model_hidden_dims=[1] * len(config.models),
        n_facts=config.panel.n_facts,
        n_residual=config.panel.n_residual,
        n_operations=config.panel.n_operations,
        operation_holdout_fraction=config.panel.operation_holdout_fraction,
        train_fraction=config.panel.train_fraction,
        validation_fraction=config.panel.validation_fraction,
        seed=config.panel.seed,
    )
    split = build_split_manifest(placeholder_data, operations)
    split_by_world = _split_by_world(split)

    captured_by_model = []
    extraction_models: list[dict[str, Any]] = []
    all_records: list[FutureSignatureRecord] = []
    for model_index, spec in enumerate(config.models):
        adapter = HFModelAdapter(spec, config.capture)
        captured = adapter.capture_panel(panel, split_by_world)
        captured_by_model.append(captured)
        all_records.extend(captured.records)
        extraction_models.append(
            {
                "model_index": model_index,
                "model_id": spec.model_id,
                "hf_id": spec.hf_id,
                "role": spec.role,
                "revision_requested": spec.revision,
                "revision_observed": captured.model_revision,
                "hidden_dim": captured.hidden_dim,
                "layer_indices": captured.layer_indices,
                "normalized_depths": config.capture.normalized_depths,
                "answer_labels": list(captured.answer_labels),
                "branch_mode_counts": captured.branch_mode_counts,
            }
        )
        del adapter
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    model_hidden_dims = [captured.hidden_dim for captured in captured_by_model]
    max_hidden = max(model_hidden_dims)
    total_views = sum(len(captured.world_ids) for captured in captured_by_model)
    n_layers = len(config.capture.normalized_depths)
    hidden = np.zeros((total_views, n_layers, max_hidden), dtype=np.float32)
    hidden_mask = np.zeros_like(hidden, dtype=np.bool_)
    world_ids = np.empty(total_views, dtype=np.int64)
    model_ids = np.empty(total_views, dtype=np.int64)
    renderer_ids = np.empty(total_views, dtype=np.int64)
    facts = np.empty((total_views, config.panel.n_facts), dtype=np.float32)
    residual = np.empty((total_views, config.panel.n_residual), dtype=np.float32)
    signatures = np.empty((total_views, config.panel.n_operations), dtype=np.float32)
    model_signatures = np.empty_like(signatures)
    oracle = np.asarray(panel.oracle_signatures, dtype=np.float32)
    facts_by_world = np.stack([world.fact_vector() for world in panel.worlds], axis=0)
    residual_by_world = np.stack([world.residual_vector() for world in panel.worlds], axis=0)

    cursor = 0
    for model_index, captured in enumerate(captured_by_model):
        count = len(captured.world_ids)
        width = captured.hidden_dim
        selection = slice(cursor, cursor + count)
        hidden[selection, :, :width] = captured.hidden
        hidden_mask[selection, :, :width] = True
        world_ids[selection] = captured.world_ids
        model_ids[selection] = model_index
        renderer_ids[selection] = captured.renderer_ids
        facts[selection] = facts_by_world[captured.world_ids]
        residual[selection] = residual_by_world[captured.world_ids]
        signatures[selection] = oracle[captured.world_ids]
        model_signatures[selection] = captured.branch_signatures
        cursor += count

    bundle = RealBundle(
        world_ids=world_ids,
        model_ids=model_ids,
        renderer_ids=renderer_ids,
        hidden=hidden,
        hidden_mask=hidden_mask,
        facts=facts,
        residual=residual,
        signatures=signatures,
        model_signatures=model_signatures,
        operation_descriptors=np.stack(
            [np.asarray(row.descriptor, dtype=np.float32) for row in panel.operations], axis=0
        ),
        operations=operations,
        split=split,
        model_hidden_dims=model_hidden_dims,
        n_layers=n_layers,
        model_names=[spec.model_id for spec in config.models],
    )
    extraction_metadata = {
        "schema": "frank_eq_real_extraction_v1",
        "models": extraction_models,
        "branch_contract": {
            "capture_before_operation": True,
            "configured_mode": config.capture.branch_mode,
            "exact_replay_fallback_allowed": config.capture.allow_exact_replay_fallback,
        },
    }
    bundle.save(
        target,
        panel=panel,
        records=all_records,
        extraction_metadata=extraction_metadata,
        real_config=config,
    )
    validation = validate_real_cache(target)
    summary = {
        "schema": "frank_eq_real_cache_build_v1",
        "status": "passed",
        "views": bundle.n_views,
        "worlds": config.panel.n_worlds,
        "models": len(config.models),
        "renderers": config.panel.n_renderers,
        "operations": config.panel.n_operations,
        "model_hidden_dims": model_hidden_dims,
        "cache_dir": str(target),
        "validation": validation,
    }
    atomic_write_json(target / "build_summary.json", summary)
    return summary


def validate_real_cache(directory: str | Path) -> dict[str, Any]:
    """Fail closed on causal ordering, hashes, coverage, and grouped splits."""

    source = Path(directory)
    bundle = RealBundle.load(source)
    panel = RealPanel.from_dict(json.loads((source / "panel.json").read_text()))
    records = load_future_signature_records(source)
    required_operations = {str(operation.definition.operation_id) for operation in panel.operations}
    for record in records:
        record.validate(required_operations)
    validate_world_split_integrity(records)
    validate_split_manifest(bundle.split, len(panel.worlds), len(panel.operations))

    if len(records) != bundle.n_views:
        raise ValueError("record count does not match bundle views")
    descriptor_hash_by_id = {
        str(row.definition.operation_id): row.descriptor_sha256 for row in panel.operations
    }
    expected_coverage = {
        (model_name, str(world.world_id), str(renderer_id))
        for model_name in bundle.model_names
        for world in panel.worlds
        for renderer_id in range(int(panel.config["n_renderers"]))
    }
    observed_coverage: set[tuple[str, str, str]] = set()
    branch_brier: list[float] = []
    branch_accuracy: list[float] = []
    for row, record in enumerate(records):
        model_index = int(bundle.model_ids[row])
        width = bundle.model_hidden_dims[model_index]
        hidden_digest = sha256_bytes(
            np.asarray(bundle.hidden[row, :, :width], dtype=np.float32).tobytes()
        )
        if hidden_digest != record.capture.hidden_artifact_sha256:
            raise ValueError(f"hidden artifact hash mismatch at row {row}")
        if record.capture.model_id != bundle.model_names[model_index]:
            raise ValueError("record model ID does not match array model index")
        if int(record.capture.world_id) != int(bundle.world_ids[row]):
            raise ValueError("record world ID does not match array row")
        if int(record.capture.renderer_id) != int(bundle.renderer_ids[row]):
            raise ValueError("record renderer ID does not match array row")
        observed_coverage.add(
            (record.capture.model_id, record.capture.world_id, record.capture.renderer_id)
        )
        branch_true = np.asarray(
            [branch.outcome_probabilities[1] for branch in record.branches], dtype=np.float64
        )
        if not np.allclose(branch_true, bundle.model_signatures[row], atol=1e-7):
            raise ValueError("record branch probabilities do not match model_signatures")
        for branch in record.branches:
            if branch.operation_descriptor_sha256 != descriptor_hash_by_id[branch.operation_id]:
                raise ValueError("branch descriptor hash does not match frozen operation registry")
        truth = bundle.signatures[row]
        branch_brier.append(float(np.mean((branch_true - truth) ** 2)))
        branch_accuracy.append(float(np.mean((branch_true >= 0.5) == (truth >= 0.5))))

    if observed_coverage != expected_coverage:
        missing = sorted(expected_coverage - observed_coverage)[:10]
        extra = sorted(observed_coverage - expected_coverage)[:10]
        raise ValueError(f"incomplete model/world/renderer coverage; missing={missing}, extra={extra}")
    metadata = json.loads((source / "metadata.json").read_text())
    for file_name, key in (
        ("panel.json", "panel_sha256"),
        ("future_signature_records.jsonl", "records_sha256"),
        ("dataset.npz", "dataset_sha256"),
    ):
        if sha256_file(source / file_name) != metadata[key]:
            raise ValueError(f"cache file hash mismatch: {file_name}")

    operation_balance = np.asarray(panel.oracle_signatures, dtype=np.float64) >= 0.5
    positive_fractions = operation_balance.mean(axis=0)
    result = {
        "schema": "frank_eq_real_cache_validation_v1",
        "status": "passed",
        "causal_boundary_passed": True,
        "world_split_integrity_passed": True,
        "descriptor_hashes_passed": True,
        "hidden_hashes_passed": True,
        "coverage_passed": True,
        "views": bundle.n_views,
        "records": len(records),
        "mean_model_branch_brier_to_oracle": float(np.mean(branch_brier)),
        "mean_model_branch_accuracy_to_oracle": float(np.mean(branch_accuracy)),
        "operation_positive_fraction_min": float(positive_fractions.min()),
        "operation_positive_fraction_max": float(positive_fractions.max()),
        "cache_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "dataset": metadata["dataset_sha256"],
                    "panel": metadata["panel_sha256"],
                    "records": metadata["records_sha256"],
                }
            )
        ),
        "authorizes_training": True,
        "authorizes_scientific_claim": False,
    }
    atomic_write_json(source / "cache_validation.json", result)
    return result
