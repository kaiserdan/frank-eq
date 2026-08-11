"""Complete registered prediction matrix for one Stage-A v3 test shard."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from frank_eq.rate_compute.calibration import sigmoid
from frank_eq.utils import atomic_write_json, sha256_file

from .baselines import parse_v3_world_prefix
from .capture import V3CaptureShard
from .config import StageAV3Config
from .packet import (
    encode_rate_matched_text_basis,
    encode_typed_edge_packet,
    execute_typed_basis,
    panel_control_thresholds,
)
from .panel import V3Panel

_SEMANTIC_BASIS_CONDITIONS = {
    "primary_float",
    "primary_q1",
    "primary_q2",
    "primary_q4",
    "primary_q8",
    "train_edge_prior",
    "token_id_q4",
    "final_token_q4",
    "interactive_basis",
    "deterministic_text_parser",
    "rate_matched_canonical_text",
    "oracle_basis",
    "shuffled_world_packet",
    "wrong_world_packet",
    "zero_packet",
}
_BEHAVIORAL_BASIS_CONDITIONS = {
    "primary_float",
    "primary_q1",
    "primary_q2",
    "primary_q4",
    "primary_q8",
    "train_edge_prior",
}
_EXTRA_OPERATION_CONDITIONS = {
    "historical_continuous_quotient",
    "train_selected_direct_protocol",
    "train_operation_prior",
}


@dataclass(slots=True)
class V3PredictionBundle:
    model_id: str
    entity_count: int
    world_ids: np.ndarray
    renderer_ids: np.ndarray
    semantic_truth: np.ndarray
    behavioral_truth: np.ndarray
    operation_truth: np.ndarray
    operation_truth_hard: np.ndarray
    semantic_basis: dict[str, np.ndarray]
    behavioral_basis: dict[str, np.ndarray]
    operations: dict[str, np.ndarray]
    semantic_seed_probabilities: np.ndarray
    behavioral_seed_probabilities: np.ndarray
    token_seed_probabilities: np.ndarray
    final_token_seed_probabilities: np.ndarray
    continuous_seed_probabilities: np.ndarray
    direct_generated_tokens: np.ndarray
    direct_protocols: tuple[str, ...]
    packet_records: list[dict[str, Any]]
    compute: dict[str, Any]
    schema: str = "frank_eq_stagea_v3_prediction_bundle_v1"

    @property
    def rows(self) -> int:
        return int(self.world_ids.shape[0])

    @property
    def coordinate_count(self) -> int:
        return self.entity_count * (self.entity_count - 1)

    def validate(self) -> None:
        if self.schema != "frank_eq_stagea_v3_prediction_bundle_v1":
            raise ValueError("unsupported Stage-A v3 prediction-bundle schema")
        if self.entity_count not in {4, 6}:
            raise ValueError("prediction bundle entity count is outside registration")
        rows = self.rows
        coordinates = self.coordinate_count
        if self.renderer_ids.shape != (rows,):
            raise ValueError("prediction bundle renderer IDs have the wrong shape")
        if self.semantic_truth.shape != (rows, coordinates):
            raise ValueError("prediction bundle semantic truth has the wrong shape")
        if self.behavioral_truth.shape != (rows, coordinates):
            raise ValueError("prediction bundle behavioral truth has the wrong shape")
        if self.operation_truth.ndim != 2 or self.operation_truth.shape[0] != rows:
            raise ValueError("prediction bundle operation truth has the wrong shape")
        if self.operation_truth_hard.shape != self.operation_truth.shape:
            raise ValueError("prediction bundle hard-operation truth has the wrong shape")
        if set(self.semantic_basis) != _SEMANTIC_BASIS_CONDITIONS:
            raise ValueError("semantic prediction bundle omits or adds a frozen condition")
        if set(self.behavioral_basis) != _BEHAVIORAL_BASIS_CONDITIONS:
            raise ValueError("behavioral prediction bundle omits or adds a frozen condition")
        if set(self.operations) != _SEMANTIC_BASIS_CONDITIONS | _EXTRA_OPERATION_CONDITIONS:
            raise ValueError("operation prediction bundle omits or adds a frozen condition")
        for values in self.semantic_basis.values():
            if values.shape != (rows, coordinates):
                raise ValueError("semantic basis condition has the wrong shape")
        for values in self.behavioral_basis.values():
            if values.shape != (rows, coordinates):
                raise ValueError("behavioral basis condition has the wrong shape")
        for values in self.operations.values():
            if values.shape != self.operation_truth.shape:
                raise ValueError("operation condition has the wrong shape")
        seeds = self.semantic_seed_probabilities.shape[0]
        if seeds < 1:
            raise ValueError("prediction bundle has no compiler seed rows")
        expected_seed_basis = (seeds, rows, coordinates)
        for values in (
            self.semantic_seed_probabilities,
            self.behavioral_seed_probabilities,
            self.token_seed_probabilities,
            self.final_token_seed_probabilities,
        ):
            if values.shape != expected_seed_basis:
                raise ValueError("basis seed tensor has the wrong shape")
        if self.continuous_seed_probabilities.shape != (
            seeds,
            *self.operation_truth.shape,
        ):
            raise ValueError("continuous seed tensor has the wrong shape")
        if self.direct_generated_tokens.shape != self.operation_truth.shape:
            raise ValueError("selected direct generated-token matrix has the wrong shape")
        if len(self.direct_protocols) != self.operation_truth.shape[1]:
            raise ValueError("selected direct protocol registry is incomplete")
        for values in (
            self.semantic_truth,
            self.behavioral_truth,
            self.operation_truth,
            *self.semantic_basis.values(),
            *self.behavioral_basis.values(),
            *self.operations.values(),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError("prediction bundle contains non-finite values")
            if np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError("prediction probabilities lie outside [0,1]")


def _packetize_matrix(
    values: np.ndarray,
    *,
    entity_count: int,
    bits: int,
    condition: str,
    world_ids: np.ndarray,
    renderer_ids: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    decoded = np.empty_like(values, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(values):
        packet = encode_typed_edge_packet(row, entity_count=entity_count, bits=bits)
        decoded[index] = packet.probabilities()
        records.append(
            {
                "condition": condition,
                "world_id": int(world_ids[index]),
                "renderer_id": int(renderer_ids[index]),
                "entity_count": entity_count,
                "bits_per_coordinate": bits,
                "payload_bits": packet.payload_bits,
                "framing_bits": packet.framing_bits,
                "serialized_bits": packet.serialized_bits,
                "checksum_sha256": packet.checksum_sha256,
                "payload_hex": packet.payload_hex,
            }
        )
    return decoded, records


def _execute_rows(
    basis: np.ndarray,
    panel: V3Panel,
) -> np.ndarray:
    thresholds = panel_control_thresholds(panel.panel.worlds)
    output = np.empty((basis.shape[0], len(panel.panel.operations)), dtype=np.float64)
    for row_index, probabilities in enumerate(basis):
        for operation in panel.panel.operations:
            output[row_index, operation.definition.operation_id] = execute_typed_basis(
                probabilities,
                operation.definition,
                entity_count=panel.entity_count,
                control_thresholds=thresholds,
            )
    return output


def _shuffled_rows(
    values: np.ndarray,
    world_ids: np.ndarray,
    renderer_ids: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    output = np.empty_like(values)
    rng = np.random.default_rng(seed)
    for renderer_id in sorted(set(int(value) for value in renderer_ids.tolist())):
        indices = np.flatnonzero(renderer_ids == renderer_id)
        if len(indices) < 2:
            raise ValueError("shuffled-world control requires at least two worlds per renderer")
        order = rng.permutation(indices)
        sources = np.roll(order, 1)
        if np.any(world_ids[order] == world_ids[sources]):
            raise RuntimeError("shuffled-world control did not produce a derangement")
        output[order] = values[sources]
    return output


def _hardest_wrong_rows(
    values: np.ndarray,
    world_ids: np.ndarray,
    renderer_ids: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(values)
    normalized = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    for index in range(len(values)):
        candidates = np.flatnonzero(
            (renderer_ids == renderer_ids[index]) & (world_ids != world_ids[index])
        )
        if not len(candidates):
            raise ValueError("wrong-world control requires another world in the renderer stratum")
        similarities = normalized[candidates] @ normalized[index]
        output[index] = values[candidates[int(np.argmax(similarities))]]
    return output


def _as_probability_seed_tensor(logits: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    values = np.stack([sigmoid(np.asarray(row, dtype=np.float64)) for row in logits])
    if values.shape[1:] != shape:
        raise ValueError("compiler seed prediction tensor has the wrong shape")
    return values


def assemble_prediction_bundle(
    config: StageAV3Config,
    *,
    shard: V3CaptureShard,
    panel: V3Panel,
    semantic_primary: np.ndarray,
    behavioral_primary: np.ndarray,
    token_primary: np.ndarray,
    final_token_primary: np.ndarray,
    continuous_primary: np.ndarray,
    semantic_seed_logits: list[np.ndarray],
    behavioral_seed_logits: list[np.ndarray],
    token_seed_logits: list[np.ndarray],
    final_token_seed_logits: list[np.ndarray],
    continuous_seed_logits: list[np.ndarray],
    controls: dict[str, np.ndarray | list[str]],
    compiler_compute: dict[str, Any],
) -> V3PredictionBundle:
    """Assemble every frozen method before any metric or gate is reduced."""

    shard.validate()
    panel.validate()
    if shard.role != "test" or panel.role != "test":
        raise ValueError("outcome-bearing prediction bundles require the test role")
    if shard.entity_count != panel.entity_count:
        raise ValueError("prediction shard and panel complexity differ")
    rows = shard.rows
    coordinates = shard.coordinate_count
    basis_shape = (rows, coordinates)
    operation_shape = tuple(shard.operation_targets.shape)
    inputs = (semantic_primary, behavioral_primary, token_primary, final_token_primary)
    if any(np.asarray(value).shape != basis_shape for value in inputs):
        raise ValueError("one or more basis prediction inputs have the wrong shape")
    if np.asarray(continuous_primary).shape != operation_shape:
        raise ValueError("continuous prediction input has the wrong shape")
    bits_frontier = list(config.section("packet")["quantization_frontier_bits"])
    if bits_frontier != [1, 2, 4, 8]:
        raise ValueError("prediction assembly requires the frozen quantization frontier")

    world_ids = shard.world_ids.numpy().astype(np.int64)
    renderer_ids = shard.renderer_ids.numpy().astype(np.int64)
    semantic_float = np.asarray(semantic_primary, dtype=np.float64)
    behavioral_float = np.asarray(behavioral_primary, dtype=np.float64)
    packet_records: list[dict[str, Any]] = []
    semantic_basis: dict[str, np.ndarray] = {"primary_float": semantic_float}
    behavioral_basis: dict[str, np.ndarray] = {"primary_float": behavioral_float}
    for bits in bits_frontier:
        semantic_quantized, records = _packetize_matrix(
            semantic_float,
            entity_count=shard.entity_count,
            bits=bits,
            condition=f"semantic_primary_q{bits}",
            world_ids=world_ids,
            renderer_ids=renderer_ids,
        )
        semantic_basis[f"primary_q{bits}"] = semantic_quantized
        packet_records.extend(records)
        behavioral_quantized, records = _packetize_matrix(
            behavioral_float,
            entity_count=shard.entity_count,
            bits=bits,
            condition=f"behavioral_primary_q{bits}",
            world_ids=world_ids,
            renderer_ids=renderer_ids,
        )
        behavioral_basis[f"primary_q{bits}"] = behavioral_quantized
        packet_records.extend(records)

    token_q4, records = _packetize_matrix(
        np.asarray(token_primary, dtype=np.float64),
        entity_count=shard.entity_count,
        bits=4,
        condition="token_id_q4",
        world_ids=world_ids,
        renderer_ids=renderer_ids,
    )
    packet_records.extend(records)
    final_q4, records = _packetize_matrix(
        np.asarray(final_token_primary, dtype=np.float64),
        entity_count=shard.entity_count,
        bits=4,
        condition="final_token_q4",
        world_ids=world_ids,
        renderer_ids=renderer_ids,
    )
    packet_records.extend(records)

    parsed = np.stack(
        [
            parse_v3_world_prefix(
                bytes.fromhex(metadata["prefix_utf8_hex"]).decode("utf-8"),
                shard.entity_count,
            )
            for metadata in shard.prefix_metadata
        ]
    ).astype(np.float64)
    oracle = shard.semantic_targets.numpy().astype(np.float64)
    if not np.array_equal(parsed, oracle):
        raise RuntimeError("deterministic text parser does not reproduce the test oracle")
    text_q4 = np.empty_like(parsed)
    for index, row in enumerate(parsed):
        packet = encode_rate_matched_text_basis(
            row,
            entity_count=shard.entity_count,
            bits=4,
        )
        text_q4[index] = packet.probabilities()
        packet_records.append(
            {
                "condition": "rate_matched_canonical_text",
                "world_id": int(world_ids[index]),
                "renderer_id": int(renderer_ids[index]),
                "entity_count": shard.entity_count,
                "bits_per_coordinate": 4,
                "payload_bits": packet.payload_bits,
                "framing_bits": packet.framing_bits,
                "serialized_bits": packet.serialized_bits,
                "checksum_sha256": packet.checksum_sha256,
                "payload_hex": packet.payload_hex,
                "source": "deterministic_prefix_text_parse",
            }
        )

    semantic_basis.update(
        {
            "train_edge_prior": np.asarray(controls["semantic_edge_prior"], dtype=np.float64),
            "token_id_q4": token_q4,
            "final_token_q4": final_q4,
            "interactive_basis": np.asarray(controls["interactive_basis"], dtype=np.float64),
            "deterministic_text_parser": parsed,
            "rate_matched_canonical_text": text_q4,
            "oracle_basis": oracle,
            "zero_packet": np.full(basis_shape, 0.5, dtype=np.float64),
        }
    )
    primary_q4 = semantic_basis["primary_q4"]
    semantic_basis["shuffled_world_packet"] = _shuffled_rows(
        primary_q4,
        world_ids,
        renderer_ids,
        seed=int(config.section("evaluation")["bootstrap_seed"]) + shard.entity_count,
    )
    semantic_basis["wrong_world_packet"] = _hardest_wrong_rows(
        primary_q4, world_ids, renderer_ids
    )
    behavioral_basis["train_edge_prior"] = np.asarray(
        controls["behavioral_edge_prior"], dtype=np.float64
    )

    operations = {
        condition: _execute_rows(values, panel)
        for condition, values in semantic_basis.items()
    }
    operations.update(
        {
            "historical_continuous_quotient": np.asarray(
                continuous_primary, dtype=np.float64
            ),
            "train_selected_direct_protocol": np.asarray(
                controls["direct_probability"], dtype=np.float64
            ),
            "train_operation_prior": np.asarray(
                controls["operation_prior"], dtype=np.float64
            ),
        }
    )
    seed_count = len(config.section("compiler")["seeds"])
    semantic_seeds = _as_probability_seed_tensor(semantic_seed_logits, basis_shape)
    behavioral_seeds = _as_probability_seed_tensor(behavioral_seed_logits, basis_shape)
    token_seeds = _as_probability_seed_tensor(token_seed_logits, basis_shape)
    final_seeds = _as_probability_seed_tensor(final_token_seed_logits, basis_shape)
    continuous_seeds = _as_probability_seed_tensor(continuous_seed_logits, operation_shape)
    if any(values.shape[0] != seed_count for values in (
        semantic_seeds,
        behavioral_seeds,
        token_seeds,
        final_seeds,
        continuous_seeds,
    )):
        raise ValueError("prediction bundle does not contain every registered seed")

    compute = {
        **compiler_compute,
        "primary": {
            "prefix_forwards": int(shard.capture_summary["prefix_forwards"]),
            "post_capture_source_queries": 0,
            "generated_tokens": 0,
            "pause_tokens": 0,
            "executor_operations_per_world": len(panel.panel.operations),
        },
        "interactive_basis": {
            "post_capture_source_queries_per_prefix": coordinates,
        },
        "train_selected_direct_protocol": {
            "post_capture_source_queries_per_operation": 1,
            "generated_tokens": int(
                np.asarray(controls["direct_generated_tokens"], dtype=np.int64).sum()
            ),
        },
        "text_parser": {
            "prefix_bytes_read": sum(
                len(bytes.fromhex(row["prefix_utf8_hex"])) for row in shard.prefix_metadata
            ),
            "post_capture_source_queries": 0,
        },
        "amortized_operation_counts": config.section("evaluation")[
            "amortized_operation_counts"
        ],
    }
    bundle = V3PredictionBundle(
        model_id=shard.model_id,
        entity_count=shard.entity_count,
        world_ids=world_ids,
        renderer_ids=renderer_ids,
        semantic_truth=oracle,
        behavioral_truth=shard.behavioral_targets.numpy().astype(np.float64),
        operation_truth=shard.operation_targets.numpy().astype(np.float64),
        operation_truth_hard=shard.operation_targets_hard.numpy().astype(np.int8),
        semantic_basis=semantic_basis,
        behavioral_basis=behavioral_basis,
        operations=operations,
        semantic_seed_probabilities=semantic_seeds,
        behavioral_seed_probabilities=behavioral_seeds,
        token_seed_probabilities=token_seeds,
        final_token_seed_probabilities=final_seeds,
        continuous_seed_probabilities=continuous_seeds,
        direct_generated_tokens=np.asarray(
            controls["direct_generated_tokens"], dtype=np.int64
        ),
        direct_protocols=tuple(str(value) for value in controls["direct_protocols"]),
        packet_records=packet_records,
        compute=compute,
    )
    bundle.validate()
    return bundle


def write_prediction_bundle(path: str | Path, bundle: V3PredictionBundle) -> str:
    bundle.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "world_ids": bundle.world_ids,
        "renderer_ids": bundle.renderer_ids,
        "semantic_truth": bundle.semantic_truth,
        "behavioral_truth": bundle.behavioral_truth,
        "operation_truth": bundle.operation_truth,
        "operation_truth_hard": bundle.operation_truth_hard,
        "semantic_seed_probabilities": bundle.semantic_seed_probabilities,
        "behavioral_seed_probabilities": bundle.behavioral_seed_probabilities,
        "token_seed_probabilities": bundle.token_seed_probabilities,
        "final_token_seed_probabilities": bundle.final_token_seed_probabilities,
        "continuous_seed_probabilities": bundle.continuous_seed_probabilities,
        "direct_generated_tokens": bundle.direct_generated_tokens,
    }
    arrays.update({f"semantic_basis__{key}": value for key, value in bundle.semantic_basis.items()})
    arrays.update(
        {f"behavioral_basis__{key}": value for key, value in bundle.behavioral_basis.items()}
    )
    arrays.update({f"operations__{key}": value for key, value in bundle.operations.items()})
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, target)
    return sha256_file(target)


def write_prediction_bundle_artifacts(
    array_path: str | Path,
    metadata_path: str | Path,
    bundle: V3PredictionBundle,
    *,
    config_sha256: str,
) -> dict[str, Any]:
    arrays = Path(array_path)
    metadata = Path(metadata_path)
    array_sha256 = write_prediction_bundle(arrays, bundle)
    payload = {
        "schema": "frank_eq_stagea_v3_prediction_metadata_v1",
        "config_sha256": config_sha256,
        "bundle_schema": bundle.schema,
        "model_id": bundle.model_id,
        "entity_count": bundle.entity_count,
        "array_file": arrays.name,
        "array_sha256": array_sha256,
        "semantic_basis_conditions": sorted(bundle.semantic_basis),
        "behavioral_basis_conditions": sorted(bundle.behavioral_basis),
        "operation_conditions": sorted(bundle.operations),
        "direct_protocols": list(bundle.direct_protocols),
        "packet_records": bundle.packet_records,
        "compute": bundle.compute,
    }
    atomic_write_json(metadata, payload)
    return {
        "array_path": str(arrays),
        "array_sha256": array_sha256,
        "metadata_path": str(metadata),
        "metadata_sha256": sha256_file(metadata),
    }


def load_prediction_bundle(
    array_path: str | Path,
    metadata_path: str | Path,
    *,
    config_sha256: str,
    expected_array_sha256: str | None = None,
    expected_metadata_sha256: str | None = None,
) -> V3PredictionBundle:
    arrays_path = Path(array_path)
    metadata_source = Path(metadata_path)
    if expected_array_sha256 is not None and sha256_file(arrays_path) != expected_array_sha256:
        raise ValueError("prediction array artifact hash mismatch")
    if (
        expected_metadata_sha256 is not None
        and sha256_file(metadata_source) != expected_metadata_sha256
    ):
        raise ValueError("prediction metadata artifact hash mismatch")
    metadata = json.loads(metadata_source.read_text())
    if metadata.get("schema") != "frank_eq_stagea_v3_prediction_metadata_v1":
        raise ValueError("unsupported prediction metadata schema")
    if metadata.get("config_sha256") != config_sha256:
        raise ValueError("prediction metadata belongs to another frozen config")
    if metadata.get("array_file") != arrays_path.name:
        raise ValueError("prediction metadata names a different array artifact")
    if metadata.get("array_sha256") != sha256_file(arrays_path):
        raise ValueError("prediction metadata does not bind the array artifact")
    with np.load(arrays_path, allow_pickle=False) as loaded:
        values = {key: loaded[key] for key in loaded.files}
    semantic_basis = {
        key.removeprefix("semantic_basis__"): value
        for key, value in values.items()
        if key.startswith("semantic_basis__")
    }
    behavioral_basis = {
        key.removeprefix("behavioral_basis__"): value
        for key, value in values.items()
        if key.startswith("behavioral_basis__")
    }
    operations = {
        key.removeprefix("operations__"): value
        for key, value in values.items()
        if key.startswith("operations__")
    }
    if sorted(semantic_basis) != metadata["semantic_basis_conditions"]:
        raise ValueError("prediction semantic condition registry differs")
    if sorted(behavioral_basis) != metadata["behavioral_basis_conditions"]:
        raise ValueError("prediction behavioral condition registry differs")
    if sorted(operations) != metadata["operation_conditions"]:
        raise ValueError("prediction operation condition registry differs")
    bundle = V3PredictionBundle(
        model_id=str(metadata["model_id"]),
        entity_count=int(metadata["entity_count"]),
        world_ids=values["world_ids"],
        renderer_ids=values["renderer_ids"],
        semantic_truth=values["semantic_truth"],
        behavioral_truth=values["behavioral_truth"],
        operation_truth=values["operation_truth"],
        operation_truth_hard=values["operation_truth_hard"],
        semantic_basis=semantic_basis,
        behavioral_basis=behavioral_basis,
        operations=operations,
        semantic_seed_probabilities=values["semantic_seed_probabilities"],
        behavioral_seed_probabilities=values["behavioral_seed_probabilities"],
        token_seed_probabilities=values["token_seed_probabilities"],
        final_token_seed_probabilities=values["final_token_seed_probabilities"],
        continuous_seed_probabilities=values["continuous_seed_probabilities"],
        direct_generated_tokens=values["direct_generated_tokens"],
        direct_protocols=tuple(str(value) for value in metadata["direct_protocols"]),
        packet_records=list(metadata["packet_records"]),
        compute=dict(metadata["compute"]),
        schema=str(metadata["bundle_schema"]),
    )
    bundle.validate()
    return bundle
