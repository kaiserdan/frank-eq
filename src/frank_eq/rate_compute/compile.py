"""Train-selected direct baselines and deterministic public-basis compilation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from frank_eq.data.real_panel import RealPanel, evaluate_operation

from .calibration import brier_score
from .config import RateComputeRunConfig
from .logic import COMPILED_FAMILIES, edge_vector_to_matrix, execute_public_basis


def quantize_probability_matrix(values: np.ndarray, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.round(clipped * levels) / levels


def select_direct_protocols(
    records: list[dict[str, Any]],
    config: RateComputeRunConfig,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Choose the lowest-Brier direct protocol using training worlds only."""

    selection: dict[tuple[str, int, str], dict[str, Any]] = {}
    for model in config.models:
        for n_entities in config.panel.entity_counts:
            families = sorted(
                {
                    str(row["family"])
                    for row in records
                    if row["kind"] == "target"
                    and row["split"] == "train"
                    and row["model_id"] == model.model_id
                    and row["entity_count"] == n_entities
                    and row["family"] in COMPILED_FAMILIES
                }
            )
            for family in families:
                candidates: list[tuple[float, str, int]] = []
                for protocol in config.protocols.target_protocols:
                    group = [
                        row
                        for row in records
                        if row["kind"] == "target"
                        and row["split"] == "train"
                        and row["model_id"] == model.model_id
                        and row["entity_count"] == n_entities
                        and row["family"] == family
                        and row["protocol"] == protocol
                    ]
                    if not group:
                        continue
                    candidates.append(
                        (
                            brier_score(
                                np.asarray([row["truth"] for row in group]),
                                np.asarray(
                                    [row["calibrated_probability"] for row in group]
                                ),
                            ),
                            protocol,
                            len(group),
                        )
                    )
                if not candidates:
                    raise RuntimeError(
                        f"no direct protocol candidates for {model.model_id}/{n_entities}/{family}"
                    )
                train_brier, protocol, rows = min(
                    candidates, key=lambda item: (item[0], item[1])
                )
                selection[(model.model_id, n_entities, family)] = {
                    "protocol": protocol,
                    "train_brier": float(train_brier),
                    "train_rows": rows,
                    "all_train_briers": {
                        name: float(score) for score, name, _ in candidates
                    },
                }
    return selection


def compile_validation_records(
    records: list[dict[str, Any]],
    panels: dict[int, RealPanel],
    config: RateComputeRunConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compile validation operations from calibrated elementary edge slots."""

    basis_rows = [
        row
        for row in records
        if row["split"] == "validation"
        and row["kind"] == "basis"
        and row["protocol"] == config.protocols.basis_protocol
    ]
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in basis_rows:
        key = (
            str(row["model_id"]),
            int(row["entity_count"]),
            int(row["world_id"]),
            int(row["renderer_id"]),
        )
        grouped[key].append(row)

    basis_map: dict[tuple[str, int, int, int], np.ndarray] = {}
    for key, rows in grouped.items():
        n_entities = key[1]
        expected = n_entities * (n_entities - 1)
        if len(rows) != expected:
            raise RuntimeError(f"incomplete public basis for {key}: {len(rows)} != {expected}")
        vector = np.zeros(expected, dtype=np.float64)
        observed_slots: set[int] = set()
        for row in rows:
            item_id = int(row["item_id"])
            if item_id in observed_slots:
                raise RuntimeError(f"duplicate public-basis slot {item_id} for {key}")
            observed_slots.add(item_id)
            vector[item_id] = float(row["calibrated_probability"])
        basis_map[key] = edge_vector_to_matrix(vector, n_entities)

    selected = select_direct_protocols(records, config)
    target_rows: dict[tuple[str, int, int, int, int, str], dict[str, Any]] = {}
    for row in records:
        if row["split"] != "validation" or row["kind"] != "target":
            continue
        family = str(row["family"])
        if family not in COMPILED_FAMILIES:
            continue
        chosen = selected[(str(row["model_id"]), int(row["entity_count"]), family)]
        if row["protocol"] != chosen["protocol"]:
            continue
        key = (
            str(row["model_id"]),
            int(row["entity_count"]),
            int(row["world_id"]),
            int(row["renderer_id"]),
            int(row["operation_id"]),
            family,
        )
        if key in target_rows:
            raise RuntimeError(f"duplicate selected direct target row for {key}")
        target_rows[key] = row

    result: list[dict[str, Any]] = []
    for target_key, row in sorted(target_rows.items()):
        model_id, n_entities, world_id, renderer_id, operation_id, family = target_key
        basis_key = (model_id, n_entities, world_id, renderer_id)
        if basis_key not in basis_map:
            raise RuntimeError(f"missing public-basis state for {basis_key}")
        panel_world_id = int(row["panel_world_id"])
        operation = panels[n_entities].operations[operation_id].definition
        world = panels[n_entities].worlds[panel_world_id]
        oracle_edge = world.edge_array().astype(np.float64)
        oracle_probability = execute_public_basis(oracle_edge, operation)
        oracle_hard = int(evaluate_operation(world, operation))
        if oracle_hard != int(row["truth_hard"]):
            raise RuntimeError(
                f"public executor disagrees with oracle for n={n_entities}, "
                f"world={panel_world_id}, operation={operation_id}"
            )
        public_edge = basis_map[basis_key]
        payload: dict[str, Any] = {
            "world_id": world_id,
            "panel_world_id": panel_world_id,
            "entity_count": n_entities,
            "model_id": model_id,
            "renderer_id": renderer_id,
            "operation_id": operation_id,
            "family": family,
            "polarity": float(row["polarity"]),
            "structural_support_size": int(row["structural_support_size"]),
            "truth": float(row["truth"]),
            "truth_hard": int(row["truth_hard"]),
            "prior_probability": float(row["prior_probability"]),
            "direct_protocol": str(row["protocol"]),
            "direct_protocol_train_brier": float(
                selected[(model_id, n_entities, family)]["train_brier"]
            ),
            "direct_probability": float(row["calibrated_probability"]),
            "direct_generated_tokens": int(row.get("generated_token_count", 0)),
            "direct_source_queries": 1,
            "basis_source_queries": n_entities * (n_entities - 1),
            "compiled_probability": execute_public_basis(public_edge, operation),
            "oracle_compiled_probability": oracle_probability,
        }
        for bits in config.evaluation.basis_quantization_bits:
            quantized = quantize_probability_matrix(public_edge, bits)
            payload[f"compiled_probability_q{bits}"] = execute_public_basis(
                quantized, operation
            )
            payload[f"basis_rate_bits_q{bits}"] = n_entities * (n_entities - 1) * bits
        result.append(payload)

    artifact = {
        "schema": "frank_eq_direct_protocol_selection_v1",
        "fit_split": "train",
        "groups": {
            "|".join(map(str, key)): value for key, value in sorted(selected.items())
        },
    }
    return result, artifact
