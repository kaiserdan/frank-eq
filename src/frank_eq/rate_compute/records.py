"""Paired frozen-model record capture and train-only calibration."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from frank_eq.data.real_panel import (
    RealPanel,
    edge_fact_index,
    render_operation_query,
    render_world_prefix,
)
from frank_eq.telemetry import WandbTelemetry

from .backend import (
    RateComputeModelAdapter,
    render_deliberation_query,
    render_sequence_query,
)
from .calibration import fit_platt_calibrator
from .config import RateComputeRunConfig
from .logic import operation_support_size, ordered_edges, render_basis_query


def global_world_id(n_entities: int, panel_world_id: int) -> int:
    return n_entities * 1_000_000 + panel_world_id


def _basis_query(
    source: int,
    target: int,
    n_entities: int,
    config: RateComputeRunConfig,
) -> str:
    false_display = config.protocols.candidate_false.strip()
    true_display = config.protocols.candidate_true.strip()
    cue = (
        f" Reply with exactly {false_display} for false or {true_display} for true."
        f"{config.protocols.sequence_cue}"
    )
    return render_basis_query(source, target, n_entities, final_cue=cue)


def capture_records(
    config: RateComputeRunConfig,
    panels: dict[int, RealPanel],
    splits: dict[int, dict[int, str]],
    telemetry: WandbTelemetry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Capture all elementary-basis and target-protocol branches."""

    records: list[dict[str, Any]] = []
    model_metadata: list[dict[str, Any]] = []
    for model_index, spec in enumerate(config.models):
        adapter = RateComputeModelAdapter(spec, config.capture)
        observed_revision = getattr(adapter.model.config, "_commit_hash", None) or spec.revision
        model_rows_before = len(records)
        for n_entities, panel in panels.items():
            split = splits[n_entities]
            smooth = config.panel.oracle_smoothing
            for world in panel.worlds:
                public_world_id = global_world_id(n_entities, world.world_id)
                target_truths = np.asarray(
                    panel.oracle_signatures[world.world_id], dtype=np.float64
                )
                edge = world.edge_array()
                for renderer_id in range(config.panel.n_renderers):
                    world_statement = render_world_prefix(world, renderer_id)
                    prefix_text = adapter._format_prefix(world_statement)
                    prefix_ids = adapter._tokenize(prefix_text)
                    with torch.inference_mode():
                        prefix_output = adapter.model(
                            input_ids=prefix_ids,
                            use_cache=True,
                            return_dict=True,
                        )
                    if prefix_output.past_key_values is None:
                        raise RuntimeError(f"{spec.model_id} did not return a KV cache")

                    for source, target in ordered_edges(n_entities):
                        query = _basis_query(source, target, n_entities, config)
                        query_ids = adapter._query_ids(
                            query,
                            world_statement=world_statement,
                            prefix_ids=prefix_ids,
                        )
                        if config.protocols.basis_protocol == "sequence":
                            score = adapter.score_sequence(
                                prefix_ids,
                                prefix_output.past_key_values,
                                query_ids,
                                config.protocols,
                            )
                        else:
                            score = adapter.score_with_compute(
                                prefix_ids,
                                prefix_output.past_key_values,
                                query_ids,
                                config.protocols,
                                mode=config.protocols.basis_protocol,
                            )
                        truth_hard = int(edge[source, target])
                        truth = float(smooth + truth_hard * (1.0 - 2.0 * smooth))
                        records.append(
                            {
                                "schema": "frank_eq_rate_compute_record_v1",
                                "world_id": public_world_id,
                                "panel_world_id": world.world_id,
                                "entity_count": n_entities,
                                "split": split[world.world_id],
                                "model_id": spec.model_id,
                                "model_index": model_index,
                                "renderer_id": renderer_id,
                                "kind": "basis",
                                "family": "edge",
                                "item_id": edge_fact_index(n_entities, source, target),
                                "source": source,
                                "target": target,
                                "protocol": config.protocols.basis_protocol,
                                "truth": truth,
                                "truth_hard": truth_hard,
                                **score.to_dict(),
                            }
                        )

                    for operation_index, frozen_operation in enumerate(panel.operations):
                        operation = frozen_operation.definition
                        truth = float(target_truths[operation_index])
                        truth_hard = int(truth >= 0.5)
                        for protocol in config.protocols.target_protocols:
                            if (
                                protocol in {"reason", "pause"}
                                and operation.family not in config.protocols.compute_families
                            ):
                                continue
                            if protocol == "answer_token":
                                query = render_operation_query(
                                    operation,
                                    n_entities,
                                    adapter.answer_labels[0],
                                    adapter.answer_labels[1],
                                )
                                query_ids = adapter._query_ids(
                                    query,
                                    world_statement=world_statement,
                                    prefix_ids=prefix_ids,
                                )
                                score = adapter.score_answer_token(
                                    prefix_ids,
                                    prefix_output.past_key_values,
                                    query_ids,
                                )
                            elif protocol == "sequence":
                                query = render_sequence_query(
                                    operation, n_entities, config.protocols
                                )
                                query_ids = adapter._query_ids(
                                    query,
                                    world_statement=world_statement,
                                    prefix_ids=prefix_ids,
                                )
                                score = adapter.score_sequence(
                                    prefix_ids,
                                    prefix_output.past_key_values,
                                    query_ids,
                                    config.protocols,
                                )
                            elif protocol in {"reason", "pause"}:
                                query = render_deliberation_query(
                                    operation, n_entities, config.protocols
                                )
                                query_ids = adapter._query_ids(
                                    query,
                                    world_statement=world_statement,
                                    prefix_ids=prefix_ids,
                                )
                                score = adapter.score_with_compute(
                                    prefix_ids,
                                    prefix_output.past_key_values,
                                    query_ids,
                                    config.protocols,
                                    mode=protocol,
                                )
                            else:
                                raise ValueError(f"unsupported protocol: {protocol}")
                            records.append(
                                {
                                    "schema": "frank_eq_rate_compute_record_v1",
                                    "world_id": public_world_id,
                                    "panel_world_id": world.world_id,
                                    "entity_count": n_entities,
                                    "split": split[world.world_id],
                                    "model_id": spec.model_id,
                                    "model_index": model_index,
                                    "renderer_id": renderer_id,
                                    "kind": "target",
                                    "family": operation.family,
                                    "item_id": operation.operation_id,
                                    "operation_id": operation.operation_id,
                                    "polarity": float(operation.polarity),
                                    "fact_args": list(operation.fact_args),
                                    "residual_args": list(operation.residual_args),
                                    "structural_support_size": operation_support_size(
                                        operation, n_entities
                                    ),
                                    "protocol": protocol,
                                    "truth": truth,
                                    "truth_hard": truth_hard,
                                    **score.to_dict(),
                                }
                            )
        model_metadata.append(
            {
                "model_index": model_index,
                "model_id": spec.model_id,
                "hf_id": spec.hf_id,
                "revision_requested": spec.revision,
                "revision_observed": observed_revision,
                "answer_labels": list(adapter.answer_labels),
                "records": len(records) - model_rows_before,
            }
        )
        telemetry.log(
            {
                "capture": {
                    "model_id": spec.model_id,
                    "records": len(records) - model_rows_before,
                }
            }
        )
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records, model_metadata


def _calibration_key(row: dict[str, Any]) -> tuple[str, int, str, str, str]:
    return (
        str(row["model_id"]),
        int(row["entity_count"]),
        str(row["kind"]),
        str(row["family"]),
        str(row["protocol"]),
    )


def calibrate_records(
    records: list[dict[str, Any]], config: RateComputeRunConfig
) -> dict[str, Any]:
    """Fit model-local affine readout calibration on training worlds only."""

    groups: dict[tuple[str, int, str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[_calibration_key(row)].append(index)
    artifact: dict[str, Any] = {
        "schema": "frank_eq_rate_compute_calibration_v1",
        "fit_split": "train",
        "groups": {},
    }
    for key, indices in sorted(groups.items()):
        train_indices = [index for index in indices if records[index]["split"] == "train"]
        if len(train_indices) < 2:
            raise RuntimeError(f"calibration group {key} has fewer than two training rows")
        calibrator = fit_platt_calibrator(
            np.asarray([records[index]["log_odds_score"] for index in train_indices]),
            np.asarray([records[index]["truth"] for index in train_indices]),
            l2=config.evaluation.calibration_l2,
            max_steps=config.evaluation.calibration_max_steps,
        )
        group_scores = np.asarray([records[index]["log_odds_score"] for index in indices])
        for index, probability in zip(indices, calibrator.predict(group_scores), strict=True):
            records[index]["calibrated_probability"] = float(
                np.clip(probability, 1e-7, 1.0 - 1e-7)
            )
        artifact["groups"]["|".join(map(str, key))] = {
            "rows": len(indices),
            "train_rows": len(train_indices),
            "calibrator": calibrator.to_dict(),
        }

    prior_groups: dict[tuple[int, str, str, int], list[float]] = defaultdict(list)
    for row in records:
        if row["split"] == "train":
            key = (
                int(row["entity_count"]),
                str(row["kind"]),
                str(row["family"]),
                int(row["item_id"]),
            )
            prior_groups[key].append(float(row["truth"]))
    priors = {
        key: float(np.clip(np.mean(values), 1e-6, 1.0 - 1e-6))
        for key, values in prior_groups.items()
    }
    for row in records:
        key = (
            int(row["entity_count"]),
            str(row["kind"]),
            str(row["family"]),
            int(row["item_id"]),
        )
        row["prior_probability"] = priors[key]
    artifact["operation_priors"] = {
        "|".join(map(str, key)): value for key, value in priors.items()
    }
    return artifact
