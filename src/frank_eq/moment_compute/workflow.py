"""Development-only Stage-M operation-closed event-basis audit workflow."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shutil
import socket
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from frank_eq.data.real_panel import (
    RealPanel,
    generate_real_panel,
    render_operation_query,
    render_world_prefix,
)
from frank_eq.evaluation.bootstrap import bootstrap_statistic
from frank_eq.rate_compute.backend import (
    ProtocolScore,
    RateComputeModelAdapter,
    render_deliberation_query,
    render_sequence_query,
)
from frank_eq.rate_compute.calibration import balanced_accuracy, brier_score, fit_platt_calibrator
from frank_eq.real_config import RealPanelConfig
from frank_eq.telemetry import WandbTelemetry
from frank_eq.utils import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file

from .config import MomentComputeRunConfig, load_moment_compute_config
from .events import (
    COMPILED_FAMILIES,
    HARD_FAMILIES,
    EventRegistry,
    PublicEvent,
    build_event_registry,
    compile_operation_from_events,
    compile_operation_from_marginals,
    event_truth,
    exact_executor_mismatches,
    project_event_probabilities,
    render_event_query,
)

MOMENT_COMPUTE_ALLOWED_STAGES = ("audit",)
MOMENT_ACCESS_CONTRACT = {
    "state_precedes_operation_reveal": True,
    "literal_kv_reuse": True,
    "exact_replay_fallback": False,
    "roles": ["calibration", "selection", "validation"],
    "claim_bearing_test_worlds_available": False,
    "held_sender_loaded": False,
    "receiver_tensors_available": False,
    "interactive_event_tomography": True,
    "one_shot_interface_claim": False,
}


@dataclass(slots=True)
class _PendingBranch:
    row: dict[str, Any]
    protocol: str
    query_ids: torch.Tensor


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


def parse_moment_compute_stages(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        stages = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        stages = tuple(str(item) for item in value)
    if stages != MOMENT_COMPUTE_ALLOWED_STAGES:
        raise ValueError("Stage M0 permits exactly one development stage: audit")
    return stages


def build_moment_panel(config: MomentComputeRunConfig) -> RealPanel:
    n_entities = config.panel.entity_counts[0]
    panel_config = RealPanelConfig(
        n_worlds=config.panel.worlds_per_complexity,
        n_entities=n_entities,
        n_operations=config.panel.n_target_operations,
        n_renderers=config.panel.n_renderers,
        train_fraction=0.60,
        validation_fraction=0.20,
        operation_holdout_fraction=0.25,
        oracle_smoothing=config.panel.oracle_smoothing,
        min_operation_positive_fraction=config.panel.min_operation_positive_fraction,
        max_operation_positive_fraction=config.panel.max_operation_positive_fraction,
        max_generation_attempts=config.panel.max_generation_attempts,
        seed=config.panel.seed,
    )
    return generate_real_panel(panel_config)


def build_development_roles(config: MomentComputeRunConfig) -> dict[int, str]:
    """Split worlds into calibration, protocol-selection, and frozen validation roles."""

    n_worlds = config.panel.worlds_per_complexity
    rng = np.random.default_rng(config.panel.seed + 17)
    world_ids = np.arange(n_worlds, dtype=np.int64)
    rng.shuffle(world_ids)
    calibration_count = int(round(config.panel.calibration_fraction * n_worlds))
    selection_count = int(round(config.panel.selection_fraction * n_worlds))
    validation_count = n_worlds - calibration_count - selection_count
    if min(calibration_count, selection_count, validation_count) < 4:
        raise ValueError("each Stage M0 development role requires at least four worlds")
    calibration = set(int(value) for value in world_ids[:calibration_count])
    selection = set(
        int(value)
        for value in world_ids[calibration_count : calibration_count + selection_count]
    )
    return {
        world_id: (
            "calibration"
            if world_id in calibration
            else "selection"
            if world_id in selection
            else "validation"
        )
        for world_id in range(n_worlds)
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
                raise ValueError(f"invalid JSONL in {path} at line {line_number}") from error
    return rows


def _score_pending_branches(
    adapter: RateComputeModelAdapter,
    prefix_ids: torch.Tensor,
    prefix_cache: Any,
    pending: list[_PendingBranch],
    config: MomentComputeRunConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    batch_limit = int(config.capture.branch_batch_size)
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, branch in enumerate(pending):
        grouped[(branch.protocol, int(branch.query_ids.shape[1]))].append(index)
    scores: list[ProtocolScore | None] = [None] * len(pending)
    response_batches = 0
    max_batch_size = 0
    for (protocol, _), indices in grouped.items():
        for offset in range(0, len(indices), batch_limit):
            batch_indices = indices[offset : offset + batch_limit]
            query_batch = [pending[index].query_ids for index in batch_indices]
            if protocol == "answer_token":
                batch_scores = adapter.score_answer_token_batch(prefix_ids, prefix_cache, query_batch)
            elif protocol == "sequence":
                batch_scores = adapter.score_sequence_batch(
                    prefix_ids, prefix_cache, query_batch, config.protocols
                )
            elif protocol in {"reason", "pause"}:
                batch_scores = adapter.score_with_compute_batch(
                    prefix_ids,
                    prefix_cache,
                    query_batch,
                    config.protocols,
                    mode=protocol,
                )
            else:
                raise ValueError(f"unsupported Stage M0 protocol: {protocol}")
            if len(batch_scores) != len(batch_indices):
                raise RuntimeError("Stage M0 response batch returned the wrong number of rows")
            for index, score in zip(batch_indices, batch_scores, strict=True):
                scores[index] = score
            response_batches += 1
            max_batch_size = max(max_batch_size, len(batch_indices))
    if any(score is None for score in scores):
        raise RuntimeError("Stage M0 left one or more branches unscored")
    rows = [
        {**branch.row, **score.to_dict()}
        for branch, score in zip(pending, scores, strict=True)
        if score is not None
    ]
    return rows, {
        "response_batches": response_batches,
        "max_observed_batch_size": max_batch_size,
    }


def _event_query(event: PublicEvent, config: MomentComputeRunConfig) -> str:
    false_display = config.protocols.candidate_false.strip()
    true_display = config.protocols.candidate_true.strip()
    final_cue = (
        f" Reply with exactly {false_display} for false or {true_display} for true."
        f"{config.protocols.sequence_cue}"
    )
    return render_event_query(
        event,
        config.panel.entity_counts[0],
        final_cue=final_cue,
    )


def capture_moment_records(
    config: MomentComputeRunConfig,
    panel: RealPanel,
    roles: dict[int, str],
    registry: EventRegistry,
    telemetry: WandbTelemetry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Capture all public-event and direct-operation responses from frozen sources."""

    records: list[dict[str, Any]] = []
    model_metadata: list[dict[str, Any]] = []
    smooth = config.panel.oracle_smoothing
    n_entities = panel.n_entities
    for model_index, spec in enumerate(config.models):
        adapter = RateComputeModelAdapter(spec, config.capture)
        observed_revision = getattr(adapter.model.config, "_commit_hash", None) or spec.revision
        model_start = len(records)
        prefix_count = 0
        response_batches = 0
        max_batch_size = 0
        for world in panel.worlds:
            target_truths = np.asarray(panel.oracle_signatures[world.world_id], dtype=np.float64)
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
                pending: list[_PendingBranch] = []

                for event in registry.events:
                    query = _event_query(event, config)
                    query_ids = adapter._query_ids(
                        query,
                        world_statement=world_statement,
                        prefix_ids=prefix_ids,
                    )
                    truth_hard = int(event_truth(event, world))
                    truth = float(smooth + truth_hard * (1.0 - 2.0 * smooth))
                    pending.append(
                        _PendingBranch(
                            row={
                                "schema": "frank_eq_moment_compute_record_v1",
                                "world_id": int(world.world_id),
                                "entity_count": n_entities,
                                "role": roles[world.world_id],
                                "model_id": spec.model_id,
                                "model_index": model_index,
                                "renderer_id": renderer_id,
                                "kind": "event",
                                "family": event.kind,
                                "event_id": event.event_id,
                                "event_key": event.key,
                                "event_kind": event.kind,
                                "event_order": event.order,
                                "item_id": event.event_id,
                                "protocol": config.protocols.basis_protocol,
                                "truth": truth,
                                "truth_hard": truth_hard,
                            },
                            protocol=config.protocols.basis_protocol,
                            query_ids=query_ids,
                        )
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
                        elif protocol == "sequence":
                            query = render_sequence_query(operation, n_entities, config.protocols)
                        elif protocol in {"reason", "pause"}:
                            query = render_deliberation_query(
                                operation,
                                n_entities,
                                config.protocols,
                            )
                        else:
                            raise ValueError(f"unsupported target protocol: {protocol}")
                        query_ids = adapter._query_ids(
                            query,
                            world_statement=world_statement,
                            prefix_ids=prefix_ids,
                        )
                        pending.append(
                            _PendingBranch(
                                row={
                                    "schema": "frank_eq_moment_compute_record_v1",
                                    "world_id": int(world.world_id),
                                    "entity_count": n_entities,
                                    "role": roles[world.world_id],
                                    "model_id": spec.model_id,
                                    "model_index": model_index,
                                    "renderer_id": renderer_id,
                                    "kind": "target",
                                    "family": operation.family,
                                    "operation_id": operation.operation_id,
                                    "item_id": operation.operation_id,
                                    "protocol": protocol,
                                    "truth": truth,
                                    "truth_hard": truth_hard,
                                },
                                protocol=protocol,
                                query_ids=query_ids,
                            )
                        )

                prefix_rows, stats = _score_pending_branches(
                    adapter,
                    prefix_ids,
                    prefix_output.past_key_values,
                    pending,
                    config,
                )
                records.extend(prefix_rows)
                response_batches += stats["response_batches"]
                max_batch_size = max(max_batch_size, stats["max_observed_batch_size"])
                prefix_count += 1
                if prefix_count % 16 == 0:
                    progress = {
                        "model_id": spec.model_id,
                        "prefixes": prefix_count,
                        "records": len(records) - model_start,
                        "response_batches": response_batches,
                    }
                    telemetry.log({"capture_progress": progress})
                    print(
                        "Stage M0 capture "
                        f"model={spec.model_id} prefixes={prefix_count} "
                        f"records={progress['records']}",
                        flush=True,
                    )
        model_metadata.append(
            {
                "model_index": model_index,
                "model_id": spec.model_id,
                "hf_id": spec.hf_id,
                "revision_requested": spec.revision,
                "revision_observed": observed_revision,
                "answer_labels": list(adapter.answer_labels),
                "answer_token_ids": [int(value) for value in adapter.answer_ids],
                "semantic_candidates": adapter.candidate_metadata(config.protocols),
                "records": len(records) - model_start,
                "prefixes": prefix_count,
                "branch_execution": {
                    "mode": "kv_reuse",
                    "exact_replay_response_branches": 0,
                    "allow_exact_replay_fallback": False,
                    "exclusive_cache_batching": True,
                    "response_batches": response_batches,
                    "max_observed_batch_size": max_batch_size,
                },
            }
        )
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records, model_metadata


def _calibration_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if row["kind"] == "event":
        return (
            row["model_id"],
            int(row["entity_count"]),
            "event",
            row["event_kind"],
            int(row["event_order"]),
            row["protocol"],
        )
    return (
        row["model_id"],
        int(row["entity_count"]),
        "target",
        row["family"],
        row["protocol"],
    )


def calibrate_moment_records(
    records: list[dict[str, Any]], config: MomentComputeRunConfig
) -> dict[str, Any]:
    """Fit readout maps on calibration worlds only; never fit event-ID-specific slopes."""

    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[_calibration_key(row)].append(index)
    artifact: dict[str, Any] = {
        "schema": "frank_eq_moment_compute_calibration_v1",
        "fit_role": "calibration",
        "groups": {},
    }
    for key, indices in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        fit_indices = [index for index in indices if records[index]["role"] == "calibration"]
        if len(fit_indices) < 4:
            raise RuntimeError(f"calibration group {key} has fewer than four rows")
        calibrator = fit_platt_calibrator(
            np.asarray([records[index]["log_odds_score"] for index in fit_indices]),
            np.asarray([records[index]["truth"] for index in fit_indices]),
            l2=config.evaluation.calibration_l2,
            max_steps=config.evaluation.calibration_max_steps,
        )
        scores = np.asarray([records[index]["log_odds_score"] for index in indices])
        predictions = calibrator.predict(scores)
        for index, probability in zip(indices, predictions, strict=True):
            records[index]["calibrated_probability"] = float(
                np.clip(
                    probability,
                    config.evaluation.probability_epsilon,
                    1.0 - config.evaluation.probability_epsilon,
                )
            )
        artifact["groups"]["|".join(map(str, key))] = {
            "rows": len(indices),
            "fit_rows": len(fit_indices),
            "calibrator": calibrator.to_dict(),
        }

    prior_groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in records:
        if row["role"] == "calibration":
            prior_groups[(str(row["kind"]), int(row["item_id"]))].append(float(row["truth"]))
    priors = {
        key: float(
            np.clip(
                np.mean(values),
                config.evaluation.probability_epsilon,
                1.0 - config.evaluation.probability_epsilon,
            )
        )
        for key, values in prior_groups.items()
    }
    for row in records:
        row["prior_probability"] = priors[(str(row["kind"]), int(row["item_id"]))]
    artifact["item_priors"] = {"|".join(map(str, key)): value for key, value in priors.items()}
    return artifact


def select_direct_protocols(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select direct response protocols on the disjoint selection role."""

    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["kind"] == "target" and row["role"] == "selection":
            grouped[
                (
                    str(row["model_id"]),
                    int(row["entity_count"]),
                    str(row["family"]),
                    str(row["protocol"]),
                )
            ].append(row)
    candidates: dict[tuple[str, int, str], list[tuple[float, str, int]]] = defaultdict(list)
    for (model_id, entity_count, family, protocol), rows in grouped.items():
        score = brier_score(
            np.asarray([row["truth"] for row in rows]),
            np.asarray([row["calibrated_probability"] for row in rows]),
        )
        candidates[(model_id, entity_count, family)].append((score, protocol, len(rows)))
    selected: dict[str, Any] = {
        "schema": "frank_eq_moment_compute_direct_selection_v1",
        "selection_role": "selection",
        "groups": {},
    }
    for key, values in sorted(candidates.items()):
        values.sort(key=lambda item: (item[0], item[1]))
        selected["groups"]["|".join(map(str, key))] = {
            "selected_protocol": values[0][1],
            "selection_brier": values[0][0],
            "candidates": [
                {"protocol": protocol, "brier": score, "rows": rows}
                for score, protocol, rows in values
            ],
        }
    return selected


def _selected_protocol(selection: dict[str, Any], model_id: str, n_entities: int, family: str) -> str:
    key = f"{model_id}|{n_entities}|{family}"
    try:
        return str(selection["groups"][key]["selected_protocol"])
    except KeyError as error:
        raise KeyError(f"direct protocol selection is missing {key}") from error


def compile_validation_predictions(
    records: list[dict[str, Any]],
    panel: RealPanel,
    registry: EventRegistry,
    selection: dict[str, Any],
    config: MomentComputeRunConfig,
) -> list[dict[str, Any]]:
    event_rows: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    target_rows: dict[tuple[str, int, int, int, str], dict[str, Any]] = {}
    for row in records:
        if row["role"] != "validation":
            continue
        if row["kind"] == "event":
            event_rows[(row["model_id"], row["world_id"], row["renderer_id"])].append(row)
        else:
            target_rows[
                (
                    row["model_id"],
                    row["world_id"],
                    row["renderer_id"],
                    row["operation_id"],
                    row["protocol"],
                )
            ] = row
    predictions: list[dict[str, Any]] = []
    for (model_id, world_id, renderer_id), rows in sorted(event_rows.items()):
        raw = {row["event_key"]: float(row["calibrated_probability"]) for row in rows}
        projected, projection = project_event_probabilities(
            registry,
            raw,
            epsilon=config.evaluation.probability_epsilon,
        )
        for frozen_operation in panel.operations:
            operation = frozen_operation.definition
            if operation.family not in COMPILED_FAMILIES:
                continue
            protocol = _selected_protocol(selection, model_id, panel.n_entities, operation.family)
            direct_row = target_rows[
                (model_id, world_id, renderer_id, operation.operation_id, protocol)
            ]
            moment = compile_operation_from_events(
                projected,
                operation,
                n_entities=panel.n_entities,
                epsilon=config.evaluation.probability_epsilon,
            )
            marginal = compile_operation_from_marginals(
                projected,
                operation,
                n_entities=panel.n_entities,
            )
            predictions.append(
                {
                    "schema": "frank_eq_moment_compute_prediction_v1",
                    "model_id": model_id,
                    "entity_count": panel.n_entities,
                    "world_id": int(world_id),
                    "renderer_id": int(renderer_id),
                    "operation_id": operation.operation_id,
                    "family": operation.family,
                    "truth": float(direct_row["truth"]),
                    "truth_hard": int(direct_row["truth_hard"]),
                    "moment_probability": float(moment),
                    "marginal_probability": float(marginal),
                    "direct_probability": float(direct_row["calibrated_probability"]),
                    "prior_probability": float(direct_row["prior_probability"]),
                    "direct_protocol": protocol,
                    "projection_mean_absolute_adjustment": projection[
                        "mean_absolute_adjustment"
                    ],
                    "projection_max_absolute_adjustment": projection[
                        "max_absolute_adjustment"
                    ],
                }
            )
    return predictions


def _world_interval(
    values: list[float],
    world_ids: list[int],
    config: MomentComputeRunConfig,
    seed: int,
) -> dict[str, Any]:
    worlds = np.asarray(world_ids, dtype=np.int64)
    array = np.asarray(values, dtype=np.float64)
    unique = np.unique(worlds)
    grouped = np.asarray([array[worlds == world].mean() for world in unique], dtype=np.float64)
    return bootstrap_statistic(
        grouped,
        replicates=config.evaluation.bootstrap_replicates,
        seed=config.evaluation.bootstrap_seed + seed,
    ).to_dict()


def _balanced_accuracy_interval(
    rows: list[dict[str, Any]], config: MomentComputeRunConfig, seed: int
) -> dict[str, Any]:
    worlds = np.asarray([row["world_id"] for row in rows], dtype=np.int64)
    unique = np.unique(worlds)
    rng = np.random.default_rng(config.evaluation.bootstrap_seed + seed)
    values = np.empty(config.evaluation.bootstrap_replicates, dtype=np.float64)
    for index in range(config.evaluation.bootstrap_replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        truth: list[float] = []
        probability: list[float] = []
        for world in sampled:
            selected = [row for row in rows if row["world_id"] == int(world)]
            truth.extend(float(row["truth"]) for row in selected)
            probability.extend(float(row["calibrated_probability"]) for row in selected)
        values[index] = balanced_accuracy(np.asarray(truth), np.asarray(probability))
    estimate = balanced_accuracy(
        np.asarray([row["truth"] for row in rows]),
        np.asarray([row["calibrated_probability"] for row in rows]),
    )
    return {
        "estimate": float(estimate),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "replicates": int(config.evaluation.bootstrap_replicates),
    }


def _contrast(
    rows: list[dict[str, Any]],
    candidate_key: str,
    baseline_key: str,
    config: MomentComputeRunConfig,
    seed: int,
) -> dict[str, Any]:
    values = [
        (float(row[baseline_key]) - float(row["truth"])) ** 2
        - (float(row[candidate_key]) - float(row["truth"])) ** 2
        for row in rows
    ]
    interval = _world_interval(values, [row["world_id"] for row in rows], config, seed)
    return {
        "candidate_brier": brier_score(
            np.asarray([row["truth"] for row in rows]),
            np.asarray([row[candidate_key] for row in rows]),
        ),
        "baseline_brier": brier_score(
            np.asarray([row["truth"] for row in rows]),
            np.asarray([row[baseline_key] for row in rows]),
        ),
        "brier_gain": float(np.mean(values)),
        "brier_gain_ci": interval,
        "rows": len(rows),
        "worlds": len(set(row["world_id"] for row in rows)),
    }


def evaluate_moment_compute(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    panel: RealPanel,
    registry: EventRegistry,
    config: MomentComputeRunConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation_events = [
        row for row in records if row["role"] == "validation" and row["kind"] == "event"
    ]
    event_groups: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in validation_events:
        event_groups[
            (
                str(row["model_id"]),
                int(row["entity_count"]),
                str(row["event_kind"]),
                int(row["event_order"]),
            )
        ].append(row)
    event_metrics: dict[str, Any] = {}
    event_checks: list[bool] = []
    for group_index, (key, rows) in enumerate(sorted(event_groups.items())):
        gain_values = [
            (float(row["prior_probability"]) - float(row["truth"])) ** 2
            - (float(row["calibrated_probability"]) - float(row["truth"])) ** 2
            for row in rows
        ]
        gain_ci = _world_interval(
            gain_values,
            [row["world_id"] for row in rows],
            config,
            10_000 + group_index,
        )
        accuracy_ci = _balanced_accuracy_interval(rows, config, 20_000 + group_index)
        passed = (
            float(gain_ci["lower"]) >= config.gates.min_event_brier_gain_lower95
            and float(accuracy_ci["lower"])
            >= config.gates.min_event_balanced_accuracy_lower95
        )
        event_checks.append(passed)
        event_metrics["|".join(map(str, key))] = {
            "candidate_brier": brier_score(
                np.asarray([row["truth"] for row in rows]),
                np.asarray([row["calibrated_probability"] for row in rows]),
            ),
            "prior_brier": brier_score(
                np.asarray([row["truth"] for row in rows]),
                np.asarray([row["prior_probability"] for row in rows]),
            ),
            "brier_gain_ci": gain_ci,
            "balanced_accuracy_ci": accuracy_ci,
            "passed": passed,
            "rows": len(rows),
        }

    hard_rows = [row for row in predictions if row["family"] in HARD_FAMILIES]
    if not hard_rows:
        raise RuntimeError("Stage M0 produced no hard-family predictions")
    aggregate_marginal = _contrast(
        hard_rows,
        "moment_probability",
        "marginal_probability",
        config,
        30_001,
    )
    aggregate_direct = _contrast(
        hard_rows,
        "moment_probability",
        "direct_probability",
        config,
        30_002,
    )
    by_model: dict[str, Any] = {}
    robust_model_checks: list[bool] = []
    for model_index, model_id in enumerate(sorted({row["model_id"] for row in hard_rows})):
        model_rows = [row for row in hard_rows if row["model_id"] == model_id]
        over_marginal = _contrast(
            model_rows,
            "moment_probability",
            "marginal_probability",
            config,
            31_000 + model_index * 2,
        )
        over_direct = _contrast(
            model_rows,
            "moment_probability",
            "direct_probability",
            config,
            31_001 + model_index * 2,
        )
        passed = (
            float(over_marginal["brier_gain_ci"]["lower"])
            > config.gates.min_moment_over_marginal_gain_lower95
            and float(over_direct["brier_gain_ci"]["lower"])
            > config.gates.min_moment_over_direct_gain_lower95
        )
        robust_model_checks.append(passed)
        by_model[model_id] = {
            "over_marginal": over_marginal,
            "over_direct": over_direct,
            "passed": passed,
        }

    by_family: dict[str, Any] = {}
    for family_index, family in enumerate(sorted(HARD_FAMILIES)):
        rows = [row for row in hard_rows if row["family"] == family]
        by_family[family] = {
            "over_marginal": _contrast(
                rows,
                "moment_probability",
                "marginal_probability",
                config,
                40_000 + family_index * 2,
            ),
            "over_direct": _contrast(
                rows,
                "moment_probability",
                "direct_probability",
                config,
                40_001 + family_index * 2,
            ),
        }

    atomic_rows = [row for row in predictions if row["family"] in {"lookup", "inverse"}]
    atomic_retention = _contrast(
        atomic_rows,
        "moment_probability",
        "marginal_probability",
        config,
        50_001,
    )
    operations = [operation.definition for operation in panel.operations]
    mismatches = exact_executor_mismatches(registry, panel.worlds, operations)

    checks = {
        "event_algebra_exact": mismatches <= config.gates.max_executor_mismatches,
        "operation_closed_events_readable": bool(event_checks) and all(event_checks),
        "moment_over_marginal_aggregate": (
            float(aggregate_marginal["brier_gain_ci"]["lower"])
            > config.gates.min_moment_over_marginal_gain_lower95
        ),
        "moment_over_direct_aggregate": (
            float(aggregate_direct["brier_gain_ci"]["lower"])
            > config.gates.min_moment_over_direct_gain_lower95
        ),
        "moment_advantage_robust_by_model": bool(robust_model_checks)
        and all(robust_model_checks),
        "atomic_retention": (
            float(atomic_retention["brier_gain_ci"]["lower"])
            >= config.gates.min_atomic_retention_lower95
        ),
    }
    if not checks["event_algebra_exact"]:
        diagnosis = "EVENT_ALGEBRA_IMPLEMENTATION_INVALID"
    elif not checks["operation_closed_events_readable"]:
        diagnosis = "OPERATION_CLOSED_EVENTS_NOT_READABLE"
    elif not checks["moment_over_marginal_aggregate"]:
        diagnosis = "NO_JOINT_MOMENT_ADVANTAGE_OVER_MARGINALS"
    elif not checks["moment_over_direct_aggregate"]:
        diagnosis = "MOMENT_BASIS_DOES_NOT_BEAT_CROSSFITTED_DIRECT"
    elif not checks["atomic_retention"]:
        diagnosis = "MOMENT_PROJECTION_HARMS_ATOMIC_OPERATIONS"
    elif not checks["moment_advantage_robust_by_model"]:
        diagnosis = "MOMENT_ADVANTAGE_NOT_ROBUST_ACROSS_MODELS_AND_COMPLEXITIES"
    else:
        diagnosis = "OPERATION_CLOSED_MOMENT_BASIS_SUPPORTED"
    passed = all(checks.values())
    metrics = {
        "schema": "frank_eq_moment_compute_metrics_v1",
        "role": "development_validation",
        "event_registry_sha256": registry.sha256,
        "event_groups": event_metrics,
        "composition": {
            "aggregate_over_marginal": aggregate_marginal,
            "aggregate_over_crossfitted_direct": aggregate_direct,
            "by_model": by_model,
            "by_hard_family": by_family,
        },
        "atomic_retention": atomic_retention,
        "executor_mismatches": mismatches,
        "bootstrap_replicates": config.evaluation.bootstrap_replicates,
    }
    decision = {
        "schema": "frank_eq_moment_compute_decision_v1",
        "status": "pass" if passed else "fail",
        "diagnosis": diagnosis,
        "checks": checks,
        "authorization": {
            "successor_protocol_draft_authorized": bool(passed),
            "one_shot_compiler_run_authorized": False,
            "held_sender_authorized": False,
            "claim_bearing_test_authorized": False,
            "receiver_protocol_draft_authorized": False,
            "receiver_execution_authorized": False,
            "scientific_claim_authorized": False,
            "paper_claim_authorized": False,
        },
    }
    return metrics, decision


def _artifact_manifest(root: Path, relative_paths: list[str]) -> dict[str, Any]:
    return {
        "schema": "frank_eq_moment_compute_artifact_manifest_v1",
        "files": {
            relative: sha256_file(root / relative)
            for relative in relative_paths
            if (root / relative).is_file()
        },
    }


def run_moment_compute_audit(
    config: MomentComputeRunConfig,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    stages: str | Iterable[str] = MOMENT_COMPUTE_ALLOWED_STAGES,
) -> dict[str, Any]:
    selected = parse_moment_compute_stages(stages)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_file = Path(config_path)
    shutil.copyfile(config_file, root / "config.yaml")
    telemetry = WandbTelemetry(config.logging.wandb, run_name=config.run_name, job=_environment())
    run_manifest = {
        "schema": "frank_eq_moment_compute_run_manifest_v1",
        "run_name": config.run_name,
        "protocol_version": config.protocol_version,
        "development_only": True,
        "created_at": _timestamp(),
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "stages": list(selected),
        "environment": _environment(),
        "access_contract": MOMENT_ACCESS_CONTRACT,
    }
    atomic_write_json(root / "run_manifest.json", run_manifest)
    status: dict[str, Any] = {
        "schema": "frank_eq_moment_compute_status_v1",
        "state": "running",
        "current_stage": "audit",
        "completed_stages": [],
        "started_at": _timestamp(),
        "failure": None,
    }
    atomic_write_json(root / "workflow_status.json", status)
    started = time.time()
    try:
        panel = build_moment_panel(config)
        roles = build_development_roles(config)
        registry = build_event_registry(panel.n_entities)
        atomic_write_json(root / "panels" / "n4.json", panel.to_dict())
        # Generic cluster verification historically expects both panel paths for
        # audit jobs. This explicit tombstone prevents an unused n=6 experiment
        # from being mistaken for a missing artifact.
        atomic_write_json(
            root / "panels" / "n6.json",
            {
                "schema": "frank_eq_moment_compute_unused_panel_v1",
                "entity_count": 6,
                "used": False,
                "reason": "Stage M0 is frozen to the four-entity kill canary",
            },
        )
        atomic_write_json(root / "event_registry.json", registry.to_dict())
        role_payload = {
            "schema": "frank_eq_moment_compute_roles_v1",
            "calibration_world_ids": sorted(
                world for world, role in roles.items() if role == "calibration"
            ),
            "selection_world_ids": sorted(
                world for world, role in roles.items() if role == "selection"
            ),
            "validation_world_ids": sorted(
                world for world, role in roles.items() if role == "validation"
            ),
            "test_world_ids": [],
        }
        atomic_write_json(root / "development_splits.json", role_payload)

        records, model_metadata = capture_moment_records(
            config,
            panel,
            roles,
            registry,
            telemetry,
        )
        atomic_write_json(root / "models.json", model_metadata)
        _write_jsonl(root / "records_raw.jsonl", records)
        calibration = calibrate_moment_records(records, config)
        atomic_write_json(root / "calibration.json", calibration)
        _write_jsonl(root / "records_calibrated.jsonl", records)
        selection = select_direct_protocols(records)
        atomic_write_json(root / "direct_protocol_selection.json", selection)
        predictions = compile_validation_predictions(
            records,
            panel,
            registry,
            selection,
            config,
        )
        _write_jsonl(root / "compiled_predictions.jsonl", predictions)
        metrics, decision = evaluate_moment_compute(
            records,
            predictions,
            panel,
            registry,
            config,
        )
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
            "schema": "frank_eq_moment_compute_run_v1",
            "status": "completed",
            "workflow_integrity_passed": True,
            "development_only": True,
            "root": str(root),
            "records": len(records),
            "compiled_predictions": len(predictions),
            "event_coordinates": len(registry.events),
            "decision": decision,
            "authorizes_scientific_claim": False,
            "telemetry": telemetry.status(),
        }
        atomic_write_json(root / "run_summary.json", summary)
        artifact_paths = [
            "config.yaml",
            "run_manifest.json",
            "workflow_status.json",
            "development_splits.json",
            "panels/n4.json",
            "panels/n6.json",
            "event_registry.json",
            "models.json",
            "records_raw.jsonl",
            "calibration.json",
            "records_calibrated.jsonl",
            "compiled_predictions.jsonl",
            "direct_protocol_selection.json",
            "metrics.json",
            "decision.json",
            "run_summary.json",
        ]
        atomic_write_json(root / "artifact_manifest.json", _artifact_manifest(root, artifact_paths))
        from .verify import verify_moment_compute_run

        verification = verify_moment_compute_run(
            root,
            config_path=root / "config.yaml",
            write_verification=True,
        )
        if not verification["passed"]:
            raise RuntimeError("independent Stage M0 verification failed")
        telemetry.log(
            {
                "decision": {
                    "status": decision["status"],
                    "diagnosis": decision["diagnosis"],
                }
            }
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
    finally:
        telemetry.finish()


def static_contract_summary(config_path: str | Path) -> dict[str, Any]:
    config = load_moment_compute_config(config_path)
    panel = build_moment_panel(config)
    registry = build_event_registry(panel.n_entities)
    mismatches = exact_executor_mismatches(
        registry,
        panel.worlds,
        [operation.definition for operation in panel.operations],
    )
    if mismatches:
        raise RuntimeError(f"operation-closed executor has {mismatches} exact mismatches")
    return {
        "schema": "frank_eq_moment_compute_static_validation_v1",
        "status": "passed",
        "protocol_version": config.protocol_version,
        "models": [model.model_id for model in config.models],
        "entity_counts": config.panel.entity_counts,
        "worlds": len(panel.worlds),
        "operations": len(panel.operations),
        "event_coordinates": len(registry.events),
        "event_registry_sha256": registry.sha256,
        "executor_mismatches": mismatches,
        "protected_authorizations_closed": True,
        "contract_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "config": config.as_dict(),
                    "event_registry": registry.to_dict(),
                    "panel_config": panel.config,
                }
            )
        ),
    }
