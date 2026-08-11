"""World-grouped Stage-A v3 metrics, rate accounting, and machine decision."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from frank_eq.evaluation.metrics import model_identity_probe
from frank_eq.rate_compute.calibration import (
    aggregate_by_world,
    balanced_accuracy,
    brier_score,
    expected_calibration_error,
    interval,
)
from frank_eq.rate_compute.logic import HARD_COMPOSITION_FAMILIES

from .config import StageAV3Config
from .panel import V3Panel
from .predictions import V3PredictionBundle


def _model_complexity_predicate(model_id: str, entity_count: int) -> Any:
    def predicate(bundle: V3PredictionBundle) -> bool:
        return bundle.model_id == model_id and bundle.entity_count == entity_count

    return predicate


def _global_world_ids(bundle: V3PredictionBundle, repeats: int) -> np.ndarray:
    base = bundle.entity_count * 1_000_000 + bundle.world_ids.astype(np.int64)
    return np.repeat(base, repeats)


def _basis_rows(
    bundles: list[V3PredictionBundle],
    *,
    channel: str,
    condition: str,
    predicate: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    worlds: list[np.ndarray] = []
    for bundle in bundles:
        if predicate is not None and not predicate(bundle):
            continue
        if channel == "semantic":
            truth = bundle.semantic_truth
            prediction = bundle.semantic_basis[condition]
        elif channel == "behavioral":
            truth = bundle.behavioral_truth
            prediction = bundle.behavioral_basis[condition]
        else:
            raise ValueError(f"unsupported Stage-A v3 metric channel: {channel}")
        truths.append(truth.reshape(-1))
        predictions.append(prediction.reshape(-1))
        worlds.append(_global_world_ids(bundle, bundle.coordinate_count))
    if not truths:
        raise RuntimeError("basis metric stratum is empty")
    return np.concatenate(truths), np.concatenate(predictions), np.concatenate(worlds)


def _operation_rows(
    bundles: list[V3PredictionBundle],
    panels: dict[int, V3Panel],
    *,
    condition: str,
    families: set[str] | frozenset[str] | None = None,
    predicate: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    worlds: list[np.ndarray] = []
    for bundle in bundles:
        if predicate is not None and not predicate(bundle):
            continue
        panel = panels[bundle.entity_count]
        operation_ids = [
            operation.definition.operation_id
            for operation in panel.panel.operations
            if families is None or operation.definition.family in families
        ]
        if not operation_ids:
            continue
        truth = bundle.operation_truth[:, operation_ids]
        prediction = bundle.operations[condition][:, operation_ids]
        truths.append(truth.reshape(-1))
        predictions.append(prediction.reshape(-1))
        worlds.append(_global_world_ids(bundle, len(operation_ids)))
    if not truths:
        raise RuntimeError("operation metric stratum is empty")
    return np.concatenate(truths), np.concatenate(predictions), np.concatenate(worlds)


def _world_interval(values: np.ndarray, worlds: np.ndarray, config: StageAV3Config, seed: int) -> dict[str, Any]:
    _, grouped = aggregate_by_world(values, worlds)
    evaluation = config.section("evaluation")
    return interval(
        grouped,
        replicates=int(evaluation["bootstrap_replicates"]),
        seed=int(evaluation["bootstrap_seed"]) + seed,
    )


def _gain_summary(
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    worlds: np.ndarray,
    config: StageAV3Config,
    *,
    seed: int,
) -> dict[str, Any]:
    gains = (truth - baseline) ** 2 - (truth - candidate) ** 2
    return {
        "candidate_brier": brier_score(truth, candidate),
        "baseline_brier": brier_score(truth, baseline),
        "brier_gain": float(np.mean(gains)),
        "brier_gain_ci": _world_interval(gains, worlds, config, seed),
    }


def _balanced_accuracy_interval(
    truth: np.ndarray,
    prediction: np.ndarray,
    worlds: np.ndarray,
    config: StageAV3Config,
    *,
    seed: int,
) -> dict[str, Any]:
    unique_worlds = np.unique(worlds)
    values = np.asarray(
        [balanced_accuracy(truth[worlds == world], prediction[worlds == world]) for world in unique_worlds],
        dtype=np.float64,
    )
    summary = _world_interval(values, unique_worlds, config, seed)
    summary["pooled"] = balanced_accuracy(truth, prediction)
    return summary


def _basis_group_summary(
    bundles: list[V3PredictionBundle],
    config: StageAV3Config,
    *,
    channel: str,
    candidate: str,
    baseline: str,
    seed: int,
    predicate: Any | None = None,
) -> dict[str, Any]:
    truth, prediction, worlds = _basis_rows(
        bundles,
        channel=channel,
        condition=candidate,
        predicate=predicate,
    )
    _, baseline_prediction, baseline_worlds = _basis_rows(
        bundles,
        channel=channel,
        condition=baseline,
        predicate=predicate,
    )
    if not np.array_equal(worlds, baseline_worlds):
        raise RuntimeError("basis candidate/baseline world rows differ")
    result = _gain_summary(
        truth,
        prediction,
        baseline_prediction,
        worlds,
        config,
        seed=seed,
    )
    result.update(
        {
            "rows": int(truth.size),
            "worlds": int(np.unique(worlds).size),
            "balanced_accuracy_ci": _balanced_accuracy_interval(
                truth,
                prediction,
                worlds,
                config,
                seed=seed + 1,
            ),
            "ece": expected_calibration_error(
                truth,
                prediction,
                bins=int(config.section("evaluation")["ece_bins"]),
            ),
        }
    )
    return result


def _operation_gain_summary(
    bundles: list[V3PredictionBundle],
    panels: dict[int, V3Panel],
    config: StageAV3Config,
    *,
    candidate: str,
    baseline: str,
    families: set[str] | frozenset[str] | None,
    seed: int,
    predicate: Any | None = None,
) -> dict[str, Any]:
    truth, prediction, worlds = _operation_rows(
        bundles,
        panels,
        condition=candidate,
        families=families,
        predicate=predicate,
    )
    _, baseline_prediction, baseline_worlds = _operation_rows(
        bundles,
        panels,
        condition=baseline,
        families=families,
        predicate=predicate,
    )
    if not np.array_equal(worlds, baseline_worlds):
        raise RuntimeError("operation candidate/baseline world rows differ")
    result = _gain_summary(
        truth,
        prediction,
        baseline_prediction,
        worlds,
        config,
        seed=seed,
    )
    result.update({"rows": int(truth.size), "worlds": int(np.unique(worlds).size)})
    return result


def _aggregate_model_views(
    values: np.ndarray,
    world_ids: np.ndarray,
    renderer_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    del renderer_ids
    worlds = np.unique(world_ids)
    return (
        np.stack([values[world_ids == world].mean(axis=0) for world in worlds]),
        worlds,
    )


def _retrieval_metrics(
    bundles: list[V3PredictionBundle],
    config: StageAV3Config,
) -> dict[str, Any]:
    outcomes: list[float] = []
    margins: list[float] = []
    worlds: list[int] = []
    by_complexity: dict[str, Any] = {}
    for entity_count in config.section("panel")["entity_counts"]:
        selected = [bundle for bundle in bundles if bundle.entity_count == entity_count]
        codes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for bundle in selected:
            codes[bundle.model_id] = _aggregate_model_views(
                bundle.semantic_basis["primary_q4"],
                bundle.world_ids,
                bundle.renderer_ids,
            )
        complexity_outcomes: list[float] = []
        complexity_margins: list[float] = []
        for source_id, (source_code, source_worlds) in codes.items():
            source_code = source_code / np.clip(
                np.linalg.norm(source_code, axis=1, keepdims=True), 1e-12, None
            )
            for target_id, (target_code, target_worlds) in codes.items():
                if target_id == source_id:
                    continue
                target_code = target_code / np.clip(
                    np.linalg.norm(target_code, axis=1, keepdims=True), 1e-12, None
                )
                similarities = source_code @ target_code.T
                for source_index, world in enumerate(source_worlds):
                    matching = np.flatnonzero(target_worlds == world)
                    if len(matching) != 1:
                        raise RuntimeError("cross-model retrieval world alignment is incomplete")
                    correct = int(matching[0])
                    wrong = np.delete(similarities[source_index], correct)
                    outcome = float(int(np.argmax(similarities[source_index])) == correct)
                    margin = float(similarities[source_index, correct] - np.max(wrong))
                    complexity_outcomes.append(outcome)
                    complexity_margins.append(margin)
                    outcomes.append(outcome)
                    margins.append(margin)
                    worlds.append(entity_count * 1_000_000 + int(world))
        by_complexity[str(entity_count)] = {
            "retrieval": float(np.mean(complexity_outcomes)),
            "margin": float(np.mean(complexity_margins)),
        }
    world_array = np.asarray(worlds, dtype=np.int64)
    return {
        "retrieval_ci": _world_interval(
            np.asarray(outcomes), world_array, config, 70_000
        ),
        "wrong_world_margin_ci": _world_interval(
            np.asarray(margins), world_array, config, 70_001
        ),
        "by_complexity": by_complexity,
    }


def _model_identity_metrics(
    bundles: list[V3PredictionBundle],
    config: StageAV3Config,
    train_identity_basis: dict[tuple[str, int], dict[str, np.ndarray]],
) -> dict[str, Any]:
    model_lookup = {model.model_id: index for index, model in enumerate(config.models)}
    by_complexity: dict[str, float] = {}
    weighted_correct = 0.0
    weighted_rows = 0
    for entity_count in config.section("panel")["entity_counts"]:
        train_code: list[np.ndarray] = []
        train_models: list[np.ndarray] = []
        test_code: list[np.ndarray] = []
        test_models: list[np.ndarray] = []
        for model in config.models:
            key = (model.model_id, entity_count)
            if key not in train_identity_basis:
                raise RuntimeError("model-identity probe lacks a train-role packet group")
            train = train_identity_basis[key]
            train_aggregated, _ = _aggregate_model_views(
                train["probabilities"], train["world_ids"], train["renderer_ids"]
            )
            train_code.append(train_aggregated)
            train_models.append(
                np.full(len(train_aggregated), model_lookup[model.model_id], dtype=np.int64)
            )
            bundle = next(
                bundle
                for bundle in bundles
                if bundle.model_id == model.model_id and bundle.entity_count == entity_count
            )
            test_aggregated, _ = _aggregate_model_views(
                bundle.semantic_basis["primary_q4"],
                bundle.world_ids,
                bundle.renderer_ids,
            )
            test_code.append(test_aggregated)
            test_models.append(
                np.full(len(test_aggregated), model_lookup[model.model_id], dtype=np.int64)
            )
        train_code_array = np.concatenate(train_code)
        train_model_array = np.concatenate(train_models)
        test_code_array = np.concatenate(test_code)
        test_model_array = np.concatenate(test_models)
        accuracy = model_identity_probe(
            train_code_array,
            train_model_array,
            test_code_array,
            test_model_array,
            ridge=1.0,
        )
        by_complexity[str(entity_count)] = accuracy
        weighted_correct += accuracy * len(test_model_array)
        weighted_rows += len(test_model_array)
    accuracy = weighted_correct / weighted_rows
    chance = 1.0 / len(config.models)
    return {
        "accuracy": accuracy,
        "chance": chance,
        "over_chance": accuracy - chance,
        "fit_role": "train",
        "score_role": "test",
        "by_complexity": by_complexity,
    }


def _oracle_hard_mismatches(
    bundles: list[V3PredictionBundle],
    panels: dict[int, V3Panel],
) -> int:
    mismatches = 0
    for bundle in bundles:
        operation_ids = [
            operation.definition.operation_id
            for operation in panels[bundle.entity_count].panel.operations
            if operation.definition.family in HARD_COMPOSITION_FAMILIES
        ]
        predicted = bundle.operations["oracle_basis"][:, operation_ids] >= 0.5
        truth = bundle.operation_truth_hard[:, operation_ids].astype(bool)
        mismatches += int(np.count_nonzero(predicted != truth))
    return mismatches


def summarize_rate_compute(
    bundles: list[V3PredictionBundle],
    config: StageAV3Config,
) -> dict[str, Any]:
    packet_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for bundle in bundles:
        for record in bundle.packet_records:
            packet_groups[
                (
                    str(record["condition"]),
                    int(record["entity_count"]),
                    int(record["bits_per_coordinate"]),
                )
            ].append(record)
    packet_rates = {
        "|".join(map(str, key)): {
            "rows": len(records),
            "payload_bits": int(records[0]["payload_bits"]),
            "framing_bits": int(records[0]["framing_bits"]),
            "serialized_bits": int(records[0]["serialized_bits"]),
        }
        for key, records in sorted(packet_groups.items())
    }
    amortized: dict[str, Any] = {}
    for entity_count in config.section("panel")["entity_counts"]:
        coordinates = entity_count * (entity_count - 1)
        amortized[str(entity_count)] = {}
        for operations in config.section("evaluation")["amortized_operation_counts"]:
            amortized[str(entity_count)][str(operations)] = {
                "primary": {
                    "payload_bits_per_operation": coordinates * 4 / operations,
                    "post_capture_source_queries": 0,
                    "executor_operations": operations,
                },
                "interactive_basis": {
                    "basis_source_queries_per_operation": coordinates / operations,
                    "executor_operations": operations,
                },
                "direct": {
                    "source_queries_per_operation": 1,
                    "total_source_queries": operations,
                },
            }
    return {
        "schema": "frank_eq_stagea_v3_rate_compute_v1",
        "packet_rates": packet_rates,
        "amortized": amortized,
        "bundle_compute": {
            f"{bundle.model_id}|{bundle.entity_count}": bundle.compute for bundle in bundles
        },
        "consumer_compute_declared": True,
        "framing_counted_separately": True,
    }


def evaluate_stagea_v3(
    config: StageAV3Config,
    *,
    bundles: list[V3PredictionBundle],
    panels: dict[int, V3Panel],
    train_identity_basis: dict[tuple[str, int], dict[str, np.ndarray]],
    integrity_checks: dict[str, bool],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reduce all registered test predictions under world-grouped intervals."""

    expected_groups = {
        (model.model_id, entity_count)
        for model in config.models
        for entity_count in config.section("panel")["entity_counts"]
    }
    observed_groups = {(bundle.model_id, bundle.entity_count) for bundle in bundles}
    if observed_groups != expected_groups or len(bundles) != len(expected_groups):
        raise ValueError("prediction bundles do not cover the exact model/complexity registry")
    if set(panels) != set(config.section("panel")["entity_counts"]):
        raise ValueError("evaluation panels do not cover every complexity")
    for bundle in bundles:
        bundle.validate()
    gates = config.section("gates")
    metrics: dict[str, Any] = {
        "schema": "frank_eq_stagea_v3_metrics_v1",
        "role": "test",
        "world_grouped": True,
        "bootstrap_replicates": config.section("evaluation")["bootstrap_replicates"],
        "semantic_basis": {},
        "behavioral_basis": {},
        "renderer_strata": {},
        "compiler_seed_strata": {},
        "composition": {},
    }

    semantic_checks: list[bool] = []
    behavioral_checks: list[bool] = []
    unseen_checks: list[bool] = []
    semantic_gains: dict[tuple[str, int], float] = {}
    behavioral_gains: dict[tuple[str, int], float] = {}
    for model_index, model in enumerate(config.models):
        metrics["semantic_basis"][model.model_id] = {}
        metrics["behavioral_basis"][model.model_id] = {}
        metrics["renderer_strata"][model.model_id] = {}
        metrics["compiler_seed_strata"][model.model_id] = {}
        for complexity_index, entity_count in enumerate(
            config.section("panel")["entity_counts"]
        ):
            predicate = _model_complexity_predicate(model.model_id, entity_count)
            seed = 1000 * model_index + 100 * complexity_index
            semantic = _basis_group_summary(
                bundles,
                config,
                channel="semantic",
                candidate="primary_q4",
                baseline="train_edge_prior",
                seed=seed,
                predicate=predicate,
            )
            semantic["passed"] = (
                float(semantic["brier_gain_ci"]["lower"])
                > float(gates["semantic_basis_brier_gain_lower95_strict_gt"])
                and float(semantic["candidate_brier"])
                <= float(gates["max_semantic_basis_brier"])
                and float(semantic["balanced_accuracy_ci"]["lower"])
                >= float(gates["min_semantic_balanced_accuracy_lower95"])
            )
            semantic_checks.append(bool(semantic["passed"]))
            semantic_gains[(model.model_id, entity_count)] = float(semantic["brier_gain"])
            metrics["semantic_basis"][model.model_id][str(entity_count)] = semantic

            behavioral = _basis_group_summary(
                bundles,
                config,
                channel="behavioral",
                candidate="primary_q4",
                baseline="train_edge_prior",
                seed=seed + 10,
                predicate=predicate,
            )
            behavioral["passed"] = (
                float(behavioral["brier_gain_ci"]["lower"])
                > float(gates["behavioral_brier_gain_lower95_strict_gt"])
            )
            behavioral_checks.append(bool(behavioral["passed"]))
            behavioral_gains[(model.model_id, entity_count)] = float(
                behavioral["brier_gain"]
            )
            metrics["behavioral_basis"][model.model_id][str(entity_count)] = behavioral

            renderer_result: dict[str, Any] = {}
            for renderer_id, label in ((0, "natural"), (1, "adjacency"), (2, "unseen")):
                selected_bundle = next(bundle for bundle in bundles if predicate(bundle))
                selection = selected_bundle.renderer_ids == renderer_id
                truth = selected_bundle.semantic_truth[selection].reshape(-1)
                candidate = selected_bundle.semantic_basis["primary_q4"][selection].reshape(-1)
                baseline = selected_bundle.semantic_basis["train_edge_prior"][selection].reshape(-1)
                worlds = np.repeat(
                    entity_count * 1_000_000 + selected_bundle.world_ids[selection],
                    selected_bundle.coordinate_count,
                )
                summary = _gain_summary(
                    truth,
                    candidate,
                    baseline,
                    worlds,
                    config,
                    seed=seed + 20 + renderer_id,
                )
                renderer_result[label] = summary
                if renderer_id == 2:
                    passed = (
                        float(summary["brier_gain_ci"]["lower"])
                        > float(gates["unseen_renderer_brier_gain_lower95_strict_gt"])
                    )
                    summary["passed"] = passed
                    unseen_checks.append(passed)
            metrics["renderer_strata"][model.model_id][str(entity_count)] = renderer_result

            selected_bundle = next(bundle for bundle in bundles if predicate(bundle))
            seed_rows: dict[str, Any] = {}
            for seed_index, registered_seed in enumerate(config.section("compiler")["seeds"]):
                seed_rows[str(registered_seed)] = {
                    "semantic_brier": brier_score(
                        selected_bundle.semantic_truth,
                        selected_bundle.semantic_seed_probabilities[seed_index],
                    ),
                    "behavioral_brier": brier_score(
                        selected_bundle.behavioral_truth,
                        selected_bundle.behavioral_seed_probabilities[seed_index],
                    ),
                }
            metrics["compiler_seed_strata"][model.model_id][str(entity_count)] = seed_rows

    hard = set(HARD_COMPOSITION_FAMILIES)
    aggregate_prior = _operation_gain_summary(
        bundles,
        panels,
        config,
        candidate="primary_q4",
        baseline="train_edge_prior",
        families=hard,
        seed=30_000,
    )
    aggregate_direct = _operation_gain_summary(
        bundles,
        panels,
        config,
        candidate="primary_q4",
        baseline="train_selected_direct_protocol",
        families=hard,
        seed=30_001,
    )
    aggregate_prior["passed"] = (
        float(aggregate_prior["brier_gain_ci"]["lower"])
        > float(gates["compiled_prior_gain_lower95_strict_gt"])
    )
    aggregate_direct["passed"] = (
        float(aggregate_direct["brier_gain_ci"]["lower"])
        > float(gates["compiled_direct_gain_lower95_strict_gt"])
    )
    metrics["composition"]["aggregate_over_prior"] = aggregate_prior
    metrics["composition"]["aggregate_over_direct"] = aggregate_direct

    composition_group_checks: list[bool] = [
        bool(aggregate_prior["passed"]), bool(aggregate_direct["passed"])
    ]
    by_model_complexity: dict[str, Any] = {}
    for model_index, model in enumerate(config.models):
        by_model_complexity[model.model_id] = {}
        for complexity_index, entity_count in enumerate(
            config.section("panel")["entity_counts"]
        ):
            predicate = _model_complexity_predicate(model.model_id, entity_count)
            prior = _operation_gain_summary(
                bundles,
                panels,
                config,
                candidate="primary_q4",
                baseline="train_edge_prior",
                families=hard,
                seed=31_000 + 100 * model_index + complexity_index,
                predicate=predicate,
            )
            direct = _operation_gain_summary(
                bundles,
                panels,
                config,
                candidate="primary_q4",
                baseline="train_selected_direct_protocol",
                families=hard,
                seed=32_000 + 100 * model_index + complexity_index,
                predicate=predicate,
            )
            prior_pass = (
                float(prior["brier_gain_ci"]["lower"])
                > float(gates["compiled_prior_gain_lower95_strict_gt"])
            )
            direct_pass = (
                float(direct["brier_gain_ci"]["lower"])
                > float(gates["compiled_direct_gain_lower95_strict_gt"])
            )
            composition_group_checks.extend([prior_pass, direct_pass])
            by_model_complexity[model.model_id][str(entity_count)] = {
                "over_prior": {**prior, "passed": prior_pass},
                "over_direct": {**direct, "passed": direct_pass},
            }
    metrics["composition"]["by_model_and_complexity"] = by_model_complexity

    family_checks: list[bool] = []
    by_family: dict[str, Any] = {}
    for family_index, family in enumerate(sorted(hard)):
        prior = _operation_gain_summary(
            bundles,
            panels,
            config,
            candidate="primary_q4",
            baseline="train_edge_prior",
            families={family},
            seed=35_000 + 2 * family_index,
        )
        direct = _operation_gain_summary(
            bundles,
            panels,
            config,
            candidate="primary_q4",
            baseline="train_selected_direct_protocol",
            families={family},
            seed=35_001 + 2 * family_index,
        )
        prior_pass = (
            float(prior["brier_gain_ci"]["lower"])
            > float(gates["hard_family_gain_lower95_strict_gt"])
        )
        direct_pass = (
            float(direct["brier_gain_ci"]["lower"])
            > float(gates["hard_family_gain_lower95_strict_gt"])
        )
        family_checks.extend([prior_pass, direct_pass])
        by_family[family] = {
            "over_prior": {**prior, "passed": prior_pass},
            "over_direct": {**direct, "passed": direct_pass},
        }
    metrics["composition"]["by_hard_family"] = by_family

    activation_controls = {
        "token_id": ("token_id_q4", "activation_over_token_gain_lower95_strict_gt", True),
        "final_token": (
            "final_token_q4",
            "activation_over_final_token_gain_lower95_strict_gt",
            True,
        ),
        "continuous": (
            "historical_continuous_quotient",
            "activation_over_continuous_gain_lower95_min",
            False,
        ),
    }
    activation_checks: list[bool] = []
    activation_metrics: dict[str, Any] = {}
    for offset, (label, (baseline, gate_name, strict)) in enumerate(
        activation_controls.items()
    ):
        summary = _operation_gain_summary(
            bundles,
            panels,
            config,
            candidate="primary_q4",
            baseline=baseline,
            families=hard,
            seed=40_000 + offset,
        )
        lower = float(summary["brier_gain_ci"]["lower"])
        threshold = float(gates[gate_name])
        passed = lower > threshold if strict else lower >= threshold
        summary["passed"] = passed
        activation_checks.append(passed)
        activation_metrics[label] = summary
    metrics["activation_specificity"] = activation_metrics

    retrieval = _retrieval_metrics(bundles, config)
    retrieval["retrieval_passed"] = (
        float(retrieval["retrieval_ci"]["lower"])
        >= float(gates["min_cross_model_retrieval_lower95"])
    )
    retrieval["margin_passed"] = (
        float(retrieval["wrong_world_margin_ci"]["lower"])
        >= float(gates["min_wrong_world_margin_lower95"])
    )
    identity = _model_identity_metrics(bundles, config, train_identity_basis)
    identity["passed"] = (
        float(identity["over_chance"])
        <= float(gates["max_model_identity_accuracy_over_chance"])
    )
    metrics["public_alignment"] = {"retrieval": retrieval, "model_identity": identity}

    held = config.held_model.model_id
    founders = [model.model_id for model in config.founder_models]
    held_retention: dict[str, Any] = {}
    held_checks: list[bool] = []
    for entity_count in config.section("panel")["entity_counts"]:
        semantic_founder = float(
            np.mean([semantic_gains[(model_id, entity_count)] for model_id in founders])
        )
        behavioral_founder = float(
            np.mean([behavioral_gains[(model_id, entity_count)] for model_id in founders])
        )
        semantic_ratio = (
            semantic_gains[(held, entity_count)] / semantic_founder
            if semantic_founder > 0
            else -1.0
        )
        behavioral_ratio = (
            behavioral_gains[(held, entity_count)] / behavioral_founder
            if behavioral_founder > 0
            else -1.0
        )
        semantic_pass = semantic_ratio >= float(gates["min_held_sender_gain_retention"])
        behavioral_pass = behavioral_ratio >= float(
            gates["min_held_sender_gain_retention"]
        )
        held_checks.extend([semantic_pass, behavioral_pass])
        held_retention[str(entity_count)] = {
            "semantic": {
                "held_gain": semantic_gains[(held, entity_count)],
                "founder_mean_gain": semantic_founder,
                "retention": semantic_ratio,
                "passed": semantic_pass,
            },
            "behavioral": {
                "held_gain": behavioral_gains[(held, entity_count)],
                "founder_mean_gain": behavioral_founder,
                "retention": behavioral_ratio,
                "passed": behavioral_pass,
            },
        }
    metrics["held_sender_retention"] = held_retention

    float_direct = _operation_gain_summary(
        bundles,
        panels,
        config,
        candidate="primary_float",
        baseline="train_selected_direct_protocol",
        families=hard,
        seed=50_000,
    )
    float_gain = float(float_direct["brier_gain"])
    q4_gain = float(aggregate_direct["brier_gain"])
    retention = q4_gain / float_gain if float_gain > 0 else -1.0
    quantization_passed = (
        retention >= float(gates["min_four_bit_compiled_gain_retention"])
        and bool(aggregate_prior["passed"])
        and bool(aggregate_direct["passed"])
    )
    metrics["quantization"] = {
        "float_compiled_over_direct": float_direct,
        "four_bit_compiled_over_direct": aggregate_direct,
        "gain_retention": retention,
        "passed": quantization_passed,
    }

    oracle_mismatches = _oracle_hard_mismatches(bundles, panels)
    oracle_passed = oracle_mismatches <= int(gates["max_oracle_hard_mismatches"])
    metrics["oracle_executor"] = {
        "hard_mismatches": oracle_mismatches,
        "passed": oracle_passed,
    }
    rate_compute = summarize_rate_compute(bundles, config)

    checks = {
        "integrity": bool(integrity_checks) and all(integrity_checks.values()),
        "semantic_basis": bool(semantic_checks) and all(semantic_checks),
        "unseen_renderer": bool(unseen_checks) and all(unseen_checks),
        "behavioral_basis": bool(behavioral_checks) and all(behavioral_checks),
        "activation_specificity": bool(activation_checks) and all(activation_checks),
        "composition": bool(composition_group_checks)
        and all(composition_group_checks)
        and bool(family_checks)
        and all(family_checks),
        "public_alignment": bool(retrieval["retrieval_passed"])
        and bool(retrieval["margin_passed"])
        and bool(identity["passed"]),
        "held_sender": bool(held_checks) and all(held_checks),
        "quantization": quantization_passed,
        "oracle_executor": oracle_passed,
    }
    decision = reduce_stagea_v3_decision(checks, integrity_checks)
    metrics["gate_checks"] = checks
    metrics["integrity_checks"] = integrity_checks
    return metrics, decision, rate_compute


def reduce_stagea_v3_decision(
    checks: dict[str, bool],
    integrity_checks: dict[str, bool],
) -> dict[str, Any]:
    required = {
        "integrity",
        "semantic_basis",
        "unseen_renderer",
        "behavioral_basis",
        "activation_specificity",
        "composition",
        "public_alignment",
        "held_sender",
        "quantization",
        "oracle_executor",
    }
    if set(checks) != required:
        raise ValueError("Stage-A v3 decision checks differ from the frozen registry")
    if not checks["integrity"]:
        diagnosis = "INVALID_STAGEA_V3_RUN"
    elif not checks["semantic_basis"] or not checks["unseen_renderer"]:
        diagnosis = "ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED"
    elif not checks["behavioral_basis"]:
        diagnosis = "BEHAVIORAL_STATE_NOT_QUALIFIED"
    elif not checks["activation_specificity"]:
        diagnosis = "NO_ACTIVATION_SPECIFIC_ADVANTAGE"
    elif (
        not checks["composition"]
        or not checks["public_alignment"]
        or not checks["quantization"]
        or not checks["oracle_executor"]
    ):
        diagnosis = "NO_ONE_SHOT_COMPOSITION_ADVANTAGE"
    elif not checks["held_sender"]:
        diagnosis = "HELD_SENDER_NOT_ESTABLISHED"
    else:
        diagnosis = "STAGEA_V3_REPRESENTATION_QUALIFIED"
    passed = diagnosis == "STAGEA_V3_REPRESENTATION_QUALIFIED"
    return {
        "schema": "frank_eq_stagea_v3_decision_v1",
        "status": "pass" if passed else "fail",
        "diagnosis": diagnosis,
        "checks": checks,
        "integrity_checks": integrity_checks,
        "authorization": {
            "receiver_protocol_draft_authorized": passed,
            "receiver_execution_authorized": False,
            "new_receiver_world_access_authorized": False,
            "scientific_claim_authorized": False,
            "paper_claim_authorized": False,
        },
    }
