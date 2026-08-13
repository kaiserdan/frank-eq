"""Train-role-only SPQ0 selection, cross-family composition, and machine gates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .automaton import ControlledSystem, SharedPredictiveBasis
from .config import SPQRunConfig
from .panel import RENDERER_IDS, ROLE_IDS, SYSTEM_ROLE_IDS
from .probes import (
    LinearMap,
    aggregate_by_group,
    bootstrap_interval,
    brier_score,
    categorical_distribution,
    categorical_expectation,
    deterministic_token_hash_features,
    fit_linear_map,
    nearest_centroid_accuracy,
    paired_brier_gain_interval,
    parameter_matched_token_sequence_features,
    project_simplex,
    quantize_probabilities,
    r2_score,
    row_brier,
    select_categorical_temperature,
    select_target_reader,
    wrong_history_margin_interval,
)

_ACTIVATION_SURFACES = {
    "final_token_residual",
    "event_boundary_residuals",
    "all_token_summary",
}


def condition_masks(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    validation = arrays["role_ids"] == ROLE_IDS["validation"]
    seen_renderer = arrays["renderer_ids"] != RENDERER_IDS["symbolic"]
    unseen_renderer = arrays["renderer_ids"] == RENDERER_IDS["symbolic"]
    fit_system = arrays["system_role_ids"] == SYSTEM_ROLE_IDS["fit"]
    unseen_system = arrays["system_role_ids"] == SYSTEM_ROLE_IDS["validation_only"]
    short = arrays["lengths"] != 32
    long = arrays["lengths"] == 32
    return {
        "aggregate": validation,
        "seen": validation & fit_system & short & seen_renderer,
        "unseen_renderer": validation & fit_system & short & unseen_renderer,
        "unseen_system": validation & unseen_system & short & seen_renderer,
        "length_transfer": validation & fit_system & long & seen_renderer,
        "joint_ood": validation & unseen_system & long & unseen_renderer,
    }


def _select_temperature_and_behavior(
    config: SPQRunConfig,
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    calibration = arrays["role_ids"] == ROLE_IDS["calibration"]
    logits = np.asarray(arrays["categorical_log_likelihoods"], dtype=np.float64)
    semantic = np.concatenate(
        [arrays["semantic_public"], arrays["semantic_targets"]], axis=1
    )
    bins = np.asarray(config.probability_protocol.bins, dtype=np.float64)
    selected = select_categorical_temperature(
        logits[calibration],
        semantic[calibration],
        bins,
        config.probability_protocol.temperature_grid,
    )
    distribution = categorical_distribution(logits, selected["selected_temperature"])
    expectation = categorical_expectation(distribution, bins)
    return distribution, expectation, {
        "fit_role": "calibration",
        "selected_temperature": selected["selected_temperature"],
        "candidates": selected["candidates"],
        "bin_registry": list(config.probability_protocol.bins),
        "candidate_labels": list(config.probability_protocol.candidate_labels),
    }


def _replace_history_ids(
    values: np.ndarray,
    history_ids: np.ndarray,
    strata: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(values)
    identities = np.asarray(history_ids, dtype=np.int64)
    strata_values = np.asarray(strata, dtype=np.int64)
    for stratum in np.unique(strata_values):
        positions = np.flatnonzero(strata_values == stratum)
        unique = np.unique(identities[positions])
        shifted = np.roll(unique, 1)
        mapping = {int(source): int(target) for source, target in zip(unique, shifted, strict=True)}
        for position in positions:
            candidates = np.flatnonzero(
                (identities == mapping[int(identities[position])])
                & (strata_values == stratum)
            )
            result[position] = values[int(candidates[0])]
    return result


def _rotate_renderer_rows(
    values: np.ndarray,
    history_ids: np.ndarray,
    renderer_ids: np.ndarray,
) -> np.ndarray:
    """Swap learned packets across renderer views of the same underlying history."""

    result = np.empty_like(values)
    identities = np.asarray(history_ids, dtype=np.int64)
    renderers = np.asarray(renderer_ids, dtype=np.int64)
    for identity in np.unique(identities):
        positions = np.flatnonzero(identities == identity)
        ordered = positions[np.argsort(renderers[positions], kind="stable")]
        if ordered.size < 2:
            result[ordered] = values[ordered]
        else:
            result[ordered] = values[np.roll(ordered, 1)]
    return result


def _selection_features(
    config: SPQRunConfig,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    final_width = int(arrays["final_token_residual"].shape[-1])
    features = {
        "final_token_residual": np.asarray(arrays["final_token_residual"], dtype=np.float64),
        "event_boundary_residuals": np.asarray(
            arrays["event_boundary_summary"], dtype=np.float64
        ),
        "all_token_summary": np.asarray(arrays["all_token_summary"], dtype=np.float64),
        "mean_input_embedding": np.asarray(
            arrays["mean_input_embedding"], dtype=np.float64
        )[:, None, :],
        "parameter_matched_token_sequence": parameter_matched_token_sequence_features(
            arrays["token_ids"],
            arrays["attention_mask"],
            arrays["event_token_indices"],
            width=final_width,
            decay_grid=config.semantic_encoder.token_sequence_decay_grid,
        )[:, None, :],
    }
    return features


def _fit_surface_candidate(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    layer: int,
    ridge: float,
    method: str,
    coefficient_rank: int,
) -> LinearMap:
    return fit_linear_map(
        features[:, layer],
        targets,
        ridge=ridge,
        method=method,
        maximum_coefficient_rank=coefficient_rank,
    )


def select_semantic_encoder(
    config: SPQRunConfig,
    arrays: Mapping[str, np.ndarray],
    basis: SharedPredictiveBasis,
) -> tuple[dict[str, Any], dict[int, LinearMap], dict[str, np.ndarray]]:
    """Select surface/depth/method/ridge/rank using only the selection role."""

    calibration = arrays["role_ids"] == ROLE_IDS["calibration"]
    selection = arrays["role_ids"] == ROLE_IDS["selection"]
    fit_role = calibration | selection
    surfaces = _selection_features(config, arrays)
    candidates: list[dict[str, Any]] = []
    for surface_name in sorted(_ACTIVATION_SURFACES):
        surface = surfaces[surface_name]
        for layer in range(surface.shape[1]):
            for method in config.semantic_encoder.methods:
                for rank in config.semantic_encoder.rank_grid:
                    targets = arrays["semantic_public"][:, :rank]
                    for ridge in config.semantic_encoder.ridge_grid:
                        encoder = _fit_surface_candidate(
                            surface[calibration],
                            targets[calibration],
                            layer=layer,
                            ridge=float(ridge),
                            method=method,
                            coefficient_rank=min(rank, basis.exact_rank),
                        )
                        prediction = encoder.predict(surface[selection, layer])
                        score = brier_score(targets[selection], prediction)
                        candidates.append(
                            {
                                "surface": surface_name,
                                "layer": layer,
                                "method": method,
                                "rank": rank,
                                "ridge": float(ridge),
                                "selection_brier": score,
                            }
                        )
    selected_by_rank: dict[int, dict[str, Any]] = {}
    encoders: dict[int, LinearMap] = {}
    predictions: dict[str, np.ndarray] = {}
    for rank in config.semantic_encoder.rank_grid:
        selected = min(
            (row for row in candidates if row["rank"] == rank),
            key=lambda row: (
                row["selection_brier"],
                row["surface"],
                row["layer"],
                row["method"],
                row["ridge"],
            ),
        )
        surface = surfaces[selected["surface"]]
        targets = arrays["semantic_public"][:, :rank]
        encoder = _fit_surface_candidate(
            surface[fit_role],
            targets[fit_role],
            layer=int(selected["layer"]),
            ridge=float(selected["ridge"]),
            method=str(selected["method"]),
            coefficient_rank=min(rank, basis.exact_rank),
        )
        encoders[rank] = encoder
        predictions[f"rank_{rank}"] = encoder.predict(
            surface[:, int(selected["layer"])]
        )
        selected_by_rank[rank] = {
            **selected,
            "refit_roles": ["calibration", "selection"],
            "encoder": encoder.metadata(),
        }
    exact_selection = selected_by_rank[basis.exact_rank]
    activation_surface = surfaces[exact_selection["surface"]]
    activation_width = int(activation_surface.shape[-1])
    controls: dict[str, np.ndarray] = {}

    token_sequence = parameter_matched_token_sequence_features(
        arrays["token_ids"],
        arrays["attention_mask"],
        arrays["event_token_indices"],
        width=activation_width,
        decay_grid=config.semantic_encoder.token_sequence_decay_grid,
    )
    token_map = fit_linear_map(
        token_sequence[fit_role],
        arrays["semantic_core"][fit_role],
        ridge=float(exact_selection["ridge"]),
        method="ridge",
    )
    controls["parameter_matched_token_sequence"] = token_map.predict(token_sequence)

    token_hash = deterministic_token_hash_features(
        arrays["token_ids"],
        arrays["attention_mask"],
        width=activation_width,
        position_period=config.semantic_encoder.token_hash_position_period,
    )
    token_hash_map = fit_linear_map(
        token_hash[fit_role],
        arrays["semantic_core"][fit_role],
        ridge=float(exact_selection["ridge"]),
    )
    controls["deterministic_token_hash"] = token_hash_map.predict(token_hash)

    embedding = arrays["mean_input_embedding"]
    embedding_map = fit_linear_map(
        embedding[fit_role],
        arrays["semantic_core"][fit_role],
        ridge=float(exact_selection["ridge"]),
    )
    controls["mean_input_embedding"] = embedding_map.predict(embedding)

    final = arrays["final_token_residual"]
    selected_depth = min(int(exact_selection["layer"]), final.shape[1] - 1)
    final_map = fit_linear_map(
        final[fit_role, selected_depth],
        arrays["semantic_core"][fit_role],
        ridge=float(exact_selection["ridge"]),
    )
    controls["final_token_residual"] = final_map.predict(final[:, selected_depth])

    return (
        {
            "selection_role": "selection",
            "fit_role": "calibration",
            "refit_roles": ["calibration", "selection"],
            "selected_by_rank": {str(rank): row for rank, row in selected_by_rank.items()},
            "candidates": candidates,
            "candidate_count": len(candidates),
            "parameter_match": {
                "activation_input_width": activation_width,
                "token_sequence_input_width": int(token_sequence.shape[1]),
                "activation_learned_parameters": encoders[basis.exact_rank].learned_parameter_count,
                "token_sequence_learned_parameters": token_map.learned_parameter_count,
                "matched": (
                    encoders[basis.exact_rank].learned_parameter_count
                    == token_map.learned_parameter_count
                ),
            },
            "controls": {
                "parameter_matched_token_sequence": token_map.metadata(),
                "deterministic_token_hash": token_hash_map.metadata(),
                "mean_input_embedding": embedding_map.metadata(),
                "final_token_residual": final_map.metadata(),
            },
        },
        encoders,
        {**predictions, **{f"control__{key}": value for key, value in controls.items()}},
    )


def _system_rows(
    values: np.ndarray,
    system_ids: np.ndarray,
    mapping: Mapping[str, np.ndarray],
    *,
    columns: int,
) -> np.ndarray:
    result = np.empty((len(values), columns), dtype=np.float64)
    for system_id, matrix in mapping.items():
        index = int(system_id.split("-")[-1])
        mask = system_ids == index
        result[mask] = values[mask] @ matrix
    return result


def decode_core_rows(
    basis: SharedPredictiveBasis,
    packet: np.ndarray,
    system_ids: np.ndarray,
    *,
    rank: int,
) -> np.ndarray:
    matrices = {
        system_id: np.linalg.pinv(public[:, :rank], rcond=1e-12)
        @ public[:, : basis.exact_rank]
        for system_id, public in basis.public_matrices.items()
    }
    return _system_rows(packet, system_ids, matrices, columns=basis.exact_rank)


def execute_target_rows(
    basis: SharedPredictiveBasis,
    packet: np.ndarray,
    system_ids: np.ndarray,
    *,
    rank: int,
) -> np.ndarray:
    return np.clip(
        _system_rows(
            packet,
            system_ids,
            basis.executors[rank],
            columns=len(basis.target_tests),
        ),
        0.0,
        1.0,
    )


def _summary(
    targets: np.ndarray,
    candidate: np.ndarray,
    baselines: Mapping[str, np.ndarray],
    history_ids: np.ndarray,
    *,
    config: SPQRunConfig,
    seed_offset: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": int(len(targets)),
        "histories": int(len(np.unique(history_ids))),
        "candidate_brier": brier_score(targets, candidate),
        "candidate_r2": r2_score(targets, candidate),
        "baselines": {},
    }
    for offset, name in enumerate(sorted(baselines)):
        baseline = baselines[name]
        result["baselines"][name] = {
            "brier": brier_score(targets, baseline),
            "candidate_gain_ci": paired_brier_gain_interval(
                targets,
                candidate,
                baseline,
                history_ids,
                replicates=config.evaluation.bootstrap_replicates,
                seed=config.evaluation.bootstrap_seed + seed_offset + offset,
            ),
        }
    return result


def _heuristic_core_controls(
    arrays: Mapping[str, np.ndarray],
    basis: SharedPredictiveBasis,
    systems: tuple[ControlledSystem, ...],
) -> dict[str, np.ndarray]:
    system_lookup = {int(system.system_id.split("-")[-1]): system for system in systems}
    last_core = np.empty_like(arrays["semantic_core"], dtype=np.float64)
    empirical_core = np.empty_like(arrays["semantic_core"], dtype=np.float64)
    for row in range(len(last_core)):
        system = system_lookup[int(arrays["system_ids"][row])]
        last = int(arrays["last_observations"][row])
        last_belief = system.emissions[:, last] * system.initial_belief
        last_belief /= last_belief.sum()
        last_core[row] = basis.public_probabilities(
            system.system_id, last_belief, rank=basis.exact_rank
        )
        frequency = arrays["observation_frequencies"][row]
        emission_fit = np.linalg.lstsq(system.emissions.T, frequency, rcond=1e-12)[0]
        empirical_belief = np.clip(emission_fit, 1e-9, None)
        empirical_belief /= empirical_belief.sum()
        empirical_core[row] = basis.public_probabilities(
            system.system_id, empirical_belief, rank=basis.exact_rank
        )
    return {"last_observation_filter": last_core, "empirical_observation_filter": empirical_core}


def _fit_gcca_residual(
    config: SPQRunConfig,
    captures: Mapping[str, Mapping[str, np.ndarray]],
    evaluations: Mapping[str, ModelEvaluation],
) -> dict[str, Any]:
    """Fit a non-promotional common residual subspace in public test/bin coordinates."""

    model_ids = sorted(evaluations)
    if len(model_ids) != 2:
        raise ValueError("SPQ0 residual census is frozen to two founders")
    reference = captures[model_ids[0]]
    for model_id in model_ids[1:]:
        current = captures[model_id]
        for key in ("history_ids", "renderer_ids", "role_ids"):
            if not np.array_equal(reference[key], current[key]):
                raise ValueError("behavioral residual rows are not aligned across founders")

    calibration = reference["role_ids"] == ROLE_IDS["calibration"]
    selection = reference["role_ids"] == ROLE_IDS["selection"]
    validation = reference["role_ids"] == ROLE_IDS["validation"]
    joint = (
        validation
        & (reference["system_role_ids"] == SYSTEM_ROLE_IDS["validation_only"])
        & (reference["lengths"] == 32)
        & (reference["renderer_ids"] == RENDERER_IDS["symbolic"])
    )
    predicted = {
        model_id: evaluations[model_id]
        .predictions["predicted_behavioral_residual"]
        .reshape(len(reference["history_ids"]), -1)
        for model_id in model_ids
    }
    means = {
        model_id: predicted[model_id][calibration].mean(axis=0) for model_id in model_ids
    }
    pooled = np.concatenate(
        [predicted[model_id][calibration] - means[model_id] for model_id in model_ids],
        axis=0,
    )
    _, singular_values, components = np.linalg.svd(pooled, full_matrices=False)

    def transferred_residual(source_id: str, target_id: str, rank: int) -> np.ndarray:
        if rank == 0:
            return np.zeros_like(predicted[source_id])
        basis = components[:rank]
        centered = predicted[source_id] - means[source_id]
        return centered @ basis.T @ basis + means[target_id]

    candidates: list[dict[str, Any]] = []
    for rank in config.behavioral_residual.rank_grid:
        direction_gains: dict[str, float] = {}
        for source_id in model_ids:
            for target_id in model_ids:
                if source_id == target_id:
                    continue
                base = evaluations[target_id].predictions[
                    "target_reader_oracle_prediction"
                ]
                truth = evaluations[target_id].behavior_signatures
                residual = transferred_residual(source_id, target_id, rank).reshape(
                    base.shape
                )
                corrected = project_simplex(base + residual)
                direction_gains[f"{source_id}__to__{target_id}"] = (
                    brier_score(truth[selection], base[selection])
                    - brier_score(truth[selection], corrected[selection])
                )
        candidates.append(
            {
                "rank": rank,
                "selection_incremental_gain": float(
                    np.mean(list(direction_gains.values()))
                ),
                "ordered_direction_gains": direction_gains,
                "singular_values": singular_values[:rank].tolist(),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (row["selection_incremental_gain"], -row["rank"]),
    )
    selected_rank = int(selected["rank"])
    validation_results: dict[str, Any] = {}
    for direction_offset, source_id in enumerate(model_ids):
        for target_id in model_ids:
            if source_id == target_id:
                continue
            base = evaluations[target_id].predictions["target_reader_oracle_prediction"]
            truth = evaluations[target_id].behavior_signatures
            residual = transferred_residual(source_id, target_id, selected_rank).reshape(
                base.shape
            )
            corrected = project_simplex(base + residual)
            direction = f"{source_id}__to__{target_id}"
            validation_results[direction] = {
                "pair_specific_mapper": False,
                "source_local_residual_encoder": True,
                "validation_incremental_gain_ci": paired_brier_gain_interval(
                    truth[validation],
                    corrected[validation],
                    base[validation],
                    reference["history_ids"][validation],
                    replicates=config.evaluation.bootstrap_replicates,
                    seed=(
                        config.evaluation.bootstrap_seed
                        + 30000
                        + direction_offset * 100
                    ),
                ),
                "joint_ood_incremental_gain_ci": paired_brier_gain_interval(
                    truth[joint],
                    corrected[joint],
                    base[joint],
                    reference["history_ids"][joint],
                    replicates=config.evaluation.bootstrap_replicates,
                    seed=(
                        config.evaluation.bootstrap_seed
                        + 31000
                        + direction_offset * 100
                    ),
                ),
            }
    return {
        "schema": "frank_eq_spq0_behavioral_residual_census_v1",
        "method": "maxvar_gcca_public_coordinate_subspace",
        "fit_role": "calibration",
        "selection_role": "selection",
        "evaluation_role": "validation",
        "promotional": False,
        "models": model_ids,
        "source_local_residual_encoders": True,
        "pair_specific_mapper_count": 0,
        "candidates": candidates,
        "selected_rank": selected_rank,
        "selected_incremental_gain": selected["selection_incremental_gain"],
        "validation": validation_results,
        "interpretation": "diagnostic_only_after_conditioning_on_semantic_core",
    }


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    metrics: dict[str, Any]
    training: dict[str, Any]
    predictions: dict[str, np.ndarray]
    checkpoint_arrays: dict[str, np.ndarray]
    target_reader: Any
    behavior_signatures: np.ndarray


def evaluate_model(
    config: SPQRunConfig,
    systems: tuple[ControlledSystem, ...],
    basis: SharedPredictiveBasis,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    seed_offset: int,
) -> ModelEvaluation:
    behavior_distribution, behavior_expectation, temperature = (
        _select_temperature_and_behavior(config, arrays)
    )
    encoder_training, encoders, predictions = select_semantic_encoder(
        config, arrays, basis
    )
    if not encoder_training["parameter_match"]["matched"]:
        raise RuntimeError("activation and token-sequence controls are not parameter matched")
    exact_rank = basis.exact_rank
    packet = predictions[f"rank_{exact_rank}"]
    decoded_core = decode_core_rows(
        basis, packet, arrays["system_ids"], rank=exact_rank
    )
    compiled_targets = execute_target_rows(
        basis, packet, arrays["system_ids"], rank=exact_rank
    )
    calibration = arrays["role_ids"] == ROLE_IDS["calibration"]
    selection = arrays["role_ids"] == ROLE_IDS["selection"]
    target_start = len(basis.public_tests)
    target_behavior = behavior_distribution[:, target_start:]
    reader, reader_training = select_target_reader(
        arrays["semantic_core"],
        arrays["semantic_targets"],
        target_behavior,
        calibration,
        selection,
        basis.target_tests,
        config.target_local_reader.ridge_grid,
    )
    native_probability = behavior_expectation[:, target_start:]
    conditions = condition_masks(arrays)
    fit_role = calibration | selection
    core_prior = np.repeat(
        arrays["semantic_core"][fit_role].mean(axis=0, keepdims=True),
        len(packet),
        axis=0,
    )
    target_prior = np.repeat(
        arrays["semantic_targets"][fit_role].mean(axis=0, keepdims=True),
        len(packet),
        axis=0,
    )
    heuristic = _heuristic_core_controls(arrays, basis, systems)
    wrong_history_core = _replace_history_ids(
        arrays["semantic_core"],
        arrays["history_ids"],
        arrays["system_ids"] * 100 + arrays["lengths"],
    )
    shuffled_history = _replace_history_ids(
        decoded_core,
        arrays["history_ids"],
        arrays["system_ids"] * 100 + arrays["lengths"],
    )
    renderer_shuffled = _rotate_renderer_rows(
        decoded_core,
        arrays["history_ids"],
        arrays["renderer_ids"],
    )
    core_metrics: dict[str, Any] = {}
    compiled_metrics: dict[str, Any] = {}
    source_protocol_metrics: dict[str, Any] = {}
    for condition_offset, condition in enumerate(config.evaluation.required_conditions):
        mask = conditions[condition]
        if not np.any(mask):
            raise RuntimeError(f"SPQ0 condition {condition} has no validation rows")
        history = arrays["history_ids"][mask]
        core_metrics[condition] = _summary(
            arrays["semantic_core"][mask],
            decoded_core[mask],
            {
                "history_prior": core_prior[mask],
                "parameter_matched_token_sequence": predictions[
                    "control__parameter_matched_token_sequence"
                ][mask],
                "deterministic_token_hash": predictions[
                    "control__deterministic_token_hash"
                ][mask],
                "mean_input_embedding": predictions["control__mean_input_embedding"][mask],
                "final_token_residual": predictions["control__final_token_residual"][mask],
                "last_observation_filter": heuristic["last_observation_filter"][mask],
                "empirical_observation_filter": heuristic[
                    "empirical_observation_filter"
                ][mask],
                "wrong_history": wrong_history_core[mask],
                "shuffled_history": shuffled_history[mask],
                "renderer_shuffled": renderer_shuffled[mask],
                "zero_packet": np.zeros_like(decoded_core[mask]),
                "exact_bayes_filter": arrays["semantic_core"][mask],
            },
            history,
            config=config,
            seed_offset=seed_offset + condition_offset * 100,
        )
        core_metrics[condition]["wrong_history_margin_ci"] = (
            wrong_history_margin_interval(
                arrays["semantic_core"][mask],
                decoded_core[mask],
                history,
                arrays["system_ids"][mask] * 100 + arrays["lengths"][mask],
                replicates=config.evaluation.bootstrap_replicates,
                seed=config.evaluation.bootstrap_seed + seed_offset + 9000 + condition_offset,
            )
        )
        compiled_metrics[condition] = _summary(
            arrays["semantic_targets"][mask],
            compiled_targets[mask],
            {
                "target_prior": target_prior[mask],
                "direct_probability_forecast": native_probability[mask],
            },
            history,
            config=config,
            seed_offset=seed_offset + 1000 + condition_offset * 100,
        )
        source_protocol_metrics[condition] = _summary(
            arrays["semantic_targets"][mask],
            native_probability[mask],
            {"target_prior": target_prior[mask]},
            history,
            config=config,
            seed_offset=seed_offset + 2000 + condition_offset * 100,
        )

    rank_sweep: dict[str, Any] = {}
    joint = conditions["joint_ood"]
    for rank in config.semantic_encoder.rank_grid:
        rank_packet = predictions[f"rank_{rank}"]
        rank_core = decode_core_rows(basis, rank_packet, arrays["system_ids"], rank=rank)
        rank_targets = execute_target_rows(
            basis, rank_packet, arrays["system_ids"], rank=rank
        )
        rank_sweep[str(rank)] = {
            "selection": encoder_training["selected_by_rank"][str(rank)],
            "joint_ood_core_brier": brier_score(
                arrays["semantic_core"][joint], rank_core[joint]
            ),
            "joint_ood_target_brier": brier_score(
                arrays["semantic_targets"][joint], rank_targets[joint]
            ),
            "exact_public_rank": rank == exact_rank,
            "formally_undercomplete": rank < exact_rank,
            "formally_rank_complete": rank >= exact_rank,
        }

    quantization: dict[str, Any] = {}
    float_targets = compiled_targets[joint]
    float_prior_gain = brier_score(arrays["semantic_targets"][joint], target_prior[joint]) - brier_score(
        arrays["semantic_targets"][joint], float_targets
    )
    for bits in config.evaluation.quantization_bits:
        quantized = quantize_probabilities(packet, bits)
        quantized_targets = execute_target_rows(
            basis, quantized, arrays["system_ids"], rank=exact_rank
        )
        gain = brier_score(
            arrays["semantic_targets"][joint], target_prior[joint]
        ) - brier_score(arrays["semantic_targets"][joint], quantized_targets[joint])
        retention = 1.0 if abs(float_prior_gain) <= 1e-15 else gain / float_prior_gain
        quantization[str(bits)] = {
            "payload_bits": bits * exact_rank,
            "joint_ood_target_brier": brier_score(
                arrays["semantic_targets"][joint], quantized_targets[joint]
            ),
            "gain_over_prior": gain,
            "gain_retention": float(retention),
        }

    amortized: dict[str, Any] = {}
    reusable_losses = row_brier(
        arrays["semantic_targets"][joint], compiled_targets[joint]
    )
    direct_losses = row_brier(
        arrays["semantic_targets"][joint], native_probability[joint]
    )
    bits = 4 * exact_rank
    for count in config.evaluation.amortized_future_query_counts:
        packet_cost = config.evaluation.packet_bit_cost_brier_equivalent * bits / count
        direct_cost = config.evaluation.source_query_cost_brier_equivalent
        reusable_value = -(reusable_losses + packet_cost)
        direct_value = -(direct_losses + direct_cost)
        gain = reusable_value - direct_value
        _, grouped = aggregate_by_group(gain, arrays["history_ids"][joint])
        amortized[str(count)] = {
            "future_queries": count,
            "packet_payload_bits": bits,
            "packet_bits_per_query": bits / count,
            "source_queries_per_query_packet": 0.0,
            "source_queries_per_query_direct": 1.0,
            "packet_cost_brier_equivalent_per_query": packet_cost,
            "direct_query_cost_brier_equivalent_per_query": direct_cost,
            "utility_gain_over_direct_ci": bootstrap_interval(
                grouped,
                replicates=config.evaluation.bootstrap_replicates,
                seed=config.evaluation.bootstrap_seed + seed_offset + 12000 + count,
            ),
        }

    predicted_reader = reader.predict(decoded_core, compiled_targets, basis.target_tests)
    oracle_reader_prediction = reader.predict(
        arrays["semantic_core"], arrays["semantic_targets"], basis.target_tests
    )
    residual = target_behavior - oracle_reader_prediction
    selected_surface = encoder_training["selected_by_rank"][str(exact_rank)]
    residual_surface = _selection_features(config, arrays)[selected_surface["surface"]][
        :, int(selected_surface["layer"])
    ]
    residual_candidates: list[dict[str, Any]] = [
        {
            "rank": 0,
            "ridge": None,
            "selection_brier": brier_score(
                residual[selection], np.zeros_like(residual[selection])
            ),
        }
    ]
    for residual_rank in config.behavioral_residual.rank_grid:
        if residual_rank == 0:
            continue
        for ridge in config.semantic_encoder.ridge_grid:
            local_encoder = fit_linear_map(
                residual_surface[calibration],
                residual[calibration].reshape(int(calibration.sum()), -1),
                ridge=float(ridge),
                method="reduced_rank_regression",
                maximum_coefficient_rank=residual_rank,
            )
            prediction = local_encoder.predict(
                residual_surface[selection], clip=False
            ).reshape(residual[selection].shape)
            residual_candidates.append(
                {
                    "rank": residual_rank,
                    "ridge": float(ridge),
                    "selection_brier": brier_score(residual[selection], prediction),
                }
            )
    selected_residual = min(
        residual_candidates,
        key=lambda row: (
            row["selection_brier"],
            row["rank"],
            -1.0 if row["ridge"] is None else row["ridge"],
        ),
    )
    if selected_residual["rank"] == 0:
        predicted_residual = np.zeros_like(residual)
        residual_encoder = None
    else:
        residual_encoder = fit_linear_map(
            residual_surface[fit_role],
            residual[fit_role].reshape(int(fit_role.sum()), -1),
            ridge=float(selected_residual["ridge"]),
            method="reduced_rank_regression",
            maximum_coefficient_rank=int(selected_residual["rank"]),
        )
        predicted_residual = residual_encoder.predict(
            residual_surface, clip=False
        ).reshape(residual.shape)
    checkpoint_arrays: dict[str, np.ndarray] = {}
    for rank, encoder in encoders.items():
        checkpoint_arrays.update(encoder.arrays(f"semantic_rank_{rank}"))
    checkpoint_arrays.update(reader.arrays("target_reader"))
    if residual_encoder is not None:
        checkpoint_arrays.update(residual_encoder.arrays("behavioral_residual_encoder"))
    checkpoint_arrays["target_reader_temperature"] = np.asarray(
        [temperature["selected_temperature"]], dtype=np.float64
    )
    training = {
        "model_id": metadata["model_id"],
        "family": metadata["family"],
        "probability_protocol": temperature,
        "semantic_encoder": encoder_training,
        "target_local_reader": {
            **reader_training,
            "reader": reader.metadata(),
            "frozen_before_source_evaluation": True,
        },
        "behavioral_residual_encoder": {
            "fit_role": "calibration",
            "selection_role": "selection",
            "refit_roles": ["calibration", "selection"],
            "promotional": False,
            "selected": selected_residual,
            "candidates": residual_candidates,
            "encoder": None if residual_encoder is None else residual_encoder.metadata(),
        },
    }
    metrics = {
        "model_id": metadata["model_id"],
        "family": metadata["family"],
        "source_probability_protocol": source_protocol_metrics,
        "semantic_core": core_metrics,
        "compiled_semantic_targets": compiled_metrics,
        "rank_sweep": rank_sweep,
        "quantization": quantization,
        "amortized_utility": amortized,
        "target_reader_native_brier": brier_score(target_behavior, predicted_reader),
        "target_reader_oracle_brier": brier_score(
            target_behavior,
            reader.predict(
                arrays["semantic_core"], arrays["semantic_targets"], basis.target_tests
            ),
        ),
    }
    prediction_payload = {
        **predictions,
        "decoded_core": decoded_core,
        "compiled_targets": compiled_targets,
        "categorical_behavior": behavior_distribution,
        "categorical_expectation": behavior_expectation,
        "target_reader_from_source": predicted_reader,
        "target_reader_residual": residual,
        "target_reader_oracle_prediction": oracle_reader_prediction,
        "predicted_behavioral_residual": predicted_residual,
    }
    return ModelEvaluation(
        metrics=metrics,
        training=training,
        predictions=prediction_payload,
        checkpoint_arrays=checkpoint_arrays,
        target_reader=reader,
        behavior_signatures=target_behavior,
    )


def _rows_by_history(
    arrays: Mapping[str, np.ndarray],
) -> dict[tuple[int, int, int], int]:
    result: dict[tuple[int, int, int], int] = {}
    for index, key in enumerate(
        zip(
            arrays["history_ids"].tolist(),
            arrays["renderer_ids"].tolist(),
            arrays["role_ids"].tolist(),
            strict=True,
        )
    ):
        if key in result:
            raise ValueError("SPQ0 capture contains a duplicate history/renderer/role row")
        result[key] = index
    return result


def _align_source_to_target(
    source_arrays: Mapping[str, np.ndarray],
    target_arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    source = _rows_by_history(source_arrays)
    target = _rows_by_history(target_arrays)
    if set(source) != set(target):
        raise ValueError("cross-model source/target capture rows differ")
    keys = sorted(source)
    return (
        np.asarray([source[key] for key in keys], dtype=np.int64),
        np.asarray([target[key] for key in keys], dtype=np.int64),
    )


def _cross_family_compositions(
    config: SPQRunConfig,
    basis: SharedPredictiveBasis,
    captures: Mapping[str, Mapping[str, np.ndarray]],
    evaluations: Mapping[str, ModelEvaluation],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pairs: dict[str, Any] = {}
    prediction_arrays: dict[str, np.ndarray] = {}
    model_ids = sorted(evaluations)
    for source_id in model_ids:
        for target_id in model_ids:
            if source_id == target_id:
                continue
            source_arrays = captures[source_id]
            target_arrays = captures[target_id]
            source_rows, target_rows = _align_source_to_target(source_arrays, target_arrays)
            source_packet = evaluations[source_id].predictions["decoded_core"][source_rows]
            source_semantic_targets = evaluations[source_id].predictions[
                "compiled_targets"
            ][source_rows]
            reader = evaluations[target_id].target_reader
            transferred = reader.predict(
                source_packet,
                source_semantic_targets,
                basis.target_tests,
            )
            oracle = reader.predict(
                target_arrays["semantic_core"][target_rows],
                target_arrays["semantic_targets"][target_rows],
                basis.target_tests,
            )
            quantized_core = quantize_probabilities(source_packet, 4)
            quantized_targets = execute_target_rows(
                basis,
                quantized_core,
                source_arrays["system_ids"][source_rows],
                rank=basis.exact_rank,
            )
            quantized_transfer = reader.predict(
                quantized_core,
                quantized_targets,
                basis.target_tests,
            )
            truth = evaluations[target_id].behavior_signatures[target_rows]
            validation = target_arrays["role_ids"][target_rows] == ROLE_IDS["validation"]
            joint = (
                validation
                & (target_arrays["system_role_ids"][target_rows] == SYSTEM_ROLE_IDS["validation_only"])
                & (target_arrays["lengths"][target_rows] == 32)
                & (target_arrays["renderer_ids"][target_rows] == RENDERER_IDS["symbolic"])
            )
            fit = target_arrays["role_ids"][target_rows] != ROLE_IDS["validation"]
            prior = np.repeat(truth[fit].mean(axis=0, keepdims=True), len(truth), axis=0)
            gain_ci = paired_brier_gain_interval(
                truth[joint],
                transferred[joint],
                prior[joint],
                target_arrays["history_ids"][target_rows][joint],
                replicates=config.evaluation.bootstrap_replicates,
                seed=config.evaluation.bootstrap_seed + 20000 + len(pairs),
            )
            prior_brier = brier_score(truth[joint], prior[joint])
            transfer_brier = brier_score(truth[joint], transferred[joint])
            oracle_brier = brier_score(truth[joint], oracle[joint])
            denominator = prior_brier - oracle_brier
            retention = 0.0 if denominator <= 1e-15 else (prior_brier - transfer_brier) / denominator
            float_gain = prior_brier - transfer_brier
            quantized_brier = brier_score(truth[joint], quantized_transfer[joint])
            quantized_gain = prior_brier - quantized_brier
            four_bit_retention = (
                1.0 if abs(float_gain) <= 1e-15 else quantized_gain / float_gain
            )
            pair_name = f"{source_id}__to__{target_id}"
            pairs[pair_name] = {
                "source_model": source_id,
                "source_family": evaluations[source_id].metrics["family"],
                "target_model": target_id,
                "target_family": evaluations[target_id].metrics["family"],
                "pair_specific_mapper": False,
                "target_reader_frozen_before_source_evaluation": True,
                "condition": "joint_ood",
                "rows": int(joint.sum()),
                "histories": int(
                    len(np.unique(target_arrays["history_ids"][target_rows][joint]))
                ),
                "transferred_brier": transfer_brier,
                "target_prior_brier": prior_brier,
                "oracle_core_reader_brier": oracle_brier,
                "gain_over_target_prior_ci": gain_ci,
                "oracle_reader_gain_retention": float(retention),
                "four_bit_transferred_brier": quantized_brier,
                "four_bit_cross_family_gain_retention": float(four_bit_retention),
            }
            prediction_arrays[f"{pair_name}__transferred"] = transferred
            prediction_arrays[f"{pair_name}__oracle"] = oracle
            prediction_arrays[f"{pair_name}__four_bit"] = quantized_transfer
    return pairs, prediction_arrays


def _sender_identity(
    captures: Mapping[str, Mapping[str, np.ndarray]],
    evaluations: Mapping[str, ModelEvaluation],
) -> dict[str, Any]:
    train_packets: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    validation_packets: list[np.ndarray] = []
    validation_labels: list[np.ndarray] = []
    for label, model_id in enumerate(sorted(evaluations)):
        arrays = captures[model_id]
        packet = evaluations[model_id].predictions["decoded_core"]
        train = arrays["role_ids"] != ROLE_IDS["validation"]
        validation = arrays["role_ids"] == ROLE_IDS["validation"]
        train_packets.append(packet[train])
        train_labels.append(np.full(int(train.sum()), label, dtype=np.int64))
        validation_packets.append(packet[validation])
        validation_labels.append(np.full(int(validation.sum()), label, dtype=np.int64))
    accuracy = nearest_centroid_accuracy(
        np.concatenate(train_packets),
        np.concatenate(train_labels),
        np.concatenate(validation_packets),
        np.concatenate(validation_labels),
    )
    chance = 1.0 / len(evaluations)
    return {
        "method": "train_centroid_validation_accuracy",
        "accuracy": accuracy,
        "chance": chance,
        "accuracy_over_chance": accuracy - chance,
    }


def evaluate_all_models(
    config: SPQRunConfig,
    systems: tuple[ControlledSystem, ...],
    basis: SharedPredictiveBasis,
    captures: Mapping[str, tuple[Mapping[str, np.ndarray], Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    evaluations: dict[str, ModelEvaluation] = {}
    arrays_by_model: dict[str, Mapping[str, np.ndarray]] = {}
    for offset, model_id in enumerate(sorted(captures)):
        arrays, metadata = captures[model_id]
        arrays_by_model[model_id] = arrays
        evaluations[model_id] = evaluate_model(
            config,
            systems,
            basis,
            arrays,
            metadata,
            seed_offset=offset * 100_000,
        )
    cross_family, transfer_predictions = _cross_family_compositions(
        config, basis, arrays_by_model, evaluations
    )
    residual_census = _fit_gcca_residual(
        config,
        arrays_by_model,
        evaluations,
    )
    metrics = {
        "schema": "frank_eq_spq0_metrics_v1",
        "scope": "development-only shared predictive quotient census",
        "public_basis": {
            "exact_rank": basis.exact_rank,
            "rank_grid": config.semantic_encoder.rank_grid,
            "core_condition_numbers": dict(basis.core_condition_numbers),
            "maximum_target_l1": basis.maximum_target_l1,
            "maximum_exact_executor_error": basis.maximum_exact_executor_error,
        },
        "models": {model_id: evaluation.metrics for model_id, evaluation in evaluations.items()},
        "cross_family_composition": cross_family,
        "sender_identity": _sender_identity(arrays_by_model, evaluations),
        "behavioral_residual_census": residual_census,
        "data_usage": {
            "roles": ["calibration", "selection", "validation"],
            "test_histories": 0,
            "held_sender_rows": 0,
            "receiver_rows": 0,
        },
    }
    training = {
        "schema": "frank_eq_spq0_training_v1",
        "models": {model_id: evaluation.training for model_id, evaluation in evaluations.items()},
        "behavioral_residual_census": residual_census,
        "pair_specific_mapper_count": 0,
        "target_readers_frozen_before_source_evaluation": True,
    }
    predictions = {
        model_id: evaluation.predictions for model_id, evaluation in evaluations.items()
    }
    predictions["cross_family"] = transfer_predictions
    checkpoints = {
        model_id: evaluation.checkpoint_arrays for model_id, evaluation in evaluations.items()
    }
    return metrics, training, predictions, checkpoints


def gate_decision(config: SPQRunConfig, metrics: Mapping[str, Any]) -> dict[str, Any]:
    gate = config.gates
    model_metrics = metrics["models"]
    exact = metrics["public_basis"]
    system_valid = (
        exact["exact_rank"] == config.systems.predictive_rank
        and exact["maximum_exact_executor_error"] <= gate.max_oracle_executor_abs_error
        and max(exact["core_condition_numbers"].values())
        <= config.systems.core_condition_number_max
    )
    source_protocol = {
        model_id: all(
            row["baselines"]["target_prior"]["candidate_gain_ci"]["lower"]
            >= gate.source_probability_gain_over_prior_lower95_min
            for row in model["source_probability_protocol"].values()
        )
        for model_id, model in model_metrics.items()
    }
    semantic_readability = {
        f"{model_id}|{condition}": (
            model["semantic_core"][condition]["baselines"]["history_prior"]
            ["candidate_gain_ci"]["lower"]
            >= gate.semantic_core_gain_over_prior_lower95_min
        )
        for model_id, model in model_metrics.items()
        for condition in config.evaluation.required_conditions
    }
    activation_specificity = {
        model_id: (
            model["semantic_core"]["joint_ood"]["baselines"]
            ["parameter_matched_token_sequence"]["candidate_gain_ci"]["lower"]
            > gate.activation_over_token_sequence_lower95_strict_gt
        )
        for model_id, model in model_metrics.items()
    }
    history_specificity = {
        model_id: (
            model["semantic_core"]["joint_ood"]["wrong_history_margin_ci"]["lower"]
            > gate.wrong_history_margin_lower95_strict_gt
        )
        for model_id, model in model_metrics.items()
    }
    transfer_stability = {
        model_id: all(
            model["semantic_core"][condition]["baselines"]["history_prior"]
            ["candidate_gain_ci"]["lower"]
            >= gate.semantic_core_gain_over_prior_lower95_min
            for condition in ("unseen_renderer", "unseen_system", "length_transfer", "joint_ood")
        )
        for model_id, model in model_metrics.items()
    }
    cross_family = {
        pair: (
            row["gain_over_target_prior_ci"]["lower"]
            > gate.cross_family_target_prior_gain_lower95_strict_gt
            and row["oracle_reader_gain_retention"]
            >= gate.min_cross_family_oracle_reader_gain_retention
            and row["pair_specific_mapper"] is False
            and row["target_reader_frozen_before_source_evaluation"] is True
        )
        for pair, row in metrics["cross_family_composition"].items()
    }
    rank_identified: dict[str, bool] = {}
    for model_id, model in model_metrics.items():
        sweep = model["rank_sweep"]
        rank4 = float(sweep["4"]["joint_ood_target_brier"])
        higher = [float(sweep[str(rank)]["joint_ood_target_brier"]) for rank in (6, 8)]
        lower = [float(sweep[str(rank)]["joint_ood_target_brier"]) for rank in (1, 2, 3)]
        rank_identified[model_id] = (
            rank4 <= min(higher) + 1e-12 and min(lower) > rank4
            if gate.rank4_noninferior_to_higher_ranks
            else True
        )
    quantization = {
        pair: (
            row["four_bit_cross_family_gain_retention"]
            >= gate.min_four_bit_gain_retention
        )
        for pair, row in metrics["cross_family_composition"].items()
    }
    identity = (
        metrics["sender_identity"]["accuracy_over_chance"]
        <= gate.max_sender_identity_accuracy_over_chance
    )
    amortized = {
        model_id: (
            model["amortized_utility"][
                str(gate.amortized_query_count_for_primary_utility)
            ]["utility_gain_over_direct_ci"]["lower"]
            > gate.amortized_utility_lower95_strict_gt
        )
        for model_id, model in model_metrics.items()
    }
    checks = {
        "predictive_system_and_executor": system_valid,
        "source_probability_protocol": bool(source_protocol) and all(source_protocol.values()),
        "semantic_predictive_quotient": bool(semantic_readability)
        and all(semantic_readability.values()),
        "activation_specificity": bool(activation_specificity)
        and all(activation_specificity.values()),
        "history_specificity": bool(history_specificity) and all(history_specificity.values()),
        "renderer_system_length_stability": bool(transfer_stability)
        and all(transfer_stability.values()),
        "cross_family_public_state_transfer": bool(cross_family) and all(cross_family.values()),
        "transfer_rank_identified": bool(rank_identified) and all(rank_identified.values()),
        "four_bit_retention": bool(quantization) and all(quantization.values()),
        "sender_identity_closed": identity,
        "amortized_rate_utility": bool(amortized) and all(amortized.values()),
    }
    passed = all(checks.values())
    if not checks["predictive_system_and_executor"]:
        diagnosis = "PREDICTIVE_SYSTEM_OR_EXECUTOR_INVALID"
    elif not checks["source_probability_protocol"]:
        diagnosis = "SOURCE_PROBABILITY_PROTOCOL_NOT_QUALIFIED"
    elif not checks["semantic_predictive_quotient"]:
        diagnosis = "SEMANTIC_PREDICTIVE_QUOTIENT_NOT_READABLE"
    elif not checks["activation_specificity"] or not checks["history_specificity"]:
        diagnosis = "NO_ACTIVATION_SPECIFIC_PREDICTIVE_ADVANTAGE"
    elif not checks["renderer_system_length_stability"]:
        diagnosis = "PREDICTIVE_QUOTIENT_NOT_RENDERER_OR_LENGTH_STABLE"
    elif not checks["cross_family_public_state_transfer"]:
        diagnosis = "NO_CROSS_FAMILY_PUBLIC_STATE_TRANSFER"
    elif not checks["transfer_rank_identified"]:
        diagnosis = "TRANSFER_RANK_NOT_IDENTIFIED"
    elif not (
        checks["four_bit_retention"]
        and checks["sender_identity_closed"]
        and checks["amortized_rate_utility"]
    ):
        diagnosis = "NO_CROSS_FAMILY_PUBLIC_STATE_TRANSFER"
    else:
        diagnosis = "PUBLIC_PREDICTIVE_QUOTIENT_CANDIDATE_SUPPORTED"
    return {
        "schema": "frank_eq_spq0_decision_v1",
        "status": "pass" if passed else "fail",
        "diagnosis": diagnosis,
        "checks": checks,
        "check_details": {
            "source_probability_protocol": source_protocol,
            "semantic_predictive_quotient": semantic_readability,
            "activation_specificity": activation_specificity,
            "history_specificity": history_specificity,
            "renderer_system_length_stability": transfer_stability,
            "cross_family_public_state_transfer": cross_family,
            "transfer_rank_identified": rank_identified,
            "four_bit_retention": quantization,
            "amortized_rate_utility": amortized,
        },
        "authorization": {
            "spq1_protocol_draft_authorized": passed,
            "spq1_execution_authorized": False,
            "held_sender_access_authorized": False,
            "claim_bearing_test_access_authorized": False,
            "receiver_execution_authorized": False,
            "scientific_claim_authorized": False,
            "paper_claim_authorized": False,
        },
    }


def deterministic_prediction_digest(predictions: Mapping[str, Mapping[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for group in sorted(predictions):
        for name in sorted(predictions[group]):
            array = np.asarray(predictions[group][name])
            digest.update(group.encode())
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
