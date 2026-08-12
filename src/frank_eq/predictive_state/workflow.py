"""Development-only PSR0 capture, train-only probing, and machine decision workflow."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shutil
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from frank_eq.rate_compute.backend import RateComputeModelAdapter
from frank_eq.rate_compute.calibration import fit_platt_calibrator
from frank_eq.rate_compute.config import ResponseProtocolConfig
from frank_eq.real_config import RealModelSpec
from frank_eq.telemetry import WandbTelemetry
from frank_eq.utils import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

from .automaton import PredictiveAutomaton, PredictiveBasis
from .config import PredictiveStateRunConfig
from .panel import (
    PredictivePanel,
    generate_predictive_panel,
    render_future_test_query,
    render_predictive_prefix,
)
from .probes import (
    brier_score,
    choose_ridge_and_layer,
    deterministic_token_hash_features,
    fit_ridge_probe,
    paired_brier_gain_interval,
    quantize_probabilities,
    wrong_history_margin_interval,
)

PREDICTIVE_STATE_ALLOWED_STAGES = ("audit",)
_RENDERER_IDS = {"narrative": 0, "table": 1, "symbolic": 2}
_ROLE_IDS = {"train": 0, "validation": 1}


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
    }
    if torch.cuda.is_available():
        payload["accelerators"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return payload


def parse_predictive_state_stages(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        stages = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        stages = tuple(str(item) for item in value)
    if stages != PREDICTIVE_STATE_ALLOWED_STAGES:
        raise ValueError("PSR0 permits exactly one development stage: audit")
    return stages


def _basis_from_config(config: PredictiveStateRunConfig) -> tuple[PredictiveAutomaton, PredictiveBasis]:
    automaton = config.build_automaton()
    basis = automaton.build_basis(
        horizons=config.automaton.candidate_horizons,
        n_target_tests=config.automaton.n_target_tests,
        target_seed=config.automaton.target_seed,
        max_condition_number=config.automaton.max_core_condition_number,
        max_target_l1=config.automaton.max_target_executor_l1,
    )
    return automaton, basis


def build_predictive_state_plan(
    config: PredictiveStateRunConfig,
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    """Build a deterministic pre-execution plan without loading any checkpoint."""

    config.validate()
    _, basis = _basis_from_config(config)
    train_histories = (
        len(config.panel.train.lengths) * config.panel.train.histories_per_length
    )
    validation_histories = (
        len(config.panel.validation.lengths)
        * config.panel.validation.histories_per_length
    )
    train_prefixes = train_histories * len(config.panel.fit_renderers)
    validation_prefixes = validation_histories * len(config.panel.validation_renderers)
    tests_per_prefix = len(basis.core_tests) + len(basis.target_tests)
    payload: dict[str, Any] = {
        "schema": "frank_eq_predictive_state_plan_v1",
        "protocol_version": config.protocol_version,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "development_only": True,
        "models": [
            {
                "model_id": model.model_id,
                "hf_id": model.hf_id,
                "revision": model.revision,
            }
            for model in config.models
        ],
        "public_basis": {
            "rank": basis.rank,
            "core_tests": [test.to_dict() for test in basis.core_tests],
            "target_tests": [test.to_dict() for test in basis.target_tests],
            "condition_number": basis.condition_number,
            "maximum_target_l1": basis.maximum_target_l1,
        },
        "panel": {
            "train_histories": train_histories,
            "validation_histories": validation_histories,
            "train_lengths": config.panel.train.lengths,
            "validation_lengths": config.panel.validation.lengths,
            "fit_renderers": config.panel.fit_renderers,
            "validation_renderers": config.panel.validation_renderers,
        },
        "compute": {
            "prefixes_per_model": train_prefixes + validation_prefixes,
            "response_tests_per_prefix": tests_per_prefix,
            "response_branches_per_model": (train_prefixes + validation_prefixes)
            * tests_per_prefix,
            "models_loaded_sequentially": True,
            "runtime_basis_queries_are_development_tomography": True,
        },
        "access": {
            "claim_bearing_test_role": False,
            "held_sender": False,
            "receiver": False,
            "future_operation_revealed_before_capture": False,
        },
    }
    payload["plan_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def write_predictive_state_plan(path: str | Path, plan: dict[str, Any]) -> None:
    expected = dict(plan)
    observed = expected.pop("plan_sha256", None)
    if observed != sha256_bytes(canonical_json_bytes(expected)):
        raise ValueError("predictive-state plan has an invalid internal hash")
    atomic_write_json(path, plan)


def _build_panels(
    config: PredictiveStateRunConfig,
    automaton: PredictiveAutomaton,
    basis: PredictiveBasis,
) -> dict[str, PredictivePanel]:
    panel = config.panel
    return {
        "train": generate_predictive_panel(
            automaton,
            basis,
            role="train",
            lengths=panel.train.lengths,
            histories_per_length=panel.train.histories_per_length,
            seed=panel.train.seed,
            min_entropy=panel.min_belief_entropy,
            max_entropy=panel.max_belief_entropy,
            min_core_variance=panel.min_core_variance,
            max_attempt_multiplier=panel.max_generation_attempt_multiplier,
        ),
        "validation": generate_predictive_panel(
            automaton,
            basis,
            role="validation",
            lengths=panel.validation.lengths,
            histories_per_length=panel.validation.histories_per_length,
            seed=panel.validation.seed,
            min_entropy=panel.min_belief_entropy,
            max_entropy=panel.max_belief_entropy,
            min_core_variance=panel.min_core_variance,
            max_attempt_multiplier=panel.max_generation_attempt_multiplier,
        ),
    }


def _response_protocol(config: PredictiveStateRunConfig) -> ResponseProtocolConfig:
    teacher = config.teacher
    return ResponseProtocolConfig(
        candidate_false=teacher.candidate_false,
        candidate_true=teacher.candidate_true,
        basis_protocol="sequence",
        target_protocols=["sequence"],
        compute_families=["lookup"],
        rationale_budget=1,
        pause_budget=1,
        pause_text=" ...",
        reasoning_instruction="",
        final_cue="\nAnswer:",
        sequence_cue=teacher.sequence_cue,
        normalize_sequence_log_likelihood=teacher.normalize_sequence_log_likelihood,
    )


def _score_queries(
    adapter: RateComputeModelAdapter,
    prefix_ids: torch.Tensor,
    prefix_cache: Any,
    query_ids: list[torch.Tensor],
    protocols: ResponseProtocolConfig,
    *,
    batch_size: int,
) -> list[Any]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, query in enumerate(query_ids):
        groups[int(query.shape[1])].append(index)
    result: list[Any | None] = [None] * len(query_ids)
    for indices in groups.values():
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset : offset + batch_size]
            scores = adapter.score_sequence_batch(
                prefix_ids,
                prefix_cache,
                [query_ids[index] for index in selected],
                protocols,
            )
            for index, score in zip(selected, scores, strict=True):
                result[index] = score
    if any(value is None for value in result):
        raise RuntimeError("PSR0 response batching left an unscored future test")
    return [value for value in result if value is not None]


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)
    return sha256_file(path)


def _capture_model(
    config: PredictiveStateRunConfig,
    model_spec: RealModelSpec,
    automaton: PredictiveAutomaton,
    basis: PredictiveBasis,
    panels: dict[str, PredictivePanel],
    root: Path,
    telemetry: WandbTelemetry,
) -> dict[str, Any]:
    adapter = RateComputeModelAdapter(model_spec, config.capture)
    protocols = _response_protocol(config)
    tests = [*basis.core_tests, *basis.target_tests]
    final_features: list[np.ndarray] = []
    embedding_features: list[np.ndarray] = []
    token_rows: list[np.ndarray] = []
    history_ids: list[int] = []
    renderer_ids: list[int] = []
    role_ids: list[int] = []
    lengths: list[int] = []
    semantic_core: list[np.ndarray] = []
    semantic_target: list[np.ndarray] = []
    teacher_probabilities: list[np.ndarray] = []
    teacher_log_odds: list[np.ndarray] = []
    prefix_hashes: list[str] = []
    exact_prefix_checks = 0
    response_branches = 0

    for role, panel in panels.items():
        renderer_names = (
            config.panel.fit_renderers
            if role == "train"
            else config.panel.validation_renderers
        )
        for history in panel.histories:
            for renderer_name in renderer_names:
                statement = render_predictive_prefix(automaton, history, renderer_name)
                prefix_text = adapter._format_prefix(statement)
                prefix_ids = adapter._tokenize(prefix_text)
                with torch.inference_mode():
                    prefix_output = adapter.model(
                        input_ids=prefix_ids,
                        output_hidden_states=True,
                        use_cache=True,
                        return_dict=True,
                    )
                if prefix_output.hidden_states is None or prefix_output.past_key_values is None:
                    raise RuntimeError("PSR0 checkpoint did not return hidden states and a KV cache")
                selected = torch.stack(
                    [prefix_output.hidden_states[index][0, -1] for index in adapter.layer_indices],
                    dim=0,
                )
                embeddings = prefix_output.hidden_states[0][0].mean(dim=0)
                queries: list[torch.Tensor] = []
                for test in tests:
                    query = render_future_test_query(
                        automaton,
                        test,
                        false_display=protocols.candidate_false,
                        true_display=protocols.candidate_true,
                        sequence_cue=protocols.sequence_cue,
                    )
                    queries.append(
                        adapter._query_ids(
                            query,
                            world_statement=statement,
                            prefix_ids=prefix_ids,
                        )
                    )
                    exact_prefix_checks += 1
                scores = _score_queries(
                    adapter,
                    prefix_ids,
                    prefix_output.past_key_values,
                    queries,
                    protocols,
                    batch_size=config.teacher.branch_batch_size,
                )
                response_branches += len(scores)

                final_features.append(selected.float().cpu().numpy())
                embedding_features.append(embeddings.float().cpu().numpy())
                token_rows.append(prefix_ids[0].detach().cpu().numpy().astype(np.int64))
                history_ids.append(history.history_id)
                renderer_ids.append(_RENDERER_IDS[renderer_name])
                role_ids.append(_ROLE_IDS[role])
                lengths.append(history.length)
                semantic_core.append(np.asarray(history.core_probabilities, dtype=np.float64))
                semantic_target.append(np.asarray(history.target_probabilities, dtype=np.float64))
                teacher_probabilities.append(
                    np.asarray([score.probability_true for score in scores], dtype=np.float64)
                )
                teacher_log_odds.append(
                    np.asarray([score.log_odds_score for score in scores], dtype=np.float64)
                )
                prefix_hashes.append(sha256_bytes(prefix_text.encode("utf-8")))

    max_tokens = max(len(row) for row in token_rows)
    token_ids = np.zeros((len(token_rows), max_tokens), dtype=np.int64)
    attention_mask = np.zeros((len(token_rows), max_tokens), dtype=np.bool_)
    for index, row in enumerate(token_rows):
        token_ids[index, : len(row)] = row
        attention_mask[index, : len(row)] = True

    arrays = {
        "final_features": np.stack(final_features).astype(np.float32),
        "embedding_features": np.stack(embedding_features).astype(np.float32),
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "history_ids": np.asarray(history_ids, dtype=np.int64),
        "renderer_ids": np.asarray(renderer_ids, dtype=np.int64),
        "role_ids": np.asarray(role_ids, dtype=np.int8),
        "lengths": np.asarray(lengths, dtype=np.int16),
        "semantic_core": np.stack(semantic_core).astype(np.float64),
        "semantic_target": np.stack(semantic_target).astype(np.float64),
        "teacher_probabilities": np.stack(teacher_probabilities).astype(np.float64),
        "teacher_log_odds": np.stack(teacher_log_odds).astype(np.float64),
    }
    capture_path = root / "captures" / f"{model_spec.model_id}.npz"
    capture_sha = _write_npz(capture_path, arrays)
    observed_revision = getattr(adapter.model.config, "_commit_hash", None) or model_spec.revision
    metadata = {
        "schema": "frank_eq_predictive_state_capture_v1",
        "model_id": model_spec.model_id,
        "hf_id": model_spec.hf_id,
        "revision_requested": model_spec.revision,
        "revision_observed": observed_revision,
        "layer_indices": list(adapter.layer_indices),
        "hidden_width": int(arrays["final_features"].shape[-1]),
        "rows": int(arrays["final_features"].shape[0]),
        "max_tokens": int(max_tokens),
        "core_tests": len(basis.core_tests),
        "target_tests": len(basis.target_tests),
        "exact_prefix_continuity_checks": exact_prefix_checks,
        "response_branches": response_branches,
        "runtime_basis_queries_are_development_tomography": True,
        "candidate_metadata": adapter.candidate_metadata(protocols),
        "prefix_hashes_sha256": sha256_bytes(canonical_json_bytes(prefix_hashes)),
        "array": str(capture_path.relative_to(root)),
        "array_sha256": capture_sha,
    }
    metadata_path = root / "captures" / f"{model_spec.model_id}.json"
    atomic_write_json(metadata_path, metadata)
    telemetry.log({"capture": {"model": model_spec.model_id, "rows": metadata["rows"]}})
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "metadata": str(metadata_path.relative_to(root)),
        "metadata_sha256": sha256_file(metadata_path),
        "array": str(capture_path.relative_to(root)),
        "array_sha256": capture_sha,
    }


def _load_capture(root: Path, entry: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    array_path = root / entry["array"]
    metadata_path = root / entry["metadata"]
    if sha256_file(array_path) != entry["array_sha256"]:
        raise ValueError("PSR0 capture array hash mismatch")
    if sha256_file(metadata_path) != entry["metadata_sha256"]:
        raise ValueError("PSR0 capture metadata hash mismatch")
    with np.load(array_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    return arrays, json.loads(metadata_path.read_text())


def _calibrate_teacher(
    arrays: dict[str, np.ndarray],
    *,
    n_core: int,
    l2: float,
    max_steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train = arrays["role_ids"] == _ROLE_IDS["train"]
    scores = arrays["teacher_log_odds"]
    target = np.concatenate([arrays["semantic_core"], arrays["semantic_target"]], axis=1)
    calibrated = np.zeros_like(scores, dtype=np.float64)
    artifact: dict[str, Any] = {"fit_role": "train", "coordinates": {}}
    for coordinate in range(scores.shape[1]):
        calibrator = fit_platt_calibrator(
            scores[train, coordinate],
            target[train, coordinate],
            l2=l2,
            max_steps=max_steps,
        )
        calibrated[:, coordinate] = calibrator.predict(scores[:, coordinate])
        artifact["coordinates"][str(coordinate)] = {
            "kind": "core" if coordinate < n_core else "target",
            "calibrator": calibrator.to_dict(),
        }
    return calibrated, artifact


def _fit_control(
    features: np.ndarray,
    targets: np.ndarray,
    history_ids: np.ndarray,
    config: PredictiveStateRunConfig,
) -> tuple[Any, dict[str, Any]]:
    selected = choose_ridge_and_layer(
        features[:, None, :],
        targets,
        history_ids,
        ridge_grid=config.probe.ridge_grid,
        selection_fraction=config.probe.selection_fraction,
        selection_seed=config.probe.selection_seed,
    )
    return selected["probe"], {
        key: value for key, value in selected.items() if key != "probe"
    }


def _condition_masks(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    validation = arrays["role_ids"] == _ROLE_IDS["validation"]
    seen_length = np.isin(arrays["lengths"], [8, 16])
    unseen = arrays["renderer_ids"] == _RENDERER_IDS["symbolic"]
    long_history = arrays["lengths"] == 32
    return {
        "aggregate": validation,
        "seen": validation & seen_length & ~unseen,
        "unseen_renderer": validation & seen_length & unseen,
        "length_transfer": validation & long_history,
        "joint_ood": validation & long_history & unseen,
    }


def _summary(
    truth: np.ndarray,
    candidate: np.ndarray,
    baselines: dict[str, np.ndarray],
    history_ids: np.ndarray,
    *,
    config: PredictiveStateRunConfig,
    seed_offset: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": int(len(truth)),
        "histories": int(len(np.unique(history_ids))),
        "candidate_brier": brier_score(truth, candidate),
        "baselines": {},
    }
    for offset, (name, prediction) in enumerate(sorted(baselines.items())):
        result["baselines"][name] = {
            "brier": brier_score(truth, prediction),
            "candidate_gain_ci": paired_brier_gain_interval(
                truth,
                candidate,
                prediction,
                history_ids,
                replicates=config.probe.bootstrap_replicates,
                seed=config.probe.bootstrap_seed + seed_offset + offset,
            ),
        }
    return result


def _evaluate_model(
    config: PredictiveStateRunConfig,
    basis: PredictiveBasis,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    seed_offset: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    n_core = len(basis.core_tests)
    train = arrays["role_ids"] == _ROLE_IDS["train"]
    validation = arrays["role_ids"] == _ROLE_IDS["validation"]
    train_history = arrays["history_ids"][train]
    core_train = arrays["semantic_core"][train]

    activation_selection = choose_ridge_and_layer(
        arrays["final_features"][train],
        core_train,
        train_history,
        ridge_grid=config.probe.ridge_grid,
        selection_fraction=config.probe.selection_fraction,
        selection_seed=config.probe.selection_seed,
    )
    activation_probe = activation_selection.pop("probe")
    selected_layer = int(activation_selection["layer"])

    embedding_probe, embedding_selection = _fit_control(
        arrays["embedding_features"][train], core_train, train_history, config
    )
    width = int(arrays["final_features"].shape[-1])
    token_features = deterministic_token_hash_features(
        arrays["token_ids"],
        arrays["attention_mask"],
        width=width,
        position_period=config.probe.token_hash_position_period,
    )
    token_probe, token_selection = _fit_control(
        token_features[train], core_train, train_history, config
    )

    activation_core = activation_probe.predict(
        arrays["final_features"][:, selected_layer]
    )
    embedding_core = embedding_probe.predict(arrays["embedding_features"])
    token_core = token_probe.predict(token_features)

    teacher_calibrated, teacher_calibration = _calibrate_teacher(
        arrays,
        n_core=n_core,
        l2=config.probe.calibration_l2,
        max_steps=config.probe.calibration_max_steps,
    )
    interactive_core = teacher_calibrated[:, :n_core]
    direct_target = teacher_calibrated[:, n_core:]

    behavioral_probe = fit_ridge_probe(
        arrays["final_features"][train, selected_layer],
        arrays["teacher_probabilities"][train, :n_core],
        ridge=float(activation_selection["ridge"]),
    )
    behavioral_core = behavioral_probe.predict(
        arrays["final_features"][:, selected_layer]
    )

    core_prior = np.repeat(
        arrays["semantic_core"][train].mean(axis=0, keepdims=True),
        len(arrays["semantic_core"]),
        axis=0,
    )
    target_prior = np.repeat(
        arrays["semantic_target"][train].mean(axis=0, keepdims=True),
        len(arrays["semantic_target"]),
        axis=0,
    )
    behavioral_prior = np.repeat(
        arrays["teacher_probabilities"][train, :n_core].mean(axis=0, keepdims=True),
        len(arrays["semantic_core"]),
        axis=0,
    )

    compiled_activation = basis.execute(activation_core)
    compiled_token = basis.execute(token_core)
    compiled_embedding = basis.execute(embedding_core)

    conditions = _condition_masks(arrays)
    metrics: dict[str, Any] = {
        "model_id": metadata["model_id"],
        "selected_layer": selected_layer,
        "layer_indices": metadata["layer_indices"],
        "activation_selection": activation_selection,
        "embedding_selection": embedding_selection,
        "token_selection": token_selection,
        "core": {},
        "compiled_targets": {},
        "compiled_targets_joint_ood_by_horizon": {},
        "compiled_targets_joint_ood_by_observation": {},
        "behavioral_core": {},
        "renderer_pair_brier_gap": None,
    }
    for condition_offset, (name, mask) in enumerate(conditions.items()):
        if not np.any(mask):
            raise RuntimeError(f"PSR0 condition {name} has no validation rows")
        identities = arrays["history_ids"][mask]
        metrics["core"][name] = _summary(
            arrays["semantic_core"][mask],
            activation_core[mask],
            {
                "prior": core_prior[mask],
                "token_hash": token_core[mask],
                "embedding_mean": embedding_core[mask],
                "interactive_teacher": interactive_core[mask],
            },
            identities,
            config=config,
            seed_offset=seed_offset + 100 * condition_offset,
        )
        metrics["core"][name]["wrong_history_margin_ci"] = (
            wrong_history_margin_interval(
                arrays["semantic_core"][mask],
                activation_core[mask],
                identities,
                arrays["lengths"][mask],
                replicates=config.probe.bootstrap_replicates,
                seed=(
                    config.probe.bootstrap_seed
                    + seed_offset
                    + 5000
                    + condition_offset
                ),
            )
        )
        metrics["compiled_targets"][name] = _summary(
            arrays["semantic_target"][mask],
            compiled_activation[mask],
            {
                "prior": target_prior[mask],
                "direct_teacher": direct_target[mask],
                "token_hash_compiled": compiled_token[mask],
                "embedding_compiled": compiled_embedding[mask],
            },
            identities,
            config=config,
            seed_offset=seed_offset + 1000 + 100 * condition_offset,
        )
        metrics["behavioral_core"][name] = _summary(
            arrays["teacher_probabilities"][mask, :n_core],
            behavioral_core[mask],
            {"prior": behavioral_prior[mask]},
            identities,
            config=config,
            seed_offset=seed_offset + 2000 + 100 * condition_offset,
        )

    seen = validation & np.isin(arrays["lengths"], [8, 16]) & (
        arrays["renderer_ids"] != _RENDERER_IDS["symbolic"]
    )
    unseen = validation & np.isin(arrays["lengths"], [8, 16]) & (
        arrays["renderer_ids"] == _RENDERER_IDS["symbolic"]
    )
    joint_ood = conditions["joint_ood"]
    target_horizons = np.asarray(
        [len(test.actions) for test in basis.target_tests], dtype=np.int64
    )
    target_observations = np.asarray(
        [test.observation for test in basis.target_tests], dtype=np.int64
    )
    for horizon in sorted(set(target_horizons.tolist())):
        columns = np.flatnonzero(target_horizons == horizon)
        metrics["compiled_targets_joint_ood_by_horizon"][str(horizon)] = _summary(
            arrays["semantic_target"][joint_ood][:, columns],
            compiled_activation[joint_ood][:, columns],
            {
                "prior": target_prior[joint_ood][:, columns],
                "direct_teacher": direct_target[joint_ood][:, columns],
                "token_hash_compiled": compiled_token[joint_ood][:, columns],
            },
            arrays["history_ids"][joint_ood],
            config=config,
            seed_offset=seed_offset + 3000 + 100 * horizon,
        )
    for observation in sorted(set(target_observations.tolist())):
        columns = np.flatnonzero(target_observations == observation)
        metrics["compiled_targets_joint_ood_by_observation"][str(observation)] = (
            _summary(
                arrays["semantic_target"][joint_ood][:, columns],
                compiled_activation[joint_ood][:, columns],
                {
                    "prior": target_prior[joint_ood][:, columns],
                    "direct_teacher": direct_target[joint_ood][:, columns],
                    "token_hash_compiled": compiled_token[joint_ood][:, columns],
                },
                arrays["history_ids"][joint_ood],
                config=config,
                seed_offset=seed_offset + 4000 + 100 * observation,
            )
        )

    metrics["renderer_pair_brier_gap"] = abs(
        brier_score(arrays["semantic_core"][seen], activation_core[seen])
        - brier_score(arrays["semantic_core"][unseen], activation_core[unseen])
    )

    training_artifact = {
        "model_id": metadata["model_id"],
        "semantic_activation_probe": activation_probe.to_dict(),
        "semantic_embedding_probe": embedding_probe.to_dict(),
        "semantic_token_probe": token_probe.to_dict(),
        "behavioral_activation_probe": behavioral_probe.to_dict(),
        "teacher_calibration": teacher_calibration,
    }
    predictions = {
        "activation_core": activation_core,
        "token_core": token_core,
        "embedding_core": embedding_core,
        "interactive_core": interactive_core,
        "behavioral_core": behavioral_core,
        "compiled_activation": compiled_activation,
        "compiled_token": compiled_token,
        "direct_target": direct_target,
    }
    return metrics, training_artifact, predictions


def _gate_decision(
    config: PredictiveStateRunConfig,
    basis: PredictiveBasis,
    metrics_by_model: dict[str, Any],
    oracle_error: float,
) -> dict[str, Any]:
    gate = config.gates
    core_checks: dict[str, bool] = {}
    activation_checks: dict[str, bool] = {}
    specificity_checks: dict[str, bool] = {}
    renderer_checks: dict[str, bool] = {}
    length_checks: dict[str, bool] = {}
    composition_checks: dict[str, bool] = {}
    for model_id, metrics in metrics_by_model.items():
        for condition in ("seen", "unseen_renderer", "length_transfer", "joint_ood"):
            lower = metrics["core"][condition]["baselines"]["prior"][
                "candidate_gain_ci"
            ]["lower"]
            core_checks[f"{model_id}|{condition}"] = (
                float(lower) >= gate.min_core_gain_over_prior_lower95
            )
        activation_lower = metrics["core"]["joint_ood"]["baselines"]["token_hash"][
            "candidate_gain_ci"
        ]["lower"]
        activation_checks[model_id] = (
            float(activation_lower) > gate.min_activation_over_token_lower95
        )
        specificity_lower = metrics["core"]["joint_ood"][
            "wrong_history_margin_ci"
        ]["lower"]
        specificity_checks[model_id] = (
            float(specificity_lower) > gate.min_wrong_history_margin_lower95
        )
        renderer_lower = metrics["core"]["unseen_renderer"]["baselines"]["prior"][
            "candidate_gain_ci"
        ]["lower"]
        renderer_checks[model_id] = (
            float(renderer_lower) >= gate.min_unseen_renderer_gain_lower95
            and float(metrics["renderer_pair_brier_gap"])
            <= gate.max_renderer_pair_brier_gap
        )
        length_lower = metrics["core"]["length_transfer"]["baselines"]["prior"][
            "candidate_gain_ci"
        ]["lower"]
        length_checks[model_id] = (
            float(length_lower) >= gate.min_length_transfer_gain_lower95
        )
        target_group = metrics["compiled_targets"]["joint_ood"]["baselines"]
        horizon_groups = metrics["compiled_targets_joint_ood_by_horizon"]
        aggregate_composition = (
            float(target_group["prior"]["candidate_gain_ci"]["lower"])
            > gate.min_compiled_over_prior_lower95
            and float(target_group["direct_teacher"]["candidate_gain_ci"]["lower"])
            > gate.min_compiled_over_direct_lower95
        )
        horizon_composition = all(
            float(group["baselines"]["prior"]["candidate_gain_ci"]["lower"])
            > gate.min_compiled_over_prior_lower95
            and float(
                group["baselines"]["direct_teacher"]["candidate_gain_ci"]["lower"]
            )
            > gate.min_compiled_over_direct_lower95
            for group in horizon_groups.values()
        )
        composition_checks[model_id] = (
            aggregate_composition and bool(horizon_groups) and horizon_composition
        )

    oracle_passed = (
        basis.rank == len(config.automaton.state_names)
        and basis.condition_number <= gate.max_public_basis_condition_number
        and oracle_error <= gate.max_oracle_executor_abs_error
    )
    core_passed = bool(core_checks) and all(core_checks.values())
    activation_passed = bool(activation_checks) and all(activation_checks.values())
    specificity_passed = bool(specificity_checks) and all(
        specificity_checks.values()
    )
    renderer_passed = bool(renderer_checks) and all(renderer_checks.values())
    length_passed = bool(length_checks) and all(length_checks.values())
    composition_passed = bool(composition_checks) and all(composition_checks.values())
    passed = (
        oracle_passed
        and core_passed
        and activation_passed
        and specificity_passed
        and renderer_passed
        and length_passed
        and composition_passed
    )
    if not oracle_passed:
        diagnosis = "PREDICTIVE_BASIS_OR_EXECUTOR_INVALID"
    elif not core_passed:
        diagnosis = "ACTIVATION_PREDICTIVE_STATE_NOT_READABLE"
    elif not activation_passed:
        diagnosis = "NO_ACTIVATION_SPECIFIC_PREDICTIVE_STATE_ADVANTAGE"
    elif not specificity_passed:
        diagnosis = "PREDICTIVE_STATE_NOT_HISTORY_SPECIFIC"
    elif not renderer_passed:
        diagnosis = "PREDICTIVE_STATE_NOT_RENDERER_INVARIANT"
    elif not length_passed:
        diagnosis = "PREDICTIVE_STATE_NOT_LENGTH_TRANSFERABLE"
    elif not composition_passed:
        diagnosis = "PUBLIC_PREDICTIVE_STATE_NOT_COMPOSITIONALLY_USEFUL"
    else:
        diagnosis = "PUBLIC_PREDICTIVE_STATE_CANDIDATE_SUPPORTED"
    return {
        "schema": "frank_eq_predictive_state_decision_v1",
        "status": "pass" if passed else "fail",
        "decision": (
            "DRAFT_FRESH_PSR_STAGE1_REGISTRATION"
            if passed
            else "STOP_BEFORE_CLAIM_BEARING_PSR_STAGE1"
        ),
        "diagnosis": diagnosis,
        "checks": {
            "oracle_basis": {"passed": oracle_passed},
            "activation_core_readability": {
                "passed": core_passed,
                "groups": core_checks,
            },
            "activation_specificity": {
                "passed": activation_passed,
                "models": activation_checks,
            },
            "history_specificity": {
                "passed": specificity_passed,
                "models": specificity_checks,
            },
            "renderer_transfer": {
                "passed": renderer_passed,
                "models": renderer_checks,
            },
            "length_transfer": {
                "passed": length_passed,
                "models": length_checks,
            },
            "target_composition": {
                "passed": composition_passed,
                "models": composition_checks,
            },
        },
        "authorization": {
            "psr_stage1_protocol_draft_authorized": passed,
            "psr_stage1_execution_authorized": False,
            "claim_bearing_test_access_authorized": False,
            "held_sender_onboarding_authorized": False,
            "receiver_execution_authorized": False,
            "scientific_claim_authorized": False,
            "paper_claim_authorized": False,
        },
    }


def _artifact_manifest(root: Path, paths: list[str]) -> dict[str, Any]:
    return {
        "schema": "frank_eq_predictive_state_artifact_manifest_v1",
        "files": {
            path: sha256_file(root / path) for path in paths if (root / path).is_file()
        },
    }


def run_predictive_state_audit(
    config: PredictiveStateRunConfig,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    stages: str | list[str] | tuple[str, ...] = PREDICTIVE_STATE_ALLOWED_STAGES,
    inspected_plan: dict[str, Any] | None = None,
    inspected_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one fail-closed development census with no held sender or test role."""

    parse_predictive_state_stages(stages)
    config.validate()
    expected_plan = build_predictive_state_plan(config, config_path=config_path)
    if inspected_plan is None:
        inspected_plan = expected_plan
    if inspected_plan != expected_plan:
        raise ValueError("inspected PSR0 plan differs from the current frozen config")
    if inspected_plan_sha256 is not None and inspected_plan_sha256 != expected_plan["plan_sha256"]:
        raise ValueError("inspected PSR0 plan SHA-256 differs from the frozen plan")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, root / "config.yaml")
    telemetry = WandbTelemetry(
        config.logging.wandb,
        run_name=config.run_name,
        job=_environment(),
    )
    manifest = {
        "schema": "frank_eq_predictive_state_run_manifest_v1",
        "run_name": config.run_name,
        "protocol_version": config.protocol_version,
        "development_only": True,
        "config_path": str(config_path),
        "config_sha256": sha256_file(root / "config.yaml"),
        "plan_sha256": expected_plan["plan_sha256"],
        "stages": ["audit"],
        "created_at": _timestamp(),
        "environment": _environment(),
        "access_contract": {
            "state_precedes_future_test": True,
            "literal_kv_reuse": True,
            "exact_replay_fallback": False,
            "claim_bearing_test_role": False,
            "held_sender": False,
            "receiver": False,
        },
    }
    atomic_write_json(root / "run_manifest.json", manifest)
    atomic_write_json(root / "dry_run_plan.json", expected_plan)
    status: dict[str, Any] = {
        "schema": "frank_eq_predictive_state_status_v1",
        "state": "running",
        "current_stage": "audit",
        "completed_stages": [],
        "started_at": _timestamp(),
        "failure": None,
    }
    atomic_write_json(root / "workflow_status.json", status)
    started = time.time()

    try:
        automaton, basis = _basis_from_config(config)
        panels = _build_panels(config, automaton, basis)
        atomic_write_json(root / "automaton.json", {
            "schema": "frank_eq_predictive_automaton_v1",
            "state_names": list(automaton.state_names),
            "action_names": list(automaton.action_names),
            "observation_names": list(automaton.observation_names),
            "transitions": automaton.transitions.tolist(),
            "emissions": automaton.emissions.tolist(),
            "initial_belief": automaton.initial_belief.tolist(),
        })
        atomic_write_json(root / "public_basis.json", basis.to_dict())
        for role, panel in panels.items():
            atomic_write_json(root / "panels" / f"{role}.json", panel.to_dict())

        capture_entries: dict[str, Any] = {}
        for model in config.models:
            capture_entries[model.model_id] = _capture_model(
                config,
                model,
                automaton,
                basis,
                panels,
                root,
                telemetry,
            )
        capture_manifest = {
            "schema": "frank_eq_predictive_state_capture_manifest_v1",
            "entries": capture_entries,
        }
        atomic_write_json(root / "capture_manifest.json", capture_manifest)

        oracle_rows = np.concatenate(
            [
                np.asarray([history.core_probabilities for history in panel.histories])
                for panel in panels.values()
            ],
            axis=0,
        )
        target_rows = np.concatenate(
            [
                np.asarray([history.target_probabilities for history in panel.histories])
                for panel in panels.values()
            ],
            axis=0,
        )
        oracle_error = float(
            np.max(np.abs(basis.execute(oracle_rows, clip=False) - target_rows))
        )

        metrics_by_model: dict[str, Any] = {}
        training: dict[str, Any] = {}
        prediction_manifest: dict[str, Any] = {
            "schema": "frank_eq_predictive_state_predictions_manifest_v1",
            "entries": {},
        }
        for model_offset, (model_id, entry) in enumerate(sorted(capture_entries.items())):
            arrays, metadata = _load_capture(root, entry)
            metrics, training_artifact, predictions = _evaluate_model(
                config,
                basis,
                arrays,
                metadata,
                seed_offset=10_000 * model_offset,
            )
            metrics_by_model[model_id] = metrics
            training[model_id] = training_artifact
            prediction_path = root / "predictions" / f"{model_id}.npz"
            prediction_sha = _write_npz(prediction_path, predictions)
            prediction_manifest["entries"][model_id] = {
                "path": str(prediction_path.relative_to(root)),
                "sha256": prediction_sha,
            }
        atomic_write_json(root / "probe_training.json", training)
        atomic_write_json(root / "predictions_manifest.json", prediction_manifest)

        quantization: dict[str, Any] = {}
        for bits in config.probe.quantization_bits:
            quantized = quantize_probabilities(oracle_rows, bits)
            quantization[str(bits)] = {
                "bits_per_coordinate": bits,
                "message_bits": bits * len(basis.core_tests),
                "oracle_target_brier": brier_score(
                    target_rows, basis.execute(quantized)
                ),
            }
        metrics_payload = {
            "schema": "frank_eq_predictive_state_metrics_v1",
            "scope": "development-only public predictive-state census",
            "public_basis": {
                "rank": basis.rank,
                "condition_number": basis.condition_number,
                "maximum_target_l1": basis.maximum_target_l1,
                "oracle_executor_max_abs_error": oracle_error,
                "core_tests": [test.to_dict() for test in basis.core_tests],
                "target_tests": [test.to_dict() for test in basis.target_tests],
            },
            "models": metrics_by_model,
            "quantization": quantization,
            "data_usage": {
                "train_histories": len(panels["train"].histories),
                "validation_histories": len(panels["validation"].histories),
                "claim_bearing_test_histories": 0,
                "held_sender_rows": 0,
                "receiver_rows": 0,
            },
        }
        decision = _gate_decision(config, basis, metrics_by_model, oracle_error)
        atomic_write_json(root / "metrics.json", metrics_payload)
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
        artifact_paths = [
            "config.yaml",
            "run_manifest.json",
            "dry_run_plan.json",
            "workflow_status.json",
            "automaton.json",
            "public_basis.json",
            "panels/train.json",
            "panels/validation.json",
            "capture_manifest.json",
            "probe_training.json",
            "predictions_manifest.json",
            "metrics.json",
            "decision.json",
            *[
                value[key]
                for value in capture_entries.values()
                for key in ("metadata", "array")
            ],
            *[entry["path"] for entry in prediction_manifest["entries"].values()],
        ]
        summary = {
            "schema": "frank_eq_predictive_state_run_v1",
            "status": "completed",
            "workflow_integrity_passed": True,
            "development_only": True,
            "root": str(root),
            "decision": decision,
            "artifact_manifest": str(root / "artifact_manifest.json"),
            "authorization": decision["authorization"],
            "telemetry": telemetry.status(),
        }
        atomic_write_json(root / "run_summary.json", summary)
        artifact_paths.append("run_summary.json")
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
        atomic_write_json(root / "workflow_status.json", status)
        raise
    finally:
        telemetry.finish()
