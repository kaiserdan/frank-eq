"""Development-only native-competence qualification for real Stage A.

This module operates before quotient training and never consumes claim-bearing
test worlds. It asks whether frozen founder checkpoints use the query-blind
cached state to answer held-out future operations better than an operation-wise
oracle prior estimated on training worlds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from frank_eq.data.real import RealBundle
from frank_eq.evaluation.bootstrap import bootstrap_statistic
from frank_eq.utils import atomic_write_json


def _group_by_world(values: np.ndarray, world_ids: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    worlds = np.asarray(world_ids, dtype=np.int64)
    return np.asarray(
        [array[worlds == world].mean(axis=0) for world in np.unique(worlds)],
        dtype=np.float64,
    )


def _brier(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return (np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)) ** 2


def compute_native_competence_qualification(
    bundle: RealBundle,
    *,
    min_brier_gain_lower95: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Evaluate frozen founder competence using train/validation worlds only.

    The paired unit is a world. Model and renderer views are averaged within
    world before bootstrapping, preventing pseudo-replication. The operation
    prior is fitted only from training-world oracle labels. Test worlds remain
    completely unused. Every founder must pass independently; an aggregate gain
    cannot hide an incompetent sender.
    """

    heldout_ops = np.asarray(bundle.split.heldout_operation_ids, dtype=np.int64)
    if heldout_ops.size == 0:
        raise ValueError("native qualification requires held-out operations")

    founder_ids = np.asarray(bundle.split.founder_model_ids, dtype=np.int64)
    train_worlds = np.asarray(bundle.split.train_world_ids, dtype=np.int64)
    validation_worlds = np.asarray(bundle.split.validation_world_ids, dtype=np.int64)
    founder_mask = np.isin(bundle.model_ids, founder_ids)
    train_mask = founder_mask & np.isin(bundle.world_ids, train_worlds)
    validation_mask = founder_mask & np.isin(bundle.world_ids, validation_worlds)
    if not np.any(train_mask) or not np.any(validation_mask):
        raise ValueError("native qualification has no founder train/validation rows")

    train_truth = np.asarray(bundle.signatures[train_mask][:, heldout_ops], dtype=np.float64)
    validation_truth = np.asarray(
        bundle.signatures[validation_mask][:, heldout_ops], dtype=np.float64
    )
    validation_native = np.asarray(
        bundle.model_signatures[validation_mask][:, heldout_ops], dtype=np.float64
    )
    validation_row_worlds = np.asarray(bundle.world_ids[validation_mask], dtype=np.int64)

    prior = np.clip(train_truth.mean(axis=0, keepdims=True), 1e-6, 1.0 - 1e-6)
    prior_prediction = np.repeat(prior, validation_truth.shape[0], axis=0)
    native_loss = _brier(validation_truth, validation_native)
    prior_loss = _brier(validation_truth, prior_prediction)
    row_gain = (prior_loss - native_loss).mean(axis=1)
    world_gain = _group_by_world(row_gain, validation_row_worlds)
    interval = bootstrap_statistic(
        world_gain,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    by_model: dict[str, Any] = {}
    founder_checks: dict[str, Any] = {}
    for offset, model_id in enumerate(founder_ids.tolist()):
        selection = validation_mask & (bundle.model_ids == model_id)
        target = np.asarray(bundle.signatures[selection][:, heldout_ops], dtype=np.float64)
        native = np.asarray(bundle.model_signatures[selection][:, heldout_ops], dtype=np.float64)
        worlds = np.asarray(bundle.world_ids[selection], dtype=np.int64)
        model_prior = np.repeat(prior, target.shape[0], axis=0)
        model_world_gain = _group_by_world(
            (_brier(target, model_prior) - _brier(target, native)).mean(axis=1),
            worlds,
        )
        model_interval = bootstrap_statistic(
            model_world_gain,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 100 + offset,
        )
        model_name = bundle.model_names[model_id]
        model_passed = model_interval.lower >= min_brier_gain_lower95
        by_model[model_name] = {
            "native_brier": float(np.mean(_brier(target, native))),
            "prior_brier": float(np.mean(_brier(target, model_prior))),
            "brier_gain_ci": model_interval.to_dict(),
        }
        founder_checks[model_name] = {
            "required": f"lower_95 >= {min_brier_gain_lower95}",
            "observed": model_interval.lower,
            "passed": model_passed,
        }

    by_family: dict[str, Any] = {}
    families = np.asarray([bundle.operations[index].family for index in heldout_ops])
    for offset, family in enumerate(sorted(set(families.tolist()))):
        operation_selection = np.flatnonzero(families == family)
        family_row_gain = (
            prior_loss[:, operation_selection] - native_loss[:, operation_selection]
        ).mean(axis=1)
        family_world_gain = _group_by_world(family_row_gain, validation_row_worlds)
        family_interval = bootstrap_statistic(
            family_world_gain,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1000 + offset,
        )
        by_family[family] = {
            "heldout_operation_ids": heldout_ops[operation_selection].tolist(),
            "brier_gain_ci": family_interval.to_dict(),
        }

    aggregate_passed = interval.lower >= min_brier_gain_lower95
    founders_passed = all(check["passed"] for check in founder_checks.values())
    passed = aggregate_passed and founders_passed
    held_model_id = bundle.split.held_model_id
    return {
        "schema": "frank_eq_native_competence_qualification_v1",
        "scope": "development-only frozen-source competence prerequisite",
        "protocol_mode": "frozen",
        "status": "pass" if passed else "fail",
        "decision": (
            "NATIVE_COMPETENCE_QUALIFIED_FOR_PROTOCOL_DESIGN"
            if passed
            else "STOP_BEFORE_REPRESENTATION_TRAINING"
        ),
        "aggregate_check": {
            "required": f"lower_95 >= {min_brier_gain_lower95}",
            "observed": interval.lower,
            "passed": aggregate_passed,
        },
        "founder_checks": founder_checks,
        "native_brier": float(np.mean(native_loss)),
        "coordinate_prior_brier": float(np.mean(prior_loss)),
        "brier_gain_ci": interval.to_dict(),
        "heldout_operation_ids": heldout_ops.tolist(),
        "by_model": by_model,
        "by_family": by_family,
        "data_usage": {
            "train_worlds": int(len(np.unique(train_worlds))),
            "validation_worlds": int(len(np.unique(validation_worlds))),
            "test_worlds_used": 0,
            "test_labels_consumed": False,
            "founder_model_ids": founder_ids.tolist(),
            "held_sender_rows_used": False,
            "held_sender_cache_present": held_model_id is not None,
            "held_sender_development_exposed": held_model_id is not None,
            "held_sender_used": False,
            "future_held_sender_reuse_permitted": False,
        },
        "authorization": {
            "new_outcome_run_authorized": False,
            "test_access_authorized": False,
            "receiver_execution_authorized": False,
            "scientific_claim_authorized": False,
        },
    }


def qualify_real_cache(
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    min_brier_gain_lower95: float | None = None,
    bootstrap_replicates: int | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    """Load a real cache, resolve the frozen threshold, and write qualification.

    Optional overrides are retained for exploratory development. Any override
    makes the artifact explicitly non-promotional regardless of its numerical
    result.
    """

    source = Path(cache_dir)
    metadata_path = source / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"real cache metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    real_config = metadata.get("real_config", {})
    gate_config = real_config.get("gates", {})
    evaluation_config = real_config.get("evaluation", {})
    override_requested = any(
        value is not None
        for value in (min_brier_gain_lower95, bootstrap_replicates, bootstrap_seed)
    )

    threshold = (
        float(min_brier_gain_lower95)
        if min_brier_gain_lower95 is not None
        else float(gate_config.get("min_native_competence_brier_gain", 0.0))
    )
    replicates = (
        int(bootstrap_replicates)
        if bootstrap_replicates is not None
        else int(evaluation_config.get("bootstrap_replicates", 2000))
    )
    seed = (
        int(bootstrap_seed)
        if bootstrap_seed is not None
        else int(evaluation_config.get("bootstrap_seed", 991))
    )
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")

    bundle = RealBundle.load(source)
    result = compute_native_competence_qualification(
        bundle,
        min_brier_gain_lower95=threshold,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
    )
    if override_requested:
        result["protocol_mode"] = "exploratory_override"
        result["frozen_result_before_override_demotion"] = {
            "status": result["status"],
            "decision": result["decision"],
        }
        result["status"] = "exploratory"
        result["decision"] = "NO_PROMOTION_EXPLORATORY_OVERRIDE"
    result["cache"] = {
        "path": str(source),
        "prompt_format": real_config.get("capture", {}).get("prompt_format"),
        "panel_seed": real_config.get("panel", {}).get("seed"),
        "model_names": list(bundle.model_names),
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target / "qualification.json", result)
    return result
