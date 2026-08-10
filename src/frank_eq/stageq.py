"""Paired development comparison for Stage-Q prompt/capture candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from frank_eq.data.real import RealBundle
from frank_eq.evaluation.bootstrap import bootstrap_statistic
from frank_eq.qualification import compute_native_competence_qualification
from frank_eq.utils import atomic_write_json


def _world_means(values: np.ndarray, world_ids: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    worlds = np.asarray(world_ids, dtype=np.int64)
    return np.asarray(
        [array[worlds == world].mean(axis=0) for world in np.unique(worlds)],
        dtype=np.float64,
    )


def _assert_paired_contract(baseline: RealBundle, candidate: RealBundle) -> None:
    scalar_pairs = (
        (baseline.model_names, candidate.model_names, "model roster"),
        (baseline.split.to_dict(), candidate.split.to_dict(), "split manifest"),
        (
            [operation.to_dict() for operation in baseline.operations],
            [operation.to_dict() for operation in candidate.operations],
            "operation registry",
        ),
    )
    for left, right, name in scalar_pairs:
        if left != right:
            raise ValueError(f"Stage-Q caches differ in {name}")
    array_pairs = (
        (baseline.world_ids, candidate.world_ids, "world IDs"),
        (baseline.model_ids, candidate.model_ids, "model IDs"),
        (baseline.renderer_ids, candidate.renderer_ids, "renderer IDs"),
        (baseline.signatures, candidate.signatures, "oracle signatures"),
        (baseline.facts, candidate.facts, "fact labels"),
        (baseline.residual, candidate.residual, "residual labels"),
        (
            baseline.operation_descriptors,
            candidate.operation_descriptors,
            "operation descriptors",
        ),
    )
    for left, right, name in array_pairs:
        if not np.array_equal(left, right):
            raise ValueError(f"Stage-Q caches are not paired on {name}")


def compare_native_competence_bundles(
    baseline: RealBundle,
    candidate: RealBundle,
    *,
    min_candidate_brier_gain_lower95: float,
    min_paired_improvement_lower95: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Compare two capture contracts on identical development examples.

    Candidate source qualification and prompt-effect identification are distinct
    decisions. A competent candidate may be used as a prerequisite even if it
    does not outperform the legacy placement; in that case no causal claim
    about prompt format is permitted.
    """

    _assert_paired_contract(baseline, candidate)
    heldout_ops = np.asarray(baseline.split.heldout_operation_ids, dtype=np.int64)
    founder_ids = np.asarray(baseline.split.founder_model_ids, dtype=np.int64)
    validation_worlds = np.asarray(baseline.split.validation_world_ids, dtype=np.int64)
    validation_mask = np.isin(baseline.world_ids, validation_worlds) & np.isin(
        baseline.model_ids, founder_ids
    )
    truth = np.asarray(baseline.signatures[validation_mask][:, heldout_ops], dtype=np.float64)
    baseline_native = np.asarray(
        baseline.model_signatures[validation_mask][:, heldout_ops], dtype=np.float64
    )
    candidate_native = np.asarray(
        candidate.model_signatures[validation_mask][:, heldout_ops], dtype=np.float64
    )
    worlds = np.asarray(baseline.world_ids[validation_mask], dtype=np.int64)
    if truth.size == 0:
        raise ValueError("Stage-Q comparison has no founder validation examples")

    # Positive means the candidate reduces Brier relative to the baseline on
    # the identical model/world/renderer/operation unit.
    paired_row_improvement = (
        (truth - baseline_native) ** 2 - (truth - candidate_native) ** 2
    ).mean(axis=1)
    world_improvement = _world_means(paired_row_improvement, worlds)
    paired_interval = bootstrap_statistic(
        world_improvement,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    by_model: dict[str, Any] = {}
    for offset, model_id in enumerate(founder_ids.tolist()):
        selection = validation_mask & (baseline.model_ids == model_id)
        target = np.asarray(baseline.signatures[selection][:, heldout_ops], dtype=np.float64)
        base = np.asarray(baseline.model_signatures[selection][:, heldout_ops], dtype=np.float64)
        cand = np.asarray(candidate.model_signatures[selection][:, heldout_ops], dtype=np.float64)
        model_worlds = np.asarray(baseline.world_ids[selection], dtype=np.int64)
        model_world_gain = _world_means(
            (((target - base) ** 2 - (target - cand) ** 2).mean(axis=1)),
            model_worlds,
        )
        by_model[baseline.model_names[model_id]] = bootstrap_statistic(
            model_world_gain,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 100 + offset,
        ).to_dict()

    families = np.asarray([baseline.operations[index].family for index in heldout_ops])
    by_family: dict[str, Any] = {}
    element_improvement = (truth - baseline_native) ** 2 - (truth - candidate_native) ** 2
    for offset, family in enumerate(sorted(set(families.tolist()))):
        operation_selection = np.flatnonzero(families == family)
        family_world_gain = _world_means(
            element_improvement[:, operation_selection].mean(axis=1),
            worlds,
        )
        by_family[family] = bootstrap_statistic(
            family_world_gain,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1000 + offset,
        ).to_dict()

    candidate_qualification = compute_native_competence_qualification(
        candidate,
        min_brier_gain_lower95=min_candidate_brier_gain_lower95,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 10_000,
    )
    prompt_effect_passed = paired_interval.lower >= min_paired_improvement_lower95
    source_qualified = candidate_qualification["status"] == "pass"
    return {
        "schema": "frank_eq_stageq_paired_prompt_comparison_v1",
        "scope": "development-only paired capture-contract comparison",
        "status": "pass" if source_qualified else "fail",
        "decision": (
            "SOURCE_CONTRACT_QUALIFIED_FOR_STAGEA_REGISTRATION"
            if source_qualified
            else "STOP_STAGEQ_CANDIDATE"
        ),
        "source_contract_qualified": source_qualified,
        "prompt_effect_identified": prompt_effect_passed,
        "prompt_effect_decision": (
            "PROPER_CHAT_TURN_IMPROVEMENT_IDENTIFIED"
            if prompt_effect_passed
            else "NO_PROMPT_EFFECT_CLAIM"
        ),
        "paired_improvement_check": {
            "required_for_prompt_effect_claim_only": (
                f"lower_95 >= {min_paired_improvement_lower95}"
            ),
            "observed": paired_interval.lower,
            "passed": prompt_effect_passed,
        },
        "paired_brier_improvement_ci": paired_interval.to_dict(),
        "candidate_competence": candidate_qualification,
        "by_model": by_model,
        "by_family": by_family,
        "data_usage": {
            "validation_worlds": int(len(np.unique(validation_worlds))),
            "test_worlds_used": 0,
            "test_labels_consumed": False,
            "founder_model_ids": founder_ids.tolist(),
            "held_sender_used": False,
        },
        "authorization": {
            "new_stagea_outcome_run_authorized": False,
            "test_access_authorized": False,
            "receiver_execution_authorized": False,
            "scientific_claim_authorized": False,
        },
    }


def compare_native_competence_caches(
    baseline_cache: str | Path,
    candidate_cache: str | Path,
    output_dir: str | Path,
    *,
    min_paired_improvement_lower95: float = 0.0,
) -> dict[str, Any]:
    """Load two paired caches and write a Stage-Q comparison artifact."""

    baseline_path = Path(baseline_cache)
    candidate_path = Path(candidate_cache)
    baseline_metadata = json.loads((baseline_path / "metadata.json").read_text())
    candidate_metadata = json.loads((candidate_path / "metadata.json").read_text())
    baseline_config = baseline_metadata.get("real_config", {})
    candidate_config = candidate_metadata.get("real_config", {})
    evaluation = candidate_config.get("evaluation", {})
    gates = candidate_config.get("gates", {})
    replicates = int(evaluation.get("bootstrap_replicates", 2000))
    seed = int(evaluation.get("bootstrap_seed", 991))
    candidate_threshold = float(gates.get("min_native_competence_brier_gain", 0.0))

    result = compare_native_competence_bundles(
        RealBundle.load(baseline_path),
        RealBundle.load(candidate_path),
        min_candidate_brier_gain_lower95=candidate_threshold,
        min_paired_improvement_lower95=min_paired_improvement_lower95,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
    )
    result["caches"] = {
        "baseline": {
            "path": str(baseline_path),
            "prompt_format": baseline_config.get("capture", {}).get("prompt_format"),
            "panel_seed": baseline_config.get("panel", {}).get("seed"),
        },
        "candidate": {
            "path": str(candidate_path),
            "prompt_format": candidate_config.get("capture", {}).get("prompt_format"),
            "panel_seed": candidate_config.get("panel", {}).get("seed"),
        },
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target / "comparison.json", result)
    return result
