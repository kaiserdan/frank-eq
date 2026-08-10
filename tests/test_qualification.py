from __future__ import annotations

import numpy as np
import pytest

from frank_eq.data.real import RealBundle
from frank_eq.qualification import compute_native_competence_qualification
from frank_eq.schemas import OperationDefinition, SplitManifest
from frank_eq.stageq import compare_native_competence_bundles


def _bundle(*, competent: bool) -> RealBundle:
    rng = np.random.default_rng(19)
    n_worlds = 30
    n_models = 3
    n_renderers = 2
    n_operations = 4
    truth_by_world = 0.02 + 0.96 * rng.integers(
        0, 2, size=(n_worlds, n_operations)
    ).astype(np.float32)
    facts_by_world = rng.integers(0, 2, size=(n_worlds, 4)).astype(np.float32)
    residual_by_world = rng.normal(size=(n_worlds, 2)).astype(np.float32)

    rows = n_worlds * n_models * n_renderers
    world_ids = np.empty(rows, dtype=np.int64)
    model_ids = np.empty(rows, dtype=np.int64)
    renderer_ids = np.empty(rows, dtype=np.int64)
    hidden = np.zeros((rows, 2, 8), dtype=np.float32)
    hidden_mask = np.ones_like(hidden, dtype=np.bool_)
    facts = np.empty((rows, 4), dtype=np.float32)
    residual = np.empty((rows, 2), dtype=np.float32)
    signatures = np.empty((rows, n_operations), dtype=np.float32)
    model_signatures = np.empty_like(signatures)

    cursor = 0
    for world in range(n_worlds):
        for model in range(n_models):
            for renderer in range(n_renderers):
                world_ids[cursor] = world
                model_ids[cursor] = model
                renderer_ids[cursor] = renderer
                facts[cursor] = facts_by_world[world]
                residual[cursor] = residual_by_world[world]
                signatures[cursor] = truth_by_world[world]
                model_signatures[cursor] = (
                    truth_by_world[world]
                    if competent or model == 2
                    else 1.0 - truth_by_world[world]
                )
                cursor += 1

    operations = [
        OperationDefinition(
            operation_id=index,
            family="lookup" if index < 2 else "compose",
            fact_args=(0, 1),
            residual_args=(2, 3),
            polarity=1.0,
        )
        for index in range(n_operations)
    ]
    return RealBundle(
        world_ids=world_ids,
        model_ids=model_ids,
        renderer_ids=renderer_ids,
        hidden=hidden,
        hidden_mask=hidden_mask,
        facts=facts,
        residual=residual,
        signatures=signatures,
        model_signatures=model_signatures,
        operation_descriptors=np.eye(n_operations, dtype=np.float32),
        operations=operations,
        split=SplitManifest(
            train_world_ids=tuple(range(18)),
            validation_world_ids=tuple(range(18, 24)),
            test_world_ids=tuple(range(24, 30)),
            train_operation_ids=(0, 2),
            heldout_operation_ids=(1, 3),
            founder_model_ids=(0, 1),
            held_model_id=2,
        ),
        model_hidden_dims=[8, 8, 8],
        n_layers=2,
        model_names=["founder-a", "founder-b", "held-c"],
    )


def _qualify(bundle: RealBundle) -> dict:
    return compute_native_competence_qualification(
        bundle,
        min_brier_gain_lower95=0.0,
        bootstrap_replicates=200,
        bootstrap_seed=7,
    )


def test_native_qualification_passes_for_competent_founders() -> None:
    result = _qualify(_bundle(competent=True))
    assert result["status"] == "pass"
    assert result["brier_gain_ci"]["lower"] > 0.0
    assert all(check["passed"] for check in result["founder_checks"].values())
    assert result["data_usage"]["test_worlds_used"] == 0
    assert result["data_usage"]["held_sender_used"] is False
    assert not any(result["authorization"].values())


def test_native_qualification_stops_anti_predictive_founders() -> None:
    result = _qualify(_bundle(competent=False))
    assert result["status"] == "fail"
    assert result["decision"] == "STOP_BEFORE_REPRESENTATION_TRAINING"
    assert result["brier_gain_ci"]["upper"] < 0.0
    assert set(result["by_family"]) == {"compose", "lookup"}


def test_aggregate_gain_cannot_hide_one_failed_founder() -> None:
    bundle = _bundle(competent=True)
    train_world_mask = np.isin(bundle.world_ids, bundle.split.train_world_ids)
    operation_prior = bundle.signatures[train_world_mask].mean(axis=0)
    failed_model = bundle.model_ids == 1
    failed_truth = bundle.signatures[failed_model]
    away_from_truth = np.where(failed_truth >= 0.5, -0.05, 0.05)
    bundle.model_signatures[failed_model] = np.clip(
        operation_prior[None, :] + away_from_truth,
        1e-4,
        1.0 - 1e-4,
    )
    result = _qualify(bundle)
    assert result["aggregate_check"]["passed"] is True
    assert result["founder_checks"]["founder-a"]["passed"] is True
    assert result["founder_checks"]["founder-b"]["passed"] is False
    assert result["status"] == "fail"


def test_stageq_paired_comparison_identifies_improvement_when_both_pass() -> None:
    result = compare_native_competence_bundles(
        _bundle(competent=False),
        _bundle(competent=True),
        min_candidate_brier_gain_lower95=0.0,
        min_paired_improvement_lower95=0.0,
        bootstrap_replicates=200,
        bootstrap_seed=11,
    )
    assert result["status"] == "pass"
    assert result["source_contract_qualified"] is True
    assert result["prompt_effect_identified"] is True
    assert result["paired_brier_improvement_ci"]["lower"] > 0.0
    assert result["candidate_competence"]["status"] == "pass"
    assert result["data_usage"]["test_worlds_used"] == 0
    assert not any(result["authorization"].values())


def test_source_can_qualify_without_a_prompt_effect_claim() -> None:
    baseline = _bundle(competent=True)
    candidate = _bundle(competent=True)
    result = compare_native_competence_bundles(
        baseline,
        candidate,
        min_candidate_brier_gain_lower95=0.0,
        min_paired_improvement_lower95=0.01,
        bootstrap_replicates=200,
        bootstrap_seed=11,
    )
    assert result["status"] == "pass"
    assert result["decision"] == "SOURCE_CONTRACT_QUALIFIED_FOR_STAGEA_REGISTRATION"
    assert result["source_contract_qualified"] is True
    assert result["prompt_effect_identified"] is False
    assert result["prompt_effect_decision"] == "NO_PROMPT_EFFECT_CLAIM"


def test_stageq_rejects_unpaired_caches() -> None:
    baseline = _bundle(competent=False)
    candidate = _bundle(competent=True)
    candidate.world_ids = candidate.world_ids.copy()
    candidate.world_ids[0] = 999
    with pytest.raises(ValueError, match="world IDs"):
        compare_native_competence_bundles(
            baseline,
            candidate,
            min_candidate_brier_gain_lower95=0.0,
            min_paired_improvement_lower95=0.0,
            bootstrap_replicates=20,
            bootstrap_seed=11,
        )
