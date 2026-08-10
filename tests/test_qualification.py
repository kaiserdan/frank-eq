from __future__ import annotations

import numpy as np

from frank_eq.data.real import RealBundle
from frank_eq.qualification import compute_native_competence_qualification
from frank_eq.schemas import OperationDefinition, SplitManifest


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


def test_native_qualification_passes_for_competent_founders() -> None:
    result = compute_native_competence_qualification(
        _bundle(competent=True),
        min_brier_gain_lower95=0.0,
        bootstrap_replicates=200,
        bootstrap_seed=7,
    )
    assert result["status"] == "pass"
    assert result["brier_gain_ci"]["lower"] > 0.0
    assert result["data_usage"]["test_worlds_used"] == 0
    assert result["data_usage"]["held_sender_used"] is False
    assert not any(result["authorization"].values())


def test_native_qualification_stops_anti_predictive_founders() -> None:
    result = compute_native_competence_qualification(
        _bundle(competent=False),
        min_brier_gain_lower95=0.0,
        bootstrap_replicates=200,
        bootstrap_seed=7,
    )
    assert result["status"] == "fail"
    assert result["decision"] == "STOP_BEFORE_REPRESENTATION_TRAINING"
    assert result["brier_gain_ci"]["upper"] < 0.0
    assert set(result["by_family"]) == {"compose", "lookup"}
