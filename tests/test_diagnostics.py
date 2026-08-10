from __future__ import annotations

import json

import numpy as np

from frank_eq.data.real import RealBundle
from frank_eq.diagnostics import diagnose_real_cache
from frank_eq.schemas import OperationDefinition, SplitManifest
from frank_eq.workflow import parse_real_stages


def _bundle() -> RealBundle:
    rng = np.random.default_rng(7)
    n_worlds = 30
    n_models = 3
    n_renderers = 2
    n_layers = 2
    n_facts = 4
    n_residual = 2
    n_operations = 4
    facts_by_world = rng.integers(0, 2, size=(n_worlds, n_facts)).astype(np.float32)
    residual_by_world = rng.normal(size=(n_worlds, n_residual)).astype(np.float32)
    signatures_by_world = np.stack(
        [
            0.02 + 0.96 * facts_by_world[:, 0],
            0.02 + 0.96 * facts_by_world[:, 1],
            1.0 / (1.0 + np.exp(-residual_by_world[:, 0])),
            1.0 / (1.0 + np.exp(-residual_by_world[:, 1])),
        ],
        axis=1,
    ).astype(np.float32)

    rows = n_worlds * n_models * n_renderers
    hidden = np.zeros((rows, n_layers, 8), dtype=np.float32)
    hidden_mask = np.ones_like(hidden, dtype=np.bool_)
    world_ids = np.empty(rows, dtype=np.int64)
    model_ids = np.empty(rows, dtype=np.int64)
    renderer_ids = np.empty(rows, dtype=np.int64)
    facts = np.empty((rows, n_facts), dtype=np.float32)
    residual = np.empty((rows, n_residual), dtype=np.float32)
    signatures = np.empty((rows, n_operations), dtype=np.float32)
    model_signatures = np.empty_like(signatures)
    cursor = 0
    for world in range(n_worlds):
        public = np.concatenate([facts_by_world[world], residual_by_world[world]])
        for model in range(n_models):
            transform = np.roll(np.eye(8, dtype=np.float32), model, axis=1)
            for renderer in range(n_renderers):
                vector = np.pad(public, (0, 2)) @ transform
                vector += 0.001 * renderer
                hidden[cursor, 0] = vector
                hidden[cursor, 1] = 0.7 * vector
                world_ids[cursor] = world
                model_ids[cursor] = model
                renderer_ids[cursor] = renderer
                facts[cursor] = facts_by_world[world]
                residual[cursor] = residual_by_world[world]
                signatures[cursor] = signatures_by_world[world]
                model_signatures[cursor] = signatures_by_world[world]
                cursor += 1

    operations = [
        OperationDefinition(
            operation_id=index,
            family=("lookup" if index < 2 else "residual"),
            fact_args=(0, 1),
            residual_args=(0, 1),
            polarity=1.0,
        )
        for index in range(n_operations)
    ]
    descriptors = np.eye(n_operations, dtype=np.float32)
    split = SplitManifest(
        train_world_ids=tuple(range(18)),
        validation_world_ids=tuple(range(18, 24)),
        test_world_ids=tuple(range(24, 30)),
        train_operation_ids=(0, 2),
        heldout_operation_ids=(1, 3),
        founder_model_ids=(0, 1),
        held_model_id=2,
    )
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
        operation_descriptors=descriptors,
        operations=operations,
        split=split,
        model_hidden_dims=[8, 8, 8],
        n_layers=n_layers,
        model_names=["founder-a", "founder-b", "held-c"],
    )


def test_diagnostic_uses_train_and_validation_only(tmp_path) -> None:
    bundle = _bundle()
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez_compressed(
        cache / "dataset.npz",
        world_ids=bundle.world_ids,
        model_ids=bundle.model_ids,
        renderer_ids=bundle.renderer_ids,
        hidden=bundle.hidden,
        hidden_mask=bundle.hidden_mask,
        facts=bundle.facts,
        residual=bundle.residual,
        signatures=bundle.signatures,
        model_signatures=bundle.model_signatures,
        operation_descriptors=bundle.operation_descriptors,
    )
    metadata = {
        "schema": "frank_eq_real_bundle_v1",
        "task_family": bundle.task_family,
        "scope": bundle.scope,
        "model_hidden_dims": bundle.model_hidden_dims,
        "model_names": bundle.model_names,
        "n_layers": bundle.n_layers,
        "operations": [operation.to_dict() for operation in bundle.operations],
        "split": bundle.split.to_dict(),
    }
    (cache / "metadata.json").write_text(json.dumps(metadata))
    (cache / "panel.json").write_text("{}")
    (cache / "future_signature_records.jsonl").write_text("")

    report = diagnose_real_cache(cache, tmp_path / "diagnostics", ridge=1.0)
    assert report["data_usage"]["test_worlds_used"] == 0
    assert report["data_usage"]["test_labels_consumed"] is False
    assert report["authorizes_new_outcome_run"] is False
    assert (tmp_path / "diagnostics" / "localization.json").is_file()
    for model in report["models"]:
        assert model["readability"]["facts"]["best"]["brier_gain_over_coordinate_prior"] > 0.0
        assert (
            model["readability"]["self_signature"]["best"][
                "brier_gain_over_coordinate_prior"
            ]
            > 0.0
        )


def test_real_stage_order_accepts_nonpromotional_diagnostic() -> None:
    assert parse_real_stages("cache,validate,diagnose") == (
        "cache",
        "validate",
        "diagnose",
    )
    assert parse_real_stages("cache,validate,train,eval") == (
        "cache",
        "validate",
        "train",
        "eval",
    )
