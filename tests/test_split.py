from frank_eq.config import DataConfig
from frank_eq.data.split import build_split_manifest, validate_split_manifest
from frank_eq.data.synthetic import generate_synthetic_bundle


def test_world_and_operation_splits_are_disjoint() -> None:
    config = DataConfig(
        n_worlds=48,
        n_founder_models=2,
        include_held_model=True,
        n_renderers=2,
        model_hidden_dims=[16, 20, 18],
        n_facts=6,
        n_residual=2,
        n_operations=24,
    )
    bundle = generate_synthetic_bundle(config)
    manifest = build_split_manifest(config, bundle.operations)
    validate_split_manifest(manifest, config.n_worlds, config.n_operations)
    assert set(manifest.train_world_ids).isdisjoint(manifest.test_world_ids)
    assert set(manifest.train_operation_ids).isdisjoint(manifest.heldout_operation_ids)
    assert manifest.held_model_id == 2
