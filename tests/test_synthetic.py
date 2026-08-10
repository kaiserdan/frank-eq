from pathlib import Path

import numpy as np

from frank_eq.config import DataConfig
from frank_eq.data.synthetic import SyntheticBundle, generate_synthetic_bundle


def small_config() -> DataConfig:
    return DataConfig(
        n_worlds=36,
        n_founder_models=2,
        include_held_model=True,
        n_renderers=2,
        n_layers=3,
        model_hidden_dims=[16, 20, 18],
        n_facts=6,
        n_residual=2,
        n_operations=24,
        seed=7,
    )


def test_synthetic_generator_is_deterministic() -> None:
    first = generate_synthetic_bundle(small_config())
    second = generate_synthetic_bundle(small_config())
    np.testing.assert_array_equal(first.hidden, second.hidden)
    np.testing.assert_array_equal(first.signatures, second.signatures)
    assert first.n_views == 36 * 3 * 2
    assert first.hidden.shape[1] == 3


def test_bundle_roundtrip(tmp_path: Path) -> None:
    bundle = generate_synthetic_bundle(small_config())
    bundle.save(tmp_path)
    loaded = SyntheticBundle.load(tmp_path)
    np.testing.assert_allclose(loaded.hidden, bundle.hidden)
    np.testing.assert_allclose(loaded.operation_descriptors, bundle.operation_descriptors)
    assert loaded.split == bundle.split
