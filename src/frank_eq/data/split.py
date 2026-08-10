"""Leakage-resistant world and operation split construction."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from frank_eq.config import DataConfig
from frank_eq.schemas import OperationDefinition, SplitManifest


def build_split_manifest(
    config: DataConfig,
    operations: list[OperationDefinition],
) -> SplitManifest:
    """Create disjoint world splits and family-stratified operation holdout.

    All views of a world remain in exactly one split. Operations are held out
    within every family so the shared operation-conditioned decoder is tested on
    unseen operation instances rather than merely unseen worlds.
    """

    rng = np.random.default_rng(config.seed + 17)
    world_ids = np.arange(config.n_worlds, dtype=np.int64)
    rng.shuffle(world_ids)

    train_end = int(round(config.train_fraction * config.n_worlds))
    validation_end = train_end + int(round(config.validation_fraction * config.n_worlds))
    train_worlds = np.sort(world_ids[:train_end])
    validation_worlds = np.sort(world_ids[train_end:validation_end])
    test_worlds = np.sort(world_ids[validation_end:])
    if len(test_worlds) < 3:
        raise ValueError("grouped split produced fewer than three test worlds")

    by_family: dict[str, list[int]] = defaultdict(list)
    for operation in operations:
        by_family[operation.family].append(operation.operation_id)

    heldout: list[int] = []
    train_operations: list[int] = []
    for family, ids in sorted(by_family.items()):
        family_ids = np.asarray(ids, dtype=np.int64)
        rng.shuffle(family_ids)
        n_holdout = max(1, int(round(config.operation_holdout_fraction * len(family_ids))))
        if n_holdout >= len(family_ids):
            raise ValueError(f"operation family {family!r} is too small for a holdout")
        heldout.extend(int(v) for v in family_ids[:n_holdout])
        train_operations.extend(int(v) for v in family_ids[n_holdout:])

    founder_ids = tuple(range(config.n_founder_models))
    held_model_id = config.n_founder_models if config.include_held_model else None
    manifest = SplitManifest(
        train_world_ids=tuple(int(v) for v in train_worlds),
        validation_world_ids=tuple(int(v) for v in validation_worlds),
        test_world_ids=tuple(int(v) for v in test_worlds),
        train_operation_ids=tuple(sorted(train_operations)),
        heldout_operation_ids=tuple(sorted(heldout)),
        founder_model_ids=founder_ids,
        held_model_id=held_model_id,
    )
    validate_split_manifest(manifest, config.n_worlds, len(operations))
    return manifest


def validate_split_manifest(
    manifest: SplitManifest,
    n_worlds: int,
    n_operations: int,
) -> None:
    world_sets = [
        set(manifest.train_world_ids),
        set(manifest.validation_world_ids),
        set(manifest.test_world_ids),
    ]
    if any(a & b for index, a in enumerate(world_sets) for b in world_sets[index + 1 :]):
        raise ValueError("world-group leakage detected across splits")
    if set.union(*world_sets) != set(range(n_worlds)):
        raise ValueError("world split does not cover the full population")

    train_ops = set(manifest.train_operation_ids)
    held_ops = set(manifest.heldout_operation_ids)
    if train_ops & held_ops:
        raise ValueError("operation leakage detected across train and holdout")
    if train_ops | held_ops != set(range(n_operations)):
        raise ValueError("operation split does not cover all operations")
