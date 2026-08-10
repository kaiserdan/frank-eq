"""Controlled multi-model benchmark for future-defined causal states.

The generator creates one public latent world, multiple model-specific hidden
charts, multiple renderer-specific nuisance views, grounded facts, and a bank of
future operations. It is intentionally synthetic: its purpose is to validate
contracts and falsification gates before any expensive real-LLM campaign.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from frank_eq.config import DataConfig
from frank_eq.schemas import OperationDefinition, SplitManifest
from frank_eq.utils import atomic_write_json

from .split import build_split_manifest

OPERATION_FAMILIES = ("lookup", "xor", "and", "implication", "residual", "hybrid")


@dataclass(slots=True)
class SyntheticBundle:
    """In-memory Stage-0 dataset and public operation registry."""

    world_ids: np.ndarray
    model_ids: np.ndarray
    renderer_ids: np.ndarray
    hidden: np.ndarray
    hidden_mask: np.ndarray
    facts: np.ndarray
    residual: np.ndarray
    signatures: np.ndarray
    operation_descriptors: np.ndarray
    operations: list[OperationDefinition]
    split: SplitManifest
    model_hidden_dims: list[int]
    n_layers: int

    @property
    def n_views(self) -> int:
        return int(self.world_ids.shape[0])

    @property
    def max_hidden_dim(self) -> int:
        return int(self.hidden.shape[-1])

    def indices_for(
        self,
        *,
        world_ids: tuple[int, ...] | list[int] | np.ndarray,
        model_ids: tuple[int, ...] | list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        world_mask = np.isin(self.world_ids, np.asarray(world_ids, dtype=np.int64))
        if model_ids is not None:
            world_mask &= np.isin(self.model_ids, np.asarray(model_ids, dtype=np.int64))
        return np.flatnonzero(world_mask)

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "dataset.npz",
            world_ids=self.world_ids,
            model_ids=self.model_ids,
            renderer_ids=self.renderer_ids,
            hidden=self.hidden,
            hidden_mask=self.hidden_mask,
            facts=self.facts,
            residual=self.residual,
            signatures=self.signatures,
            operation_descriptors=self.operation_descriptors,
        )
        atomic_write_json(
            target / "metadata.json",
            {
                "schema": "frank_eq_synthetic_v1",
                "model_hidden_dims": self.model_hidden_dims,
                "n_layers": self.n_layers,
                "operations": [operation.to_dict() for operation in self.operations],
                "split": self.split.to_dict(),
            },
        )

    @classmethod
    def load(cls, directory: str | Path) -> SyntheticBundle:
        source = Path(directory)
        if not (source / "dataset.npz").is_file() or not (source / "metadata.json").is_file():
            raise FileNotFoundError(f"incomplete synthetic bundle: {source}")
        arrays = np.load(source / "dataset.npz")
        import json

        metadata = json.loads((source / "metadata.json").read_text())
        if metadata.get("schema") != "frank_eq_synthetic_v1":
            raise ValueError("unsupported synthetic bundle schema")
        return cls(
            world_ids=arrays["world_ids"],
            model_ids=arrays["model_ids"],
            renderer_ids=arrays["renderer_ids"],
            hidden=arrays["hidden"],
            hidden_mask=arrays["hidden_mask"],
            facts=arrays["facts"],
            residual=arrays["residual"],
            signatures=arrays["signatures"],
            operation_descriptors=arrays["operation_descriptors"],
            operations=[OperationDefinition.from_dict(item) for item in metadata["operations"]],
            split=SplitManifest.from_dict(metadata["split"]),
            model_hidden_dims=[int(v) for v in metadata["model_hidden_dims"]],
            n_layers=int(metadata["n_layers"]),
        )


class ObservationDataset(Dataset[dict[str, torch.Tensor]]):
    """Torch view over selected observations in a synthetic bundle."""

    def __init__(self, bundle: SyntheticBundle, indices: np.ndarray):
        self.bundle = bundle
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        return {
            "row_index": torch.tensor(row, dtype=torch.long),
            "world_id": torch.tensor(self.bundle.world_ids[row], dtype=torch.long),
            "model_id": torch.tensor(self.bundle.model_ids[row], dtype=torch.long),
            "renderer_id": torch.tensor(self.bundle.renderer_ids[row], dtype=torch.long),
            "hidden": torch.from_numpy(self.bundle.hidden[row]).float(),
            "hidden_mask": torch.from_numpy(self.bundle.hidden_mask[row]).bool(),
            "facts": torch.from_numpy(self.bundle.facts[row]).float(),
            "residual": torch.from_numpy(self.bundle.residual[row]).float(),
            "signatures": torch.from_numpy(self.bundle.signatures[row]).float(),
        }


class WorldBatchSampler(Sampler[list[int]]):
    """Batch complete model/renderer view groups by world ID."""

    def __init__(
        self,
        dataset: ObservationDataset,
        worlds_per_batch: int,
        *,
        shuffle: bool,
        seed: int,
    ):
        if worlds_per_batch < 1:
            raise ValueError("worlds_per_batch must be positive")
        self.dataset = dataset
        self.worlds_per_batch = worlds_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._positions_by_world: dict[int, list[int]] = {}
        for position, row in enumerate(dataset.indices):
            world = int(dataset.bundle.world_ids[row])
            self._positions_by_world.setdefault(world, []).append(position)
        self._worlds = np.asarray(sorted(self._positions_by_world), dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        worlds = self._worlds.copy()
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(worlds)
        for start in range(0, len(worlds), self.worlds_per_batch):
            batch_worlds = worlds[start : start + self.worlds_per_batch]
            positions: list[int] = []
            for world in batch_worlds:
                positions.extend(self._positions_by_world[int(world)])
            yield positions

    def __len__(self) -> int:
        return int(np.ceil(len(self._worlds) / self.worlds_per_batch))


def public_basis_dimension(n_facts: int, n_residual: int) -> int:
    return n_facts + (n_facts * (n_facts - 1)) // 2 + n_residual + 1


def operation_descriptor_dimension(n_facts: int, n_residual: int) -> int:
    return len(OPERATION_FAMILIES) + public_basis_dimension(n_facts, n_residual)


def _pair_index(n_facts: int, i: int, j: int) -> int:
    if i == j:
        raise ValueError("pair indices must differ")
    if i > j:
        i, j = j, i
    offset = 0
    for left in range(i):
        offset += n_facts - left - 1
    return offset + (j - i - 1)


def _public_basis(facts: np.ndarray, residual: np.ndarray) -> np.ndarray:
    signed = 2.0 * np.asarray(facts, dtype=np.float32) - 1.0
    pair_terms = [signed[:, i] * signed[:, j] for i in range(signed.shape[1]) for j in range(i + 1, signed.shape[1])]
    pairs = np.stack(pair_terms, axis=1) if pair_terms else np.zeros((len(signed), 0), dtype=np.float32)
    constant = np.ones((len(signed), 1), dtype=np.float32)
    return np.concatenate([signed, pairs, np.asarray(residual, dtype=np.float32), constant], axis=1)


def _make_operations(config: DataConfig, rng: np.random.Generator) -> tuple[list[OperationDefinition], np.ndarray]:
    operations: list[OperationDefinition] = []
    basis_dim = public_basis_dimension(config.n_facts, config.n_residual)
    descriptor_dim = len(OPERATION_FAMILIES) + basis_dim
    descriptors = np.zeros((config.n_operations, descriptor_dim), dtype=np.float32)
    pair_offset = config.n_facts
    residual_offset = config.n_facts + (config.n_facts * (config.n_facts - 1)) // 2
    constant_offset = residual_offset + config.n_residual
    coefficient_offset = len(OPERATION_FAMILIES)

    for operation_id in range(config.n_operations):
        family = OPERATION_FAMILIES[operation_id % len(OPERATION_FAMILIES)]
        fact_args = tuple(int(v) for v in rng.choice(config.n_facts, size=2, replace=False))
        if config.n_residual >= 2:
            residual_args = tuple(int(v) for v in rng.choice(config.n_residual, size=2, replace=False))
        else:
            residual_args = (0, 0)
        polarity = float(rng.choice([-1.0, 1.0]))
        operation = OperationDefinition(
            operation_id=operation_id,
            family=family,
            fact_args=fact_args,
            residual_args=residual_args,
            polarity=polarity,
        )
        operations.append(operation)

        descriptor = descriptors[operation_id]
        descriptor[OPERATION_FAMILIES.index(family)] = 1.0
        coefficients = descriptor[coefficient_offset:]
        i, j = fact_args
        a, b = residual_args
        pair = pair_offset + _pair_index(config.n_facts, i, j)
        scale = 2.4
        if family == "lookup":
            coefficients[i] = scale * polarity
        elif family == "xor":
            coefficients[pair] = -scale * polarity
        elif family == "and":
            coefficients[i] = scale * polarity
            coefficients[j] = scale * polarity
            coefficients[pair] = scale * polarity
            coefficients[constant_offset] = -scale * polarity
        elif family == "implication":
            coefficients[constant_offset] = scale * polarity
            coefficients[i] = -scale * polarity
            coefficients[j] = scale * polarity
            coefficients[pair] = scale * polarity
        elif family == "residual":
            coefficients[residual_offset + a] = 1.8 * polarity
            coefficients[i] = 0.8 * polarity
        elif family == "hybrid":
            coefficients[residual_offset + a] = 1.4 * polarity
            coefficients[residual_offset + b] = -1.4 * polarity
            coefficients[pair] = 0.8 * polarity
        else:
            raise ValueError(f"unknown operation family: {family}")
    return operations, descriptors


def compute_signatures(
    facts: np.ndarray,
    residual: np.ndarray,
    operation_descriptors: np.ndarray,
) -> np.ndarray:
    """Compute the public future causal signature for every world."""

    basis = _public_basis(facts, residual)
    coefficients = np.asarray(operation_descriptors, dtype=np.float32)[:, len(OPERATION_FAMILIES) :]
    logits = basis @ coefficients.T
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))).astype(np.float32)


def facts_only_signatures(
    fact_probabilities: np.ndarray,
    operation_descriptors: np.ndarray,
    n_residual: int,
) -> np.ndarray:
    """Predict signatures from explicit facts while marginalizing residual state."""

    probabilities = np.asarray(fact_probabilities, dtype=np.float32)
    residual = np.zeros((len(probabilities), n_residual), dtype=np.float32)
    basis = _public_basis(probabilities, residual)
    coefficients = np.asarray(operation_descriptors, dtype=np.float32)[:, len(OPERATION_FAMILIES) :]
    logits = basis @ coefficients.T
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))).astype(np.float32)


def _base_features(facts: np.ndarray, residual: np.ndarray) -> np.ndarray:
    basis = _public_basis(facts, residual)
    residual_square = residual**2 - 1.0
    return np.concatenate([basis, residual_square], axis=1).astype(np.float32)

def generate_synthetic_bundle(config: DataConfig) -> SyntheticBundle:
    """Generate a deterministic multi-model future-signature benchmark."""

    if len(config.model_hidden_dims) < config.n_models:
        raise ValueError("not enough model_hidden_dims for configured models")
    rng = np.random.default_rng(config.seed)
    facts = rng.integers(0, 2, size=(config.n_worlds, config.n_facts), dtype=np.int8).astype(np.float32)
    residual = rng.normal(0.0, 0.8, size=(config.n_worlds, config.n_residual)).astype(np.float32)
    operations, descriptors = _make_operations(config, rng)
    signatures_by_world = compute_signatures(facts, residual, descriptors)
    base = _base_features(facts, residual)
    base_dim = base.shape[1]

    max_hidden = max(config.model_hidden_dims[: config.n_models])
    n_views = config.n_worlds * config.n_models * config.n_renderers
    hidden = np.zeros((n_views, config.n_layers, max_hidden), dtype=np.float32)
    hidden_mask = np.zeros_like(hidden, dtype=np.bool_)
    world_ids = np.empty(n_views, dtype=np.int64)
    model_ids = np.empty(n_views, dtype=np.int64)
    renderer_ids = np.empty(n_views, dtype=np.int64)
    view_facts = np.empty((n_views, config.n_facts), dtype=np.float32)
    view_residual = np.empty((n_views, config.n_residual), dtype=np.float32)
    view_signatures = np.empty((n_views, config.n_operations), dtype=np.float32)

    surface_dim = max(6, config.n_facts // 2)
    renderer_codes = rng.normal(
        0.0,
        1.0,
        size=(config.n_worlds, config.n_renderers, surface_dim),
    ).astype(np.float32)

    row = 0
    for model_id in range(config.n_models):
        hidden_dim = config.model_hidden_dims[model_id]
        model_rng = np.random.default_rng(config.seed + 1009 * (model_id + 1))
        private_projection = model_rng.normal(0.0, 1.0 / np.sqrt(base_dim), size=(base_dim, hidden_dim))
        layer_projections = [
            model_rng.normal(0.0, 1.0 / np.sqrt(base_dim), size=(base_dim, hidden_dim))
            for _ in range(config.n_layers)
        ]
        renderer_projections = [
            model_rng.normal(0.0, 1.0 / np.sqrt(surface_dim), size=(surface_dim, hidden_dim))
            for _ in range(config.n_layers)
        ]
        layer_biases = [model_rng.normal(0.0, 0.1, size=(hidden_dim,)) for _ in range(config.n_layers)]
        private = np.tanh(base @ private_projection).astype(np.float32)

        for world_id in range(config.n_worlds):
            for renderer_id in range(config.n_renderers):
                for layer_id in range(config.n_layers):
                    depth = (layer_id + 1) / config.n_layers
                    fact_residual_balance = 0.35 + 0.65 * depth
                    transformed_base = base[world_id].copy()
                    residual_start = config.n_facts
                    transformed_base[residual_start : residual_start + config.n_residual] *= fact_residual_balance
                    value = transformed_base @ layer_projections[layer_id]
                    value += (
                        config.renderer_nuisance_scale
                        * (1.15 - 0.35 * depth)
                        * renderer_codes[world_id, renderer_id]
                        @ renderer_projections[layer_id]
                    )
                    value += config.model_private_scale * private[world_id]
                    value += layer_biases[layer_id]
                    value += rng.normal(0.0, config.observation_noise, size=value.shape)
                    hidden[row, layer_id, :hidden_dim] = np.tanh(value).astype(np.float32)
                    hidden_mask[row, layer_id, :hidden_dim] = True
                world_ids[row] = world_id
                model_ids[row] = model_id
                renderer_ids[row] = renderer_id
                view_facts[row] = facts[world_id]
                view_residual[row] = residual[world_id]
                view_signatures[row] = signatures_by_world[world_id]
                row += 1

    split = build_split_manifest(config, operations)
    return SyntheticBundle(
        world_ids=world_ids,
        model_ids=model_ids,
        renderer_ids=renderer_ids,
        hidden=hidden,
        hidden_mask=hidden_mask,
        facts=view_facts,
        residual=view_residual,
        signatures=view_signatures,
        operation_descriptors=descriptors,
        operations=operations,
        split=split,
        model_hidden_dims=config.model_hidden_dims[: config.n_models],
        n_layers=config.n_layers,
    )
