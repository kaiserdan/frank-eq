"""Frozen controlled relational worlds for the real-checkpoint Stage-A canary.

The source prefix contains only a world description. Future operations are
registered separately and are revealed strictly after the hidden-state capture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from frank_eq.real_config import GRAPH_OPERATION_FAMILIES, RealPanelConfig
from frank_eq.schemas import OperationDefinition
from frank_eq.utils import canonical_json_bytes, sha256_bytes

ENTITY_NAMES = (
    "Aster",
    "Birch",
    "Cedar",
    "Dahlia",
    "Elm",
    "Fir",
    "Ginkgo",
    "Hazel",
    "Iris",
    "Juniper",
)


@dataclass(frozen=True, slots=True)
class RelationalWorld:
    """One directed closed-world graph and two public global coordinates."""

    world_id: int
    edges: tuple[tuple[int, ...], ...]
    density_score: float
    reciprocity_score: float

    @property
    def n_entities(self) -> int:
        return len(self.edges)

    def edge_array(self) -> np.ndarray:
        return np.asarray(self.edges, dtype=np.int8)

    def fact_vector(self) -> np.ndarray:
        edge = self.edge_array()
        return np.asarray(
            [edge[i, j] for i in range(self.n_entities) for j in range(self.n_entities) if i != j],
            dtype=np.float32,
        )

    def residual_vector(self) -> np.ndarray:
        return np.asarray([self.density_score, self.reciprocity_score], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RelationalWorld":
        return cls(
            world_id=int(payload["world_id"]),
            edges=tuple(tuple(int(value) for value in row) for row in payload["edges"]),
            density_score=float(payload["density_score"]),
            reciprocity_score=float(payload["reciprocity_score"]),
        )


@dataclass(frozen=True, slots=True)
class FrozenOperation:
    """Public operation registry row and its external descriptor."""

    definition: OperationDefinition
    descriptor: tuple[float, ...]
    descriptor_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition.to_dict(),
            "descriptor": list(self.descriptor),
            "descriptor_sha256": self.descriptor_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrozenOperation":
        return cls(
            definition=OperationDefinition.from_dict(payload["definition"]),
            descriptor=tuple(float(value) for value in payload["descriptor"]),
            descriptor_sha256=str(payload["descriptor_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class RealPanel:
    """Worlds, operations, exact labels, and renderer templates."""

    worlds: tuple[RelationalWorld, ...]
    operations: tuple[FrozenOperation, ...]
    oracle_signatures: tuple[tuple[float, ...], ...]
    config: dict[str, Any]
    schema: str = "frank_eq_real_panel_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "config": self.config,
            "worlds": [world.to_dict() for world in self.worlds],
            "operations": [operation.to_dict() for operation in self.operations],
            "oracle_signatures": [list(row) for row in self.oracle_signatures],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RealPanel":
        if payload.get("schema") != "frank_eq_real_panel_v1":
            raise ValueError("unsupported real-panel schema")
        panel = cls(
            worlds=tuple(RelationalWorld.from_dict(item) for item in payload["worlds"]),
            operations=tuple(FrozenOperation.from_dict(item) for item in payload["operations"]),
            oracle_signatures=tuple(
                tuple(float(value) for value in row) for row in payload["oracle_signatures"]
            ),
            config=dict(payload["config"]),
        )
        panel.validate()
        return panel

    @property
    def n_entities(self) -> int:
        return int(self.config["n_entities"])

    def validate(self) -> None:
        if not self.worlds or not self.operations:
            raise ValueError("real panel must contain worlds and operations")
        if len(self.oracle_signatures) != len(self.worlds):
            raise ValueError("oracle signature row count does not match worlds")
        operation_ids = [row.definition.operation_id for row in self.operations]
        if operation_ids != list(range(len(self.operations))):
            raise ValueError("operation IDs must be contiguous and ordered")
        for operation in self.operations:
            expected = sha256_bytes(
                canonical_json_bytes(
                    {
                        "definition": operation.definition.to_dict(),
                        "descriptor": list(operation.descriptor),
                    }
                )
            )
            if operation.descriptor_sha256 != expected:
                raise ValueError("operation descriptor hash mismatch")
        signatures = np.asarray(self.oracle_signatures, dtype=np.float64)
        if signatures.shape != (len(self.worlds), len(self.operations)):
            raise ValueError("oracle signature matrix has the wrong shape")
        if not np.all(np.isfinite(signatures)) or np.any(signatures <= 0) or np.any(signatures >= 1):
            raise ValueError("oracle signatures must be finite probabilities strictly inside (0,1)")


def edge_fact_index(n_entities: int, source: int, target: int) -> int:
    """Index a non-diagonal directed edge in canonical row-major order."""

    if source == target:
        raise ValueError("self edges are not represented")
    if not 0 <= source < n_entities or not 0 <= target < n_entities:
        raise IndexError("entity index outside graph")
    return source * (n_entities - 1) + (target if target < source else target - 1)


def operation_descriptor_dimension(n_entities: int) -> int:
    return len(GRAPH_OPERATION_FAMILIES) + 4 * n_entities + 1


def build_operation_descriptor(
    definition: OperationDefinition,
    n_entities: int,
) -> np.ndarray:
    descriptor = np.zeros(operation_descriptor_dimension(n_entities), dtype=np.float32)
    descriptor[GRAPH_OPERATION_FAMILIES.index(definition.family)] = 1.0
    offset = len(GRAPH_OPERATION_FAMILIES)
    arguments = (
        definition.fact_args[0],
        definition.fact_args[1],
        definition.residual_args[0],
        definition.residual_args[1],
    )
    for block, argument in enumerate(arguments):
        descriptor[offset + block * n_entities + int(argument)] = 1.0
    descriptor[-1] = float(definition.polarity)
    return descriptor


def _sample_operations(config: RealPanelConfig, rng: np.random.Generator) -> list[FrozenOperation]:
    per_family = config.n_operations // len(GRAPH_OPERATION_FAMILIES)
    rows: list[FrozenOperation] = []
    operation_id = 0
    for family in GRAPH_OPERATION_FAMILIES:
        for local_index in range(per_family):
            arguments = rng.choice(config.n_entities, size=4, replace=False)
            source, target, aux_source, aux_target = (int(value) for value in arguments)
            polarity = 1.0 if local_index % 2 == 0 else -1.0
            definition = OperationDefinition(
                operation_id=operation_id,
                family=family,
                fact_args=(source, target),
                residual_args=(aux_source, aux_target),
                polarity=polarity,
            )
            descriptor = build_operation_descriptor(definition, config.n_entities)
            digest = sha256_bytes(
                canonical_json_bytes(
                    {"definition": definition.to_dict(), "descriptor": descriptor.tolist()}
                )
            )
            rows.append(
                FrozenOperation(
                    definition=definition,
                    descriptor=tuple(float(value) for value in descriptor),
                    descriptor_sha256=digest,
                )
            )
            operation_id += 1
    return rows


def _sample_edge_matrix(n_entities: int, rng: np.random.Generator) -> np.ndarray:
    """Sample reciprocal-aware directed graphs with non-degenerate operations."""

    edge = np.zeros((n_entities, n_entities), dtype=np.int8)
    probabilities = np.asarray([0.33, 0.23, 0.23, 0.21], dtype=np.float64)
    for left in range(n_entities):
        for right in range(left + 1, n_entities):
            state = int(rng.choice(4, p=probabilities))
            if state == 1:
                edge[left, right] = 1
            elif state == 2:
                edge[right, left] = 1
            elif state == 3:
                edge[left, right] = 1
                edge[right, left] = 1
    return edge


def _raw_world_statistics(edge: np.ndarray) -> tuple[float, float]:
    n_entities = edge.shape[0]
    possible = n_entities * (n_entities - 1)
    density = float(edge.sum() / possible)
    mutual_pairs = sum(
        int(edge[left, right] and edge[right, left])
        for left in range(n_entities)
        for right in range(left + 1, n_entities)
    )
    reciprocity = float(mutual_pairs / max(1, n_entities * (n_entities - 1) / 2))
    return density, reciprocity


def _path_probability_truth(edge: np.ndarray, source: int, target: int) -> bool:
    n_entities = edge.shape[0]
    return any(
        bool(edge[source, middle] and edge[middle, target])
        for middle in range(n_entities)
        if middle not in {source, target}
    )


def evaluate_operation(world: RelationalWorld, operation: OperationDefinition) -> bool:
    edge = world.edge_array().copy()
    source, target = operation.fact_args
    aux_source, aux_target = operation.residual_args
    family = operation.family
    if family == "lookup":
        result = bool(edge[source, target])
    elif family == "inverse":
        result = bool(edge[target, source])
    elif family == "mutual":
        result = bool(edge[source, target] and edge[target, source])
    elif family == "compose":
        result = _path_probability_truth(edge, source, target)
    elif family == "compare_outdegree":
        result = int(edge[source].sum()) > int(edge[target].sum())
    elif family == "counterfactual_add":
        edge[source, target] = 1
        result = _path_probability_truth(edge, aux_source, aux_target)
    elif family == "density":
        result = world.density_score > 0.0
    elif family == "reciprocity":
        result = world.reciprocity_score > 0.0
    else:
        raise ValueError(f"unknown graph operation family: {family}")
    return result if operation.polarity >= 0 else not result


def _candidate_worlds(config: RealPanelConfig, rng: np.random.Generator) -> list[np.ndarray]:
    return [_sample_edge_matrix(config.n_entities, rng) for _ in range(config.n_worlds)]


def _center_worlds(edges: list[np.ndarray]) -> list[RelationalWorld]:
    statistics = np.asarray([_raw_world_statistics(edge) for edge in edges], dtype=np.float64)
    medians = np.median(statistics, axis=0)
    scales = np.std(statistics, axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)
    return [
        RelationalWorld(
            world_id=index,
            edges=tuple(tuple(int(value) for value in row) for row in edge.tolist()),
            density_score=float((statistics[index, 0] - medians[0]) / scales[0]),
            reciprocity_score=float((statistics[index, 1] - medians[1]) / scales[1]),
        )
        for index, edge in enumerate(edges)
    ]


def generate_real_panel(config: RealPanelConfig) -> RealPanel:
    """Generate a deterministic panel and reject pathological operation balance."""

    operation_rng = np.random.default_rng(config.seed + 13)
    operations = _sample_operations(config, operation_rng)
    last_fractions: np.ndarray | None = None
    for attempt in range(config.max_generation_attempts):
        world_rng = np.random.default_rng(config.seed + 1009 * (attempt + 1))
        worlds = _center_worlds(_candidate_worlds(config, world_rng))
        labels = np.asarray(
            [
                [evaluate_operation(world, operation.definition) for operation in operations]
                for world in worlds
            ],
            dtype=np.float32,
        )
        fractions = labels.mean(axis=0)
        last_fractions = fractions
        if np.all(fractions >= config.min_operation_positive_fraction) and np.all(
            fractions <= config.max_operation_positive_fraction
        ):
            smooth = config.oracle_smoothing
            signatures = labels * (1.0 - 2.0 * smooth) + smooth
            panel = RealPanel(
                worlds=tuple(worlds),
                operations=tuple(operations),
                oracle_signatures=tuple(
                    tuple(float(value) for value in row) for row in signatures.tolist()
                ),
                config=asdict(config),
            )
            panel.validate()
            return panel
    raise RuntimeError(
        "failed to generate a balanced operation panel; final positive fractions="
        f"{None if last_fractions is None else last_fractions.tolist()}"
    )


def render_world_prefix(world: RelationalWorld, renderer_id: int) -> str:
    """Render the same closed-world graph through two surface forms."""

    if not 0 <= renderer_id < 2:
        raise ValueError("the current real panel defines renderer IDs 0 and 1")
    names = ENTITY_NAMES[: world.n_entities]
    edge = world.edge_array()
    positive = [
        (names[source], names[target])
        for source in range(world.n_entities)
        for target in range(world.n_entities)
        if source != target and edge[source, target]
    ]
    density_label = "high" if world.density_score > 0 else "low"
    reciprocity_label = "high" if world.reciprocity_score > 0 else "low"
    header = (
        "Study the following closed directed world. A directed relation not listed is false. "
        "No question has been selected yet; later you will receive one registered operation."
    )
    if renderer_id == 0:
        statements = [f"{left} points to {right}." for left, right in positive]
        body = "\n".join(statements)
        globals_text = (
            f"The declared graph density class is {density_label}. "
            f"The declared reciprocity class is {reciprocity_label}."
        )
        return f"{header}\nEntities: {', '.join(names)}.\n{body}\n{globals_text}\n"

    adjacency: list[str] = []
    for source, name in enumerate(names):
        targets = [names[target] for target in range(world.n_entities) if edge[source, target]]
        adjacency.append(f"{name} -> {', '.join(targets) if targets else 'none'}")
    adjacency.reverse()
    return (
        f"{header}\nNode roster: {' | '.join(reversed(names))}.\n"
        + "\n".join(adjacency)
        + f"\nGlobal tags: density={density_label}; reciprocity={reciprocity_label}.\n"
    )


def _base_operation_clause(operation: OperationDefinition, n_entities: int) -> str:
    names = ENTITY_NAMES[:n_entities]
    source, target = operation.fact_args
    aux_source, aux_target = operation.residual_args
    if operation.family == "lookup":
        return f"{names[source]} directly points to {names[target]}"
    if operation.family == "inverse":
        return f"{names[target]} directly points to {names[source]}"
    if operation.family == "mutual":
        return f"{names[source]} and {names[target]} point to each other"
    if operation.family == "compose":
        return (
            f"there exists a third entity reached from {names[source]} that directly points to "
            f"{names[target]}"
        )
    if operation.family == "compare_outdegree":
        return f"{names[source]} points to more entities than {names[target]}"
    if operation.family == "counterfactual_add":
        return (
            f"after adding the relation {names[source]} -> {names[target]}, there exists a third "
            f"entity reached from {names[aux_source]} that directly points to {names[aux_target]}"
        )
    if operation.family == "density":
        return "the declared graph density class is high"
    if operation.family == "reciprocity":
        return "the declared graph reciprocity class is high"
    raise ValueError(f"unknown graph operation family: {operation.family}")


def render_operation_query(
    operation: OperationDefinition,
    n_entities: int,
    false_label: str,
    true_label: str,
) -> str:
    clause = _base_operation_clause(operation, n_entities)
    if operation.polarity < 0:
        clause = f"it is false that {clause}"
    false_display = false_label.strip()
    true_display = true_label.strip()
    return (
        "\nRegistered operation: decide whether the following statement is true: "
        f"{clause}. Reply with exactly {false_display} for false or {true_display} for true.\n"
        "Answer:"
    )
