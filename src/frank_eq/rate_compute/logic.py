"""Externally grounded operational basis and deterministic public executor."""

from __future__ import annotations

import numpy as np

from frank_eq.data.real_panel import ENTITY_NAMES, edge_fact_index
from frank_eq.schemas import OperationDefinition

COMPILED_FAMILIES = frozenset(
    {"lookup", "inverse", "mutual", "compose", "compare_outdegree", "counterfactual_add"}
)
HARD_COMPOSITION_FAMILIES = frozenset(
    {"mutual", "compose", "compare_outdegree", "counterfactual_add"}
)


def ordered_edges(n_entities: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (source, target)
        for source in range(n_entities)
        for target in range(n_entities)
        if source != target
    )


def render_basis_query(
    source: int,
    target: int,
    n_entities: int,
    *,
    final_cue: str,
) -> str:
    names = ENTITY_NAMES[:n_entities]
    return (
        "\nRegistered elementary operation: determine whether "
        f"{names[source]} directly points to {names[target]}."
        f"{final_cue}"
    )


def edge_vector_to_matrix(values: np.ndarray, n_entities: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    expected = n_entities * (n_entities - 1)
    if vector.size != expected:
        raise ValueError(f"expected {expected} edge values, received {vector.size}")
    matrix = np.zeros((n_entities, n_entities), dtype=np.float64)
    for source, target in ordered_edges(n_entities):
        matrix[source, target] = vector[edge_fact_index(n_entities, source, target)]
    return np.clip(matrix, 0.0, 1.0)


def _poisson_binomial(probabilities: np.ndarray) -> np.ndarray:
    distribution = np.asarray([1.0], dtype=np.float64)
    for probability in np.asarray(probabilities, dtype=np.float64).reshape(-1):
        probability = float(np.clip(probability, 0.0, 1.0))
        updated = np.zeros(distribution.size + 1, dtype=np.float64)
        updated[:-1] += distribution * (1.0 - probability)
        updated[1:] += distribution * probability
        distribution = updated
    return distribution


def _outdegree_greater_probability(edge: np.ndarray, source: int, target: int) -> float:
    source_probs = np.delete(edge[source], source)
    target_probs = np.delete(edge[target], target)
    source_distribution = _poisson_binomial(source_probs)
    target_distribution = _poisson_binomial(target_probs)
    result = 0.0
    for source_degree, source_mass in enumerate(source_distribution):
        result += float(source_mass) * float(target_distribution[:source_degree].sum())
    return float(np.clip(result, 0.0, 1.0))


def _two_hop_probability(edge: np.ndarray, source: int, target: int) -> float:
    no_path = 1.0
    for middle in range(edge.shape[0]):
        if middle in {source, target}:
            continue
        path_probability = float(edge[source, middle] * edge[middle, target])
        no_path *= 1.0 - path_probability
    return float(np.clip(1.0 - no_path, 0.0, 1.0))


def execute_public_basis(
    edge_probabilities: np.ndarray,
    operation: OperationDefinition,
) -> float:
    """Compute a target operation from public edge marginals.

    The executor assumes conditional independence of uncertain edge coordinates.
    It is exact for deterministic edge vectors and therefore provides a transparent,
    falsifiable public ABI.
    """

    edge = np.asarray(edge_probabilities, dtype=np.float64)
    if edge.ndim != 2 or edge.shape[0] != edge.shape[1]:
        raise ValueError("edge_probabilities must be a square matrix")
    source, target = operation.fact_args
    aux_source, aux_target = operation.residual_args
    family = operation.family
    if family == "lookup":
        probability = float(edge[source, target])
    elif family == "inverse":
        probability = float(edge[target, source])
    elif family == "mutual":
        probability = float(edge[source, target] * edge[target, source])
    elif family == "compose":
        probability = _two_hop_probability(edge, source, target)
    elif family == "compare_outdegree":
        probability = _outdegree_greater_probability(edge, source, target)
    elif family == "counterfactual_add":
        modified = edge.copy()
        modified[source, target] = 1.0
        probability = _two_hop_probability(modified, aux_source, aux_target)
    else:
        raise ValueError(f"operation family {family!r} is not public-basis compilable")
    if operation.polarity < 0:
        probability = 1.0 - probability
    return float(np.clip(probability, 1e-7, 1.0 - 1e-7))


def operation_support_size(operation: OperationDefinition, n_entities: int) -> int:
    """Upper bound on elementary edge coordinates consumed by an operation."""

    family = operation.family
    if family in {"lookup", "inverse"}:
        return 1
    if family == "mutual":
        return 2
    if family == "compose":
        return 2 * max(1, n_entities - 2)
    if family == "compare_outdegree":
        return 2 * (n_entities - 1)
    if family == "counterfactual_add":
        return 1 + 2 * max(1, n_entities - 2)
    if family in {"density", "reciprocity"}:
        return n_entities * (n_entities - 1)
    raise ValueError(f"unknown operation family: {family}")
