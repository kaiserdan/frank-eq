import numpy as np

from frank_eq.data.real_panel import RelationalWorld, evaluate_operation
from frank_eq.rate_compute.logic import (
    edge_vector_to_matrix,
    execute_public_basis,
    operation_support_size,
    ordered_edges,
)
from frank_eq.schemas import OperationDefinition


def _world() -> RelationalWorld:
    edge = (
        (0, 1, 1, 0),
        (0, 0, 1, 0),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
    )
    return RelationalWorld(
        world_id=0,
        edges=edge,
        density_score=0.25,
        reciprocity_score=-0.25,
    )


def _operations() -> list[OperationDefinition]:
    rows = [
        ("lookup", (0, 1), (2, 3)),
        ("inverse", (0, 2), (1, 3)),
        ("mutual", (0, 2), (1, 3)),
        ("compose", (0, 3), (1, 2)),
        ("compare_outdegree", (0, 1), (2, 3)),
        ("counterfactual_add", (1, 3), (1, 3)),
    ]
    operations: list[OperationDefinition] = []
    operation_id = 0
    for family, fact_args, residual_args in rows:
        for polarity in (1.0, -1.0):
            operations.append(
                OperationDefinition(
                    operation_id=operation_id,
                    family=family,
                    fact_args=fact_args,
                    residual_args=residual_args,
                    polarity=polarity,
                )
            )
            operation_id += 1
    return operations


def test_binary_operational_basis_exactly_compiles_registered_structural_targets() -> None:
    world = _world()
    vector = world.fact_vector()
    matrix = edge_vector_to_matrix(vector, world.n_entities)

    assert tuple(ordered_edges(world.n_entities)) == tuple(
        (source, target)
        for source in range(world.n_entities)
        for target in range(world.n_entities)
        if source != target
    )
    for operation in _operations():
        expected = evaluate_operation(world, operation)
        observed = execute_public_basis(matrix, operation) >= 0.5
        assert observed == expected, operation


def test_operation_support_bounds_are_strictly_smaller_than_full_basis_for_local_queries() -> None:
    n_entities = 6
    full = n_entities * (n_entities - 1)
    operations = _operations()
    sizes = {operation.family: operation_support_size(operation, n_entities) for operation in operations}

    assert sizes["lookup"] == 1
    assert sizes["inverse"] == 1
    assert sizes["mutual"] == 2
    assert sizes["compose"] == 2 * (n_entities - 2)
    assert sizes["compare_outdegree"] == 2 * (n_entities - 1)
    assert all(size < full for size in sizes.values())


def test_probabilistic_executor_remains_finite_and_bounded() -> None:
    rng = np.random.default_rng(7)
    vector = rng.uniform(0.05, 0.95, size=12)
    matrix = edge_vector_to_matrix(vector, 4)
    for operation in _operations():
        probability = execute_public_basis(matrix, operation)
        assert np.isfinite(probability)
        assert 0.0 < probability < 1.0
