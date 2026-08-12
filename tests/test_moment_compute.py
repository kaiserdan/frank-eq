from __future__ import annotations

import numpy as np

from frank_eq.data.real_panel import RelationalWorld, evaluate_operation
from frank_eq.moment_compute.events import (
    COMPILED_FAMILIES,
    build_event_registry,
    compile_operation_from_events,
    event_truth,
    project_event_probabilities,
)
from frank_eq.moment_compute.panel import _build_operations
from frank_eq.real_config import RealPanelConfig
from frank_eq.schemas import OperationDefinition


def _world() -> RelationalWorld:
    return RelationalWorld(
        world_id=0,
        edges=(
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (0, 1, 0, 1),
            (1, 0, 0, 0),
        ),
        density_score=0.1,
        reciprocity_score=-0.2,
    )


def _operations() -> list[OperationDefinition]:
    return [
        OperationDefinition(0, "lookup", (0, 1), (2, 3), 1.0),
        OperationDefinition(1, "inverse", (0, 1), (2, 3), 1.0),
        OperationDefinition(2, "mutual", (0, 1), (2, 3), 1.0),
        OperationDefinition(3, "compose", (0, 3), (1, 2), 1.0),
        OperationDefinition(4, "compare_outdegree", (0, 3), (1, 2), 1.0),
        OperationDefinition(5, "counterfactual_add", (2, 0), (2, 3), 1.0),
        OperationDefinition(6, "compose", (0, 3), (1, 2), -1.0),
    ]


def test_operation_closed_registry_exactly_executes_registered_families() -> None:
    registry = build_event_registry(4)
    world = _world()
    truth = {event.key: float(event_truth(event, world)) for event in registry.events}
    for operation in _operations():
        assert operation.family in COMPILED_FAMILIES
        predicted = compile_operation_from_events(truth, operation, n_entities=4) >= 0.5
        assert predicted == evaluate_operation(world, operation)


def test_projection_enforces_conjunction_bounds_and_degree_simplexes() -> None:
    registry = build_event_registry(4)
    raw = {event.key: 0.9 for event in registry.events}
    raw["edge:0>1"] = 0.2
    raw["edge:1>0"] = 0.3
    raw["mutual:0<>1"] = 0.95
    projected, diagnostics = project_event_probabilities(registry, raw)
    assert projected["mutual:0<>1"] <= 0.2 + 1e-12
    degree_groups: dict[str, list[float]] = {}
    for event in registry.events:
        if event.simplex_group is not None:
            degree_groups.setdefault(event.simplex_group, []).append(projected[event.key])
    assert degree_groups
    assert all(np.isclose(sum(values), 1.0) for values in degree_groups.values())
    assert diagnostics["max_absolute_adjustment"] > 0.0


def test_registry_is_deterministic_and_nontrivial() -> None:
    left = build_event_registry(4)
    right = build_event_registry(4)
    assert left.sha256 == right.sha256
    assert len(left.events) > 250
    assert any(event.kind == "joint_outdegree" for event in left.events)
    assert any(
        event.kind == "counterfactual_two_path_intersection" for event in left.events
    )


def test_counterfactual_generator_makes_the_added_edge_load_bearing() -> None:
    operations = _build_operations(
        RealPanelConfig(n_entities=4, n_operations=32, seed=20260831)
    )
    counterfactuals = [
        row.definition for row in operations if row.definition.family == "counterfactual_add"
    ]
    assert counterfactuals
    for operation in counterfactuals:
        source, added_target = operation.fact_args
        query_source, query_target = operation.residual_args
        assert query_source == source
        assert query_target not in {source, added_target}
