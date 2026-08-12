"""Operation-closed public event basis and exact affine executor for Stage M0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Any

import numpy as np

from frank_eq.data.real_panel import ENTITY_NAMES, RelationalWorld
from frank_eq.rate_compute.logic import edge_vector_to_matrix, execute_public_basis, ordered_edges
from frank_eq.schemas import OperationDefinition
from frank_eq.utils import canonical_json_bytes, sha256_bytes

SUPPORTED_ENTITY_COUNT = 4
COMPILED_FAMILIES = frozenset(
    {"lookup", "inverse", "mutual", "compose", "compare_outdegree", "counterfactual_add"}
)
HARD_FAMILIES = frozenset(
    {"mutual", "compose", "compare_outdegree", "counterfactual_add"}
)


@dataclass(frozen=True, slots=True)
class PublicEvent:
    """One externally identifiable event coordinate in the public predictive state."""

    event_id: int
    key: str
    kind: str
    order: int
    required_edges: tuple[tuple[int, int], ...] = ()
    intervention_edge: tuple[int, int] | None = None
    degree_pair: tuple[int, int, int, int] | None = None
    simplex_group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_edges"] = [list(edge) for edge in self.required_edges]
        if self.intervention_edge is not None:
            payload["intervention_edge"] = list(self.intervention_edge)
        if self.degree_pair is not None:
            payload["degree_pair"] = list(self.degree_pair)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublicEvent:
        return cls(
            event_id=int(payload["event_id"]),
            key=str(payload["key"]),
            kind=str(payload["kind"]),
            order=int(payload["order"]),
            required_edges=tuple(
                (int(edge[0]), int(edge[1])) for edge in payload.get("required_edges", [])
            ),
            intervention_edge=(
                None
                if payload.get("intervention_edge") is None
                else (
                    int(payload["intervention_edge"][0]),
                    int(payload["intervention_edge"][1]),
                )
            ),
            degree_pair=(
                None
                if payload.get("degree_pair") is None
                else tuple(int(value) for value in payload["degree_pair"])
            ),
            simplex_group=(
                None if payload.get("simplex_group") is None else str(payload["simplex_group"])
            ),
        )


@dataclass(frozen=True, slots=True)
class EventRegistry:
    n_entities: int
    events: tuple[PublicEvent, ...]
    sha256: str
    schema: str = "frank_eq_operation_closed_event_registry_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "n_entities": self.n_entities,
            "events": [event.to_dict() for event in self.events],
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EventRegistry:
        if payload.get("schema") != "frank_eq_operation_closed_event_registry_v1":
            raise ValueError("unsupported event-registry schema")
        events = tuple(PublicEvent.from_dict(item) for item in payload["events"])
        registry = cls(
            n_entities=int(payload["n_entities"]),
            events=events,
            sha256=str(payload["sha256"]),
        )
        registry.validate()
        return registry

    def validate(self) -> None:
        if self.n_entities != SUPPORTED_ENTITY_COUNT:
            raise ValueError("Stage M0 registry is frozen to four entities")
        if [event.event_id for event in self.events] != list(range(len(self.events))):
            raise ValueError("event IDs must be contiguous and ordered")
        keys = [event.key for event in self.events]
        if len(keys) != len(set(keys)):
            raise ValueError("event registry contains duplicate keys")
        payload = {
            "n_entities": self.n_entities,
            "events": [event.to_dict() for event in self.events],
        }
        if sha256_bytes(canonical_json_bytes(payload)) != self.sha256:
            raise ValueError("event registry SHA-256 mismatch")

    @property
    def by_key(self) -> dict[str, PublicEvent]:
        return {event.key: event for event in self.events}


def _edge_key(source: int, target: int) -> str:
    return f"edge:{source}>{target}"


def _mutual_key(left: int, right: int) -> str:
    first, second = sorted((left, right))
    return f"mutual:{first}<>{second}"


def _path_key(source: int, middle: int, target: int) -> str:
    return f"path:{source}>{middle}>{target}"


def _path_pair_key(source: int, target: int) -> str:
    return f"path_union_intersection:{source}>{target}"


def _counterfactual_path_key(
    forced_source: int,
    forced_target: int,
    source: int,
    middle: int,
    target: int,
) -> str:
    return (
        f"cf_path:add({forced_source}>{forced_target}):"
        f"{source}>{middle}>{target}"
    )


def _counterfactual_pair_key(
    forced_source: int,
    forced_target: int,
    source: int,
    target: int,
) -> str:
    return (
        f"cf_path_union_intersection:add({forced_source}>{forced_target}):"
        f"{source}>{target}"
    )


def _degree_pair_key(source: int, target: int, source_degree: int, target_degree: int) -> str:
    return f"degree_pair:{source}>{target}:{source_degree}>{target_degree}"


def _required_path_edges(source: int, middle: int, target: int) -> tuple[tuple[int, int], ...]:
    return ((source, middle), (middle, target))


def _remove_forced(
    edges: tuple[tuple[int, int], ...], forced: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({edge for edge in edges if edge != forced}))


def build_event_registry(n_entities: int = SUPPORTED_ENTITY_COUNT) -> EventRegistry:
    """Build the complete public basis required by the four-entity operation grammar."""

    if n_entities != SUPPORTED_ENTITY_COUNT:
        raise ValueError("Stage M0 builds only the four-entity operation closure")
    pending: dict[str, dict[str, Any]] = {}

    def add(
        key: str,
        *,
        kind: str,
        order: int,
        required_edges: tuple[tuple[int, int], ...] = (),
        intervention_edge: tuple[int, int] | None = None,
        degree_pair: tuple[int, int, int, int] | None = None,
        simplex_group: str | None = None,
    ) -> None:
        normalized_edges = tuple(sorted(set(required_edges)))
        row = {
            "key": key,
            "kind": kind,
            "order": order,
            "required_edges": normalized_edges,
            "intervention_edge": intervention_edge,
            "degree_pair": degree_pair,
            "simplex_group": simplex_group,
        }
        existing = pending.get(key)
        if existing is not None and existing != row:
            raise RuntimeError(f"event key collision with inconsistent semantics: {key}")
        pending[key] = row

    for source, target in ordered_edges(n_entities):
        add(
            _edge_key(source, target),
            kind="edge",
            order=1,
            required_edges=((source, target),),
        )

    for left in range(n_entities):
        for right in range(left + 1, n_entities):
            add(
                _mutual_key(left, right),
                kind="mutual_conjunction",
                order=2,
                required_edges=((left, right), (right, left)),
            )

    for source, target in ordered_edges(n_entities):
        middles = [value for value in range(n_entities) if value not in {source, target}]
        for middle in middles:
            add(
                _path_key(source, middle, target),
                kind="two_hop_path",
                order=2,
                required_edges=_required_path_edges(source, middle, target),
            )
        pair_edges = tuple(
            edge
            for middle in middles
            for edge in _required_path_edges(source, middle, target)
        )
        add(
            _path_pair_key(source, target),
            kind="two_path_intersection",
            order=len(set(pair_edges)),
            required_edges=pair_edges,
        )

    # Counterfactual operations in the frozen graph grammar use four distinct
    # arguments. Enumerating all permutations avoids target-instance leakage.
    for forced_source, forced_target, source, target in permutations(range(n_entities), 4):
        forced = (forced_source, forced_target)
        middles = [value for value in range(n_entities) if value not in {source, target}]
        for middle in middles:
            required = _remove_forced(
                _required_path_edges(source, middle, target),
                forced,
            )
            add(
                _counterfactual_path_key(
                    forced_source,
                    forced_target,
                    source,
                    middle,
                    target,
                ),
                kind="counterfactual_two_hop_path",
                order=len(required),
                required_edges=required,
                intervention_edge=forced,
            )
        pair_edges = tuple(
            edge
            for middle in middles
            for edge in _required_path_edges(source, middle, target)
        )
        required_pair = _remove_forced(pair_edges, forced)
        add(
            _counterfactual_pair_key(
                forced_source,
                forced_target,
                source,
                target,
            ),
            kind="counterfactual_two_path_intersection",
            order=len(required_pair),
            required_edges=required_pair,
            intervention_edge=forced,
        )

    for source, target in ordered_edges(n_entities):
        group = f"degree_pair:{source}>{target}"
        for source_degree in range(n_entities):
            for target_degree in range(n_entities):
                add(
                    _degree_pair_key(source, target, source_degree, target_degree),
                    kind="joint_outdegree",
                    order=2 * (n_entities - 1),
                    degree_pair=(source, target, source_degree, target_degree),
                    simplex_group=group,
                )

    rows = []
    for event_id, key in enumerate(sorted(pending)):
        row = pending[key]
        rows.append(PublicEvent(event_id=event_id, **row))
    payload = {"n_entities": n_entities, "events": [event.to_dict() for event in rows]}
    registry = EventRegistry(
        n_entities=n_entities,
        events=tuple(rows),
        sha256=sha256_bytes(canonical_json_bytes(payload)),
    )
    registry.validate()
    return registry


def event_truth(event: PublicEvent, world: RelationalWorld) -> bool:
    edge = world.edge_array().copy()
    if event.intervention_edge is not None:
        edge[event.intervention_edge] = 1
    if event.degree_pair is not None:
        source, target, source_degree, target_degree = event.degree_pair
        return int(edge[source].sum()) == source_degree and int(edge[target].sum()) == target_degree
    return all(bool(edge[source, target]) for source, target in event.required_edges)


def event_truth_vector(registry: EventRegistry, world: RelationalWorld) -> np.ndarray:
    return np.asarray([float(event_truth(event, world)) for event in registry.events])


def render_event_query(event: PublicEvent, n_entities: int, *, final_cue: str) -> str:
    if n_entities != SUPPORTED_ENTITY_COUNT:
        raise ValueError("Stage M0 query rendering is frozen to four entities")
    names = ENTITY_NAMES[:n_entities]
    if event.degree_pair is not None:
        source, target, source_degree, target_degree = event.degree_pair
        statement = (
            f"the out-degree of {names[source]} is exactly {source_degree} and "
            f"the out-degree of {names[target]} is exactly {target_degree}"
        )
    else:
        clauses = [
            f"{names[source]} directly points to {names[target]}"
            for source, target in event.required_edges
        ]
        if not clauses:
            statement = "the registered event is true"
        elif len(clauses) == 1:
            statement = clauses[0]
        else:
            statement = "all of the following hold: " + "; ".join(clauses)
    prefix = "\nRegistered public event: determine whether "
    if event.intervention_edge is not None:
        source, target = event.intervention_edge
        prefix += (
            f"after adding the edge {names[source]} to {names[target]}, "
            f"{statement}"
        )
    else:
        prefix += statement
    return prefix + f".{final_cue}"


def _event_probability(probabilities: dict[str, float], key: str) -> float:
    try:
        return float(probabilities[key])
    except KeyError as error:
        raise KeyError(f"operation-closed basis is missing public event {key}") from error


def compile_operation_from_events(
    probabilities: dict[str, float],
    operation: OperationDefinition,
    *,
    n_entities: int = SUPPORTED_ENTITY_COUNT,
    epsilon: float = 1e-7,
) -> float:
    """Execute one operation from an operation-closed event basis.

    The formula is exact for exact event probabilities. It does not introduce
    independence assumptions between uncertain edges.
    """

    if n_entities != SUPPORTED_ENTITY_COUNT:
        raise ValueError("Stage M0 executor is frozen to four entities")
    source, target = operation.fact_args
    aux_source, aux_target = operation.residual_args
    family = operation.family
    if family == "lookup":
        result = _event_probability(probabilities, _edge_key(source, target))
    elif family == "inverse":
        result = _event_probability(probabilities, _edge_key(target, source))
    elif family == "mutual":
        result = _event_probability(probabilities, _mutual_key(source, target))
    elif family == "compose":
        middles = [value for value in range(n_entities) if value not in {source, target}]
        path_sum = sum(
            _event_probability(probabilities, _path_key(source, middle, target))
            for middle in middles
        )
        result = path_sum - _event_probability(
            probabilities, _path_pair_key(source, target)
        )
    elif family == "compare_outdegree":
        result = 0.0
        for source_degree in range(n_entities):
            for target_degree in range(n_entities):
                if source_degree > target_degree:
                    result += _event_probability(
                        probabilities,
                        _degree_pair_key(
                            source,
                            target,
                            source_degree,
                            target_degree,
                        ),
                    )
    elif family == "counterfactual_add":
        middles = [
            value for value in range(n_entities) if value not in {aux_source, aux_target}
        ]
        result = sum(
            _event_probability(
                probabilities,
                _counterfactual_path_key(
                    source,
                    target,
                    aux_source,
                    middle,
                    aux_target,
                ),
            )
            for middle in middles
        ) - _event_probability(
            probabilities,
            _counterfactual_pair_key(
                source,
                target,
                aux_source,
                aux_target,
            ),
        )
    else:
        raise ValueError(f"operation family {family!r} is not event-basis compilable")
    result = float(np.clip(result, 0.0, 1.0))
    if operation.polarity < 0:
        result = 1.0 - result
    return float(np.clip(result, epsilon, 1.0 - epsilon))


def compile_operation_from_marginals(
    probabilities: dict[str, float],
    operation: OperationDefinition,
    *,
    n_entities: int = SUPPORTED_ENTITY_COUNT,
) -> float:
    vector = np.asarray(
        [probabilities[_edge_key(source, target)] for source, target in ordered_edges(n_entities)]
    )
    return execute_public_basis(edge_vector_to_matrix(vector, n_entities), operation)


def _parent_event_keys(event: PublicEvent, n_entities: int) -> tuple[str, ...]:
    if event.kind == "two_path_intersection":
        _, pair = event.key.split(":", 1)
        source, target = (int(value) for value in pair.split(">"))
        return tuple(
            _path_key(source, middle, target)
            for middle in range(n_entities)
            if middle not in {source, target}
        )
    if event.kind == "counterfactual_two_path_intersection":
        forced_text, pair = event.key.split(":", 2)[1:]
        forced = forced_text.removeprefix("add(").removesuffix(")")
        forced_source, forced_target = (int(value) for value in forced.split(">"))
        source, target = (int(value) for value in pair.split(">"))
        return tuple(
            _counterfactual_path_key(
                forced_source,
                forced_target,
                source,
                middle,
                target,
            )
            for middle in range(n_entities)
            if middle not in {source, target}
        )
    return ()


def project_event_probabilities(
    registry: EventRegistry,
    raw_probabilities: dict[str, float],
    *,
    epsilon: float = 1e-7,
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply label-free necessary coherence constraints to public event responses."""

    if set(raw_probabilities) != {event.key for event in registry.events}:
        raise ValueError("raw event probability map differs from the frozen registry")
    projected = {
        key: float(np.clip(value, epsilon, 1.0 - epsilon))
        for key, value in raw_probabilities.items()
    }
    adjustments: list[float] = []

    # Conjunction coordinates must lie inside their Frechet bounds. Intervened
    # edges are constants and therefore omitted from required_edges at registry construction.
    for event in sorted(registry.events, key=lambda item: (item.order, item.event_id)):
        if event.degree_pair is not None or event.kind == "edge":
            continue
        atom_probabilities = [
            projected[_edge_key(source, target)] for source, target in event.required_edges
        ]
        if atom_probabilities:
            lower = max(0.0, sum(atom_probabilities) - (len(atom_probabilities) - 1.0))
            upper = min(atom_probabilities)
        else:
            lower = upper = 1.0
        parents = _parent_event_keys(event, registry.n_entities)
        if parents:
            upper = min(upper, *(projected[parent] for parent in parents))
        previous = projected[event.key]
        projected[event.key] = float(np.clip(previous, lower, upper))
        adjustments.append(abs(projected[event.key] - previous))

    # Joint degree events form a categorical distribution for each ordered pair.
    groups: dict[str, list[PublicEvent]] = {}
    for event in registry.events:
        if event.simplex_group is not None:
            groups.setdefault(event.simplex_group, []).append(event)
    for group_events in groups.values():
        values = np.asarray([projected[event.key] for event in group_events], dtype=np.float64)
        total = float(values.sum())
        if not np.isfinite(total) or total <= 0.0:
            values = np.full_like(values, 1.0 / len(values))
        else:
            values /= total
        values = np.clip(values, epsilon, None)
        values /= values.sum()
        for event, value in zip(group_events, values, strict=True):
            previous = projected[event.key]
            projected[event.key] = float(value)
            adjustments.append(abs(projected[event.key] - previous))

    return projected, {
        "mean_absolute_adjustment": float(np.mean(adjustments)) if adjustments else 0.0,
        "max_absolute_adjustment": float(np.max(adjustments)) if adjustments else 0.0,
        "adjusted_coordinates": int(sum(value > 0.0 for value in adjustments)),
    }


def exact_executor_mismatches(
    registry: EventRegistry,
    worlds: tuple[RelationalWorld, ...] | list[RelationalWorld],
    operations: list[OperationDefinition] | tuple[OperationDefinition, ...],
) -> int:
    mismatches = 0
    from frank_eq.data.real_panel import evaluate_operation

    for world in worlds:
        truth_map = {
            event.key: float(event_truth(event, world)) for event in registry.events
        }
        for operation in operations:
            if operation.family not in COMPILED_FAMILIES:
                continue
            predicted = compile_operation_from_events(
                truth_map,
                operation,
                n_entities=registry.n_entities,
            ) >= 0.5
            if predicted != evaluate_operation(world, operation):
                mismatches += 1
    return mismatches
