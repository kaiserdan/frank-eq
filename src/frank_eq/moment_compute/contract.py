# ruff: noqa: I001
"""Prospective Stage-M registry overlay for non-degenerate counterfactual events."""

from __future__ import annotations

from itertools import permutations

from frank_eq.utils import canonical_json_bytes, sha256_bytes

from .events import (
    EventRegistry,
    PublicEvent,
    build_event_registry as _build_base_registry,
)


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


def _required_path_edges(
    source: int,
    middle: int,
    target: int,
) -> tuple[tuple[int, int], ...]:
    return ((source, middle), (middle, target))


def _remove_forced(
    edges: tuple[tuple[int, int], ...],
    forced: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({edge for edge in edges if edge != forced}))


def build_stage_m_event_registry(n_entities: int = 4) -> EventRegistry:
    """Replace the historical no-op counterfactual coordinates prospectively.

    The historical operation sampler selected four distinct arguments. After
    adding `u->v`, it queried reachability between two other entities, so the
    intervention could not appear in either two-hop path. Stage M instead adds
    `u->v` and queries reachability from `u` to a distinct `w`.
    """

    base = _build_base_registry(n_entities)
    retained = [
        event
        for event in base.events
        if not event.kind.startswith("counterfactual_")
    ]
    pending: dict[str, dict[str, object]] = {
        event.key: {
            "key": event.key,
            "kind": event.kind,
            "order": event.order,
            "required_edges": event.required_edges,
            "intervention_edge": event.intervention_edge,
            "degree_pair": event.degree_pair,
            "simplex_group": event.simplex_group,
        }
        for event in retained
    }

    def add(
        key: str,
        *,
        kind: str,
        order: int,
        required_edges: tuple[tuple[int, int], ...],
        intervention_edge: tuple[int, int],
    ) -> None:
        row: dict[str, object] = {
            "key": key,
            "kind": kind,
            "order": order,
            "required_edges": tuple(sorted(set(required_edges))),
            "intervention_edge": intervention_edge,
            "degree_pair": None,
            "simplex_group": None,
        }
        existing = pending.get(key)
        if existing is not None and existing != row:
            raise RuntimeError(f"counterfactual event-key collision: {key}")
        pending[key] = row

    for source, forced_target, query_target in permutations(range(n_entities), 3):
        forced = (source, forced_target)
        middles = [
            value for value in range(n_entities) if value not in {source, query_target}
        ]
        for middle in middles:
            required = _remove_forced(
                _required_path_edges(source, middle, query_target),
                forced,
            )
            add(
                _counterfactual_path_key(
                    source,
                    forced_target,
                    source,
                    middle,
                    query_target,
                ),
                kind="counterfactual_two_hop_path",
                order=len(required),
                required_edges=required,
                intervention_edge=forced,
            )
        pair_edges = tuple(
            edge
            for middle in middles
            for edge in _required_path_edges(source, middle, query_target)
        )
        required_pair = _remove_forced(pair_edges, forced)
        add(
            _counterfactual_pair_key(
                source,
                forced_target,
                source,
                query_target,
            ),
            kind="counterfactual_two_path_intersection",
            order=len(required_pair),
            required_edges=required_pair,
            intervention_edge=forced,
        )

    events = tuple(
        PublicEvent(event_id=event_id, **pending[key])
        for event_id, key in enumerate(sorted(pending))
    )
    payload = {
        "n_entities": n_entities,
        "events": [event.to_dict() for event in events],
    }
    registry = EventRegistry(
        n_entities=n_entities,
        events=events,
        sha256=sha256_bytes(canonical_json_bytes(payload)),
    )
    registry.validate()
    return registry
