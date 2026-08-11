"""Fresh role-separated panels for the Stage-A v3 representation run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from frank_eq.data.real_panel import (
    ENTITY_NAMES,
    FrozenOperation,
    RealPanel,
    RelationalWorld,
    _candidate_worlds,
    _center_worlds,
    _sample_operations,
    evaluate_operation,
    render_world_prefix,
)
from frank_eq.real_config import RealPanelConfig
from frank_eq.utils import canonical_json_bytes, sha256_bytes

from .config import StageAV3Config

_ROLE_OFFSETS = {"train": 1_000_000, "validation": 2_000_000, "test": 3_000_000}
_RENDERER_NAMES = ("natural", "adjacency", "canonical_edge_list")


@dataclass(frozen=True, slots=True)
class V3Panel:
    role: str
    entity_count: int
    panel: RealPanel
    operation_registry_sha256: str
    renderer_names: tuple[str, ...] = _RENDERER_NAMES
    schema: str = "frank_eq_stagea_v3_panel_v1"

    def public_world_id(self, local_world_id: int) -> int:
        if not 0 <= local_world_id < len(self.panel.worlds):
            raise IndexError("local Stage-A v3 world ID is outside the panel")
        return _ROLE_OFFSETS[self.role] + self.entity_count * 10_000 + local_world_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "role": self.role,
            "entity_count": self.entity_count,
            "operation_registry_sha256": self.operation_registry_sha256,
            "renderer_names": list(self.renderer_names),
            "panel": self.panel.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> V3Panel:
        if payload.get("schema") != "frank_eq_stagea_v3_panel_v1":
            raise ValueError("unsupported Stage-A v3 panel schema")
        result = cls(
            role=str(payload["role"]),
            entity_count=int(payload["entity_count"]),
            panel=RealPanel.from_dict(payload["panel"]),
            operation_registry_sha256=str(payload["operation_registry_sha256"]),
            renderer_names=tuple(str(value) for value in payload["renderer_names"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.role not in _ROLE_OFFSETS:
            raise ValueError(f"unsupported Stage-A v3 panel role: {self.role}")
        if self.entity_count not in {4, 6} or self.panel.n_entities != self.entity_count:
            raise ValueError("Stage-A v3 panel entity count is inconsistent")
        if self.renderer_names != _RENDERER_NAMES:
            raise ValueError("Stage-A v3 renderer registry changed")
        if _operation_registry_hash(self.panel.operations) != self.operation_registry_sha256:
            raise ValueError("Stage-A v3 operation registry hash mismatch")


def _operation_registry_hash(operations: tuple[FrozenOperation, ...] | list[FrozenOperation]) -> str:
    return sha256_bytes(canonical_json_bytes([operation.to_dict() for operation in operations]))


def _panel_config(config: StageAV3Config, role: str, entity_count: int) -> RealPanelConfig:
    panel = config.section("panel")
    if role not in panel["roles"]:
        raise ValueError(f"unsupported Stage-A v3 role: {role}")
    return RealPanelConfig(
        n_worlds=int(panel["roles"][role]["worlds_per_complexity"]),
        n_entities=entity_count,
        n_operations=int(panel["n_target_operations"]),
        n_renderers=3,
        train_fraction=0.6,
        validation_fraction=0.2,
        operation_holdout_fraction=0.25,
        oracle_smoothing=float(panel["oracle_smoothing"]),
        min_operation_positive_fraction=float(panel["min_operation_positive_fraction"]),
        max_operation_positive_fraction=float(panel["max_operation_positive_fraction"]),
        max_generation_attempts=int(panel["max_generation_attempts"]),
        seed=int(panel["roles"][role]["seed"]),
    )


def _frozen_operations(config: StageAV3Config, entity_count: int) -> list[FrozenOperation]:
    panel = config.section("panel")
    operation_config = _panel_config(config, "train", entity_count)
    rng = np.random.default_rng(int(panel["operation_seed"]) + entity_count)
    return _sample_operations(operation_config, rng)


def generate_v3_panel(
    config: StageAV3Config,
    role: str,
    entity_count: int,
    *,
    test_access_grant: dict[str, Any] | None = None,
) -> V3Panel:
    """Generate role-specific worlds under one complexity-specific operation registry."""

    if role == "test" and (
        test_access_grant is None
        or test_access_grant.get("schema") != "frank_eq_stagea_v3_access_ledger_v1"
        or test_access_grant.get("config_sha256") != config.config_sha256
        or test_access_grant.get("current_stage") != "evaluate"
        or test_access_grant.get("test_access_count") != 1
    ):
        raise RuntimeError(
            "registered test-panel generation requires the consumed access-ledger grant"
        )
    if entity_count not in config.section("panel")["entity_counts"]:
        raise ValueError(f"entity count {entity_count} is not registered")
    role_config = _panel_config(config, role, entity_count)
    operations = _frozen_operations(config, entity_count)
    last_fractions: np.ndarray | None = None
    for attempt in range(role_config.max_generation_attempts):
        world_rng = np.random.default_rng(
            role_config.seed + entity_count * 100_003 + 1009 * (attempt + 1)
        )
        worlds = _center_worlds(_candidate_worlds(role_config, world_rng))
        labels = np.asarray(
            [
                [evaluate_operation(world, operation.definition) for operation in operations]
                for world in worlds
            ],
            dtype=np.float32,
        )
        fractions = labels.mean(axis=0)
        last_fractions = fractions
        if np.all(fractions >= role_config.min_operation_positive_fraction) and np.all(
            fractions <= role_config.max_operation_positive_fraction
        ):
            smoothing = role_config.oracle_smoothing
            signatures = labels * (1.0 - 2.0 * smoothing) + smoothing
            raw_config = asdict(role_config)
            raw_config.update(
                {
                    "role": role,
                    "world_seed": role_config.seed,
                    "operation_seed": config.section("panel")["operation_seed"],
                    "renderer_names": list(_RENDERER_NAMES),
                }
            )
            panel = RealPanel(
                worlds=tuple(worlds),
                operations=tuple(operations),
                oracle_signatures=tuple(
                    tuple(float(value) for value in row) for row in signatures.tolist()
                ),
                config=raw_config,
            )
            panel.validate()
            result = V3Panel(
                role=role,
                entity_count=entity_count,
                panel=panel,
                operation_registry_sha256=_operation_registry_hash(panel.operations),
            )
            result.validate()
            return result
    raise RuntimeError(
        f"failed to generate balanced {role}/n{entity_count} v3 panel; "
        f"final positive fractions={None if last_fractions is None else last_fractions.tolist()}"
    )


def render_v3_world_prefix(world: RelationalWorld, renderer: str | int) -> str:
    """Render one world through a frozen seen or unseen Stage-A v3 grammar."""

    renderer_id = _RENDERER_NAMES.index(renderer) if isinstance(renderer, str) else int(renderer)
    if renderer_id in {0, 1}:
        return render_world_prefix(world, renderer_id)
    if renderer_id != 2:
        raise ValueError(f"unsupported Stage-A v3 renderer: {renderer!r}")
    names = ENTITY_NAMES[: world.n_entities]
    edge = world.edge_array()
    coordinates = [
        f"{names[source]}->{names[target]}={int(edge[source, target])}"
        for source in range(world.n_entities)
        for target in range(world.n_entities)
        if source != target
    ]
    density_label = "high" if world.density_score > 0 else "low"
    reciprocity_label = "high" if world.reciprocity_score > 0 else "low"
    return (
        "Closed directed world; every non-diagonal coordinate is listed. "
        "No operation has been selected.\n"
        f"ENTITY_ORDER={','.join(names)}\n"
        + "\n".join(coordinates)
        + f"\nTAGS density={density_label} reciprocity={reciprocity_label}\n"
    )
