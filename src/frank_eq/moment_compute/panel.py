"""Non-degenerate four-entity panel construction for Stage M0."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from frank_eq.data.real_panel import (
    FrozenOperation,
    RealPanel,
    _candidate_worlds,
    _center_worlds,
    build_operation_descriptor,
    evaluate_operation,
)
from frank_eq.real_config import GRAPH_OPERATION_FAMILIES, RealPanelConfig
from frank_eq.schemas import OperationDefinition
from frank_eq.utils import canonical_json_bytes, sha256_bytes

from .config import MomentComputeRunConfig


def _build_operations(config: RealPanelConfig) -> list[FrozenOperation]:
    """Build the frozen grammar, repairing the historical counterfactual no-op."""

    rng = np.random.default_rng(config.seed + 13)
    per_family = config.n_operations // len(GRAPH_OPERATION_FAMILIES)
    rows: list[FrozenOperation] = []
    operation_id = 0
    for family in GRAPH_OPERATION_FAMILIES:
        for local_index in range(per_family):
            if family == "counterfactual_add":
                source, added_target, query_target = (
                    int(value)
                    for value in rng.choice(config.n_entities, size=3, replace=False)
                )
                fact_args = (source, added_target)
                residual_args = (source, query_target)
            else:
                values = [
                    int(value)
                    for value in rng.choice(config.n_entities, size=4, replace=False)
                ]
                fact_args = (values[0], values[1])
                residual_args = (values[2], values[3])
            definition = OperationDefinition(
                operation_id=operation_id,
                family=family,
                fact_args=fact_args,
                residual_args=residual_args,
                polarity=1.0 if local_index % 2 == 0 else -1.0,
            )
            descriptor = build_operation_descriptor(definition, config.n_entities)
            digest = sha256_bytes(
                canonical_json_bytes(
                    {
                        "definition": definition.to_dict(),
                        "descriptor": descriptor.tolist(),
                    }
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


def build_moment_panel(config: MomentComputeRunConfig) -> RealPanel:
    """Generate a balanced panel under the prospective Stage-M operation grammar."""

    panel = config.panel
    panel_config = RealPanelConfig(
        n_worlds=panel.worlds_per_complexity,
        n_entities=panel.entity_counts[0],
        n_operations=panel.n_target_operations,
        n_renderers=panel.n_renderers,
        train_fraction=0.60,
        validation_fraction=0.20,
        operation_holdout_fraction=0.25,
        oracle_smoothing=panel.oracle_smoothing,
        min_operation_positive_fraction=panel.min_operation_positive_fraction,
        max_operation_positive_fraction=panel.max_operation_positive_fraction,
        max_generation_attempts=panel.max_generation_attempts,
        seed=panel.seed,
    )
    operations = _build_operations(panel_config)
    last_fractions: np.ndarray | None = None
    for attempt in range(panel_config.max_generation_attempts):
        rng = np.random.default_rng(panel_config.seed + 1009 * (attempt + 1))
        worlds = _center_worlds(_candidate_worlds(panel_config, rng))
        labels = np.asarray(
            [
                [evaluate_operation(world, operation.definition) for operation in operations]
                for world in worlds
            ],
            dtype=np.float32,
        )
        fractions = labels.mean(axis=0)
        last_fractions = fractions
        if np.all(fractions >= panel_config.min_operation_positive_fraction) and np.all(
            fractions <= panel_config.max_operation_positive_fraction
        ):
            smooth = panel_config.oracle_smoothing
            signatures = labels * (1.0 - 2.0 * smooth) + smooth
            raw_config = asdict(panel_config)
            raw_config.update(
                {
                    "protocol_version": config.protocol_version,
                    "counterfactual_contract": (
                        "add u->v then query two-hop reachability from u to w, "
                        "with u, v, w distinct"
                    ),
                    "historical_four_distinct_counterfactual_forbidden": True,
                }
            )
            result = RealPanel(
                worlds=tuple(worlds),
                operations=tuple(operations),
                oracle_signatures=tuple(
                    tuple(float(value) for value in row) for row in signatures.tolist()
                ),
                config=raw_config,
            )
            result.validate()
            return result
    raise RuntimeError(
        "failed to generate a balanced Stage-M panel; final positive fractions="
        f"{None if last_fractions is None else last_fractions.tolist()}"
    )
