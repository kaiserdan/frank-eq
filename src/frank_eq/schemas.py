"""Serializable metadata schemas for Stage-0 artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OperationDefinition:
    """Public operation used to define a future causal signature."""

    operation_id: int
    family: str
    fact_args: tuple[int, int]
    residual_args: tuple[int, int]
    polarity: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fact_args"] = list(self.fact_args)
        payload["residual_args"] = list(self.residual_args)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OperationDefinition:
        return cls(
            operation_id=int(payload["operation_id"]),
            family=str(payload["family"]),
            fact_args=tuple(int(v) for v in payload["fact_args"]),
            residual_args=tuple(int(v) for v in payload["residual_args"]),
            polarity=float(payload["polarity"]),
        )


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """World-grouped train/validation/test and operation split manifest."""

    train_world_ids: tuple[int, ...]
    validation_world_ids: tuple[int, ...]
    test_world_ids: tuple[int, ...]
    train_operation_ids: tuple[int, ...]
    heldout_operation_ids: tuple[int, ...]
    founder_model_ids: tuple[int, ...]
    held_model_id: int | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: list(value) if isinstance(value, tuple) else value for key, value in payload.items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitManifest:
        return cls(
            train_world_ids=tuple(int(v) for v in payload["train_world_ids"]),
            validation_world_ids=tuple(int(v) for v in payload["validation_world_ids"]),
            test_world_ids=tuple(int(v) for v in payload["test_world_ids"]),
            train_operation_ids=tuple(int(v) for v in payload["train_operation_ids"]),
            heldout_operation_ids=tuple(int(v) for v in payload["heldout_operation_ids"]),
            founder_model_ids=tuple(int(v) for v in payload["founder_model_ids"]),
            held_model_id=(
                None if payload.get("held_model_id") is None else int(payload["held_model_id"])
            ),
        )
