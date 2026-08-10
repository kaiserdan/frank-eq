"""Information-boundary contracts for real future-signature extraction.

These schemas are deliberately backend-agnostic. A Hugging Face, vLLM, or
custom model adapter may populate them, but the causal boundary is validated
before any record can enter a claim-bearing cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from frank_eq.utils import canonical_json_bytes, sha256_bytes


@dataclass(frozen=True, slots=True)
class StateCaptureRecord:
    """One hidden-state capture made before the future operation is revealed."""

    state_id: str
    world_id: str
    model_id: str
    renderer_id: str
    split: str
    prefix_sha256: str
    hidden_artifact_sha256: str
    captured_before_operation: bool
    capture_step: int

    def validate(self) -> None:
        if not self.state_id or not self.world_id or not self.model_id:
            raise ValueError("state, world, and model identifiers are required")
        if self.split not in {"train", "validation", "test", "confirmation", "locked"}:
            raise ValueError(f"unsupported split: {self.split}")
        for name, value in (
            ("prefix_sha256", self.prefix_sha256),
            ("hidden_artifact_sha256", self.hidden_artifact_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.captured_before_operation:
            raise ValueError("state was not captured before operation reveal")
        if self.capture_step < 0:
            raise ValueError("capture_step must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FutureBranchRecord:
    """Outcome distribution for one operation branched from a cached state."""

    state_id: str
    operation_id: str
    operation_descriptor_sha256: str
    outcome_probabilities: tuple[float, ...]
    branch_seed: int
    operation_reveal_step: int

    def validate(self) -> None:
        if not self.state_id or not self.operation_id:
            raise ValueError("state_id and operation_id are required")
        digest = self.operation_descriptor_sha256
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("operation_descriptor_sha256 must be a lowercase SHA-256 digest")
        probabilities = np.asarray(self.outcome_probabilities, dtype=np.float64)
        if probabilities.ndim != 1 or len(probabilities) < 2:
            raise ValueError("outcome_probabilities must contain at least two outcomes")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("outcome probabilities must be finite and non-negative")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
            raise ValueError("outcome probabilities must sum to one")
        if self.branch_seed < 0 or self.operation_reveal_step < 0:
            raise ValueError("branch_seed and operation_reveal_step must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome_probabilities"] = list(self.outcome_probabilities)
        return payload


@dataclass(frozen=True, slots=True)
class FutureSignatureRecord:
    """A complete operation-agnostic state plus its branched future signature."""

    capture: StateCaptureRecord
    branches: tuple[FutureBranchRecord, ...]
    schema: str = "frank_eq_future_signature_record_v1"

    def validate(self, required_operation_ids: set[str] | None = None) -> None:
        self.capture.validate()
        if not self.branches:
            raise ValueError("future signature has no branches")
        operation_ids: list[str] = []
        for branch in self.branches:
            branch.validate()
            if branch.state_id != self.capture.state_id:
                raise ValueError("future branch does not originate from the captured state")
            if branch.operation_reveal_step <= self.capture.capture_step:
                raise ValueError("future operation was revealed before or at state capture")
            operation_ids.append(branch.operation_id)
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("duplicate operation branches for one captured state")
        if required_operation_ids is not None and set(operation_ids) != required_operation_ids:
            raise ValueError("future signature does not cover the frozen operation registry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "capture": self.capture.to_dict(),
            "branches": [branch.to_dict() for branch in self.branches],
        }

    def content_sha256(self) -> str:
        self.validate()
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


def validate_world_split_integrity(records: list[FutureSignatureRecord]) -> None:
    """Reject any world appearing in multiple data roles."""

    split_by_world: dict[str, str] = {}
    state_ids: set[str] = set()
    for record in records:
        record.validate()
        if record.capture.state_id in state_ids:
            raise ValueError(f"duplicate state_id: {record.capture.state_id}")
        state_ids.add(record.capture.state_id)
        previous = split_by_world.setdefault(record.capture.world_id, record.capture.split)
        if previous != record.capture.split:
            raise ValueError(
                f"world {record.capture.world_id!r} crosses splits: {previous!r} and {record.capture.split!r}"
            )
