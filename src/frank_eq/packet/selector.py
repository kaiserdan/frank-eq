"""Query-conditioned selection from a query-blind operational state."""

from __future__ import annotations

import numpy as np


class QueryConditionedSelector:
    """Select the most relevant public probes by descriptor cosine similarity."""

    def __init__(self, operation_descriptors: np.ndarray):
        descriptors = np.asarray(operation_descriptors, dtype=np.float32)
        if descriptors.ndim != 2:
            raise ValueError("operation_descriptors must be two-dimensional")
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        self.normalized = descriptors / np.clip(norms, 1e-8, None)

    def select(self, query_operation_id: int, count: int) -> np.ndarray:
        if not 0 <= query_operation_id < self.normalized.shape[0]:
            raise IndexError("query_operation_id out of range")
        if count < 1:
            raise ValueError("count must be positive")
        similarities = self.normalized @ self.normalized[query_operation_id]
        ordering = np.argsort(-similarities, kind="stable")
        # The query coordinate is permitted at transmission time but support
        # probes are also retained. Keeping it first makes the packet useful
        # while preserving an auditable query-conditioned selection rule.
        return ordering[: min(count, len(ordering))].astype(np.int32)
