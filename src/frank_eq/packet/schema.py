"""Canonical operational packet schema and checksum validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from frank_eq.utils import canonical_json_bytes, sha256_bytes

from .quantization import (
    dequantize_logits,
    logits_to_probabilities,
    probabilities_to_logits,
    quantize_logits,
)


@dataclass(frozen=True, slots=True)
class OperationalPacketV1:
    """A query-conditioned view of one query-blind operational state."""

    task_family: str
    query_operation_id: int
    fact_values: tuple[int, ...]
    probe_ids: tuple[int, ...]
    probe_values: tuple[int, ...]
    quantization_bits: int
    uncertainty_value: int
    checksum: str
    schema: str = "FRANK-EQ/OPERATIONAL-PACKET/1"

    @classmethod
    def build(
        cls,
        *,
        task_family: str,
        query_operation_id: int,
        fact_probabilities: np.ndarray,
        signature_probabilities: np.ndarray,
        probe_ids: np.ndarray,
        quantization_bits: int,
        uncertainty: float,
    ) -> OperationalPacketV1:
        fact_logits = probabilities_to_logits(fact_probabilities)
        signature_logits = probabilities_to_logits(signature_probabilities)
        fact_values = tuple(int(v) for v in quantize_logits(fact_logits, quantization_bits))
        selected_ids = tuple(int(v) for v in np.asarray(probe_ids, dtype=np.int32))
        selected_values = tuple(
            int(v)
            for v in quantize_logits(signature_logits[np.asarray(selected_ids)], quantization_bits)
        )
        uncertainty_probability = float(np.clip(uncertainty, 0.0, 1.0))
        uncertainty_value = int(
            quantize_logits(
                probabilities_to_logits(np.asarray([uncertainty_probability], dtype=np.float32)),
                quantization_bits,
            )[0]
        )
        body = {
            "schema": "FRANK-EQ/OPERATIONAL-PACKET/1",
            "task_family": task_family,
            "query_operation_id": int(query_operation_id),
            "fact_values": list(fact_values),
            "probe_ids": list(selected_ids),
            "probe_values": list(selected_values),
            "quantization_bits": int(quantization_bits),
            "uncertainty_value": uncertainty_value,
        }
        checksum = sha256_bytes(canonical_json_bytes(body))
        return cls(
            task_family=task_family,
            query_operation_id=int(query_operation_id),
            fact_values=fact_values,
            probe_ids=selected_ids,
            probe_values=selected_values,
            quantization_bits=int(quantization_bits),
            uncertainty_value=uncertainty_value,
            checksum=checksum,
        )

    @classmethod
    def empty(
        cls,
        *,
        task_family: str,
        query_operation_id: int,
        n_facts: int,
        quantization_bits: int,
    ) -> OperationalPacketV1:
        return cls.build(
            task_family=task_family,
            query_operation_id=query_operation_id,
            fact_probabilities=np.full(n_facts, 0.5, dtype=np.float32),
            signature_probabilities=np.full(query_operation_id + 1, 0.5, dtype=np.float32),
            probe_ids=np.asarray([query_operation_id], dtype=np.int32),
            quantization_bits=quantization_bits,
            uncertainty=0.5,
        )

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_family": self.task_family,
            "query_operation_id": self.query_operation_id,
            "fact_values": list(self.fact_values),
            "probe_ids": list(self.probe_ids),
            "probe_values": list(self.probe_values),
            "quantization_bits": self.quantization_bits,
            "uncertainty_value": self.uncertainty_value,
        }

    def verify(self) -> None:
        expected = sha256_bytes(canonical_json_bytes(self.body()))
        if expected != self.checksum:
            raise ValueError("operational packet checksum mismatch")
        if len(self.probe_ids) != len(self.probe_values):
            raise ValueError("operational packet probe/value length mismatch")
        if len(set(self.probe_ids)) != len(self.probe_ids):
            raise ValueError("operational packet contains duplicate probe IDs")

    def serialize(self) -> bytes:
        self.verify()
        payload = dict(self.body())
        payload["checksum"] = self.checksum
        return canonical_json_bytes(payload)

    @classmethod
    def deserialize(cls, data: bytes) -> OperationalPacketV1:
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid operational packet encoding") from error
        if payload.get("schema") != "FRANK-EQ/OPERATIONAL-PACKET/1":
            raise ValueError("unsupported operational packet schema")
        packet = cls(
            schema=str(payload["schema"]),
            task_family=str(payload["task_family"]),
            query_operation_id=int(payload["query_operation_id"]),
            fact_values=tuple(int(v) for v in payload["fact_values"]),
            probe_ids=tuple(int(v) for v in payload["probe_ids"]),
            probe_values=tuple(int(v) for v in payload["probe_values"]),
            quantization_bits=int(payload["quantization_bits"]),
            uncertainty_value=int(payload["uncertainty_value"]),
            checksum=str(payload["checksum"]),
        )
        packet.verify()
        return packet

    def decoded_fact_probabilities(self) -> np.ndarray:
        logits = dequantize_logits(np.asarray(self.fact_values), self.quantization_bits)
        return logits_to_probabilities(logits)

    def decoded_probe_probabilities(self) -> np.ndarray:
        logits = dequantize_logits(np.asarray(self.probe_values), self.quantization_bits)
        return logits_to_probabilities(logits)
