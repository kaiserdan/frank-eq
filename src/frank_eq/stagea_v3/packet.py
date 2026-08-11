"""Typed edge packets, exact bit accounting, and the frozen graph executor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from frank_eq.data.real_panel import RelationalWorld
from frank_eq.packet.quantization import (
    dequantize_logits,
    logits_to_probabilities,
    probabilities_to_logits,
    quantize_logits,
)
from frank_eq.rate_compute.logic import edge_vector_to_matrix, execute_public_basis
from frank_eq.schemas import OperationDefinition
from frank_eq.utils import canonical_json_bytes, sha256_bytes

PACKET_SCHEMA = "FRANK-EQ/TYPED-EDGE-PACKET/1"


def _pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    integers = np.asarray(values, dtype=np.int64).reshape(-1)
    if bits < 1 or bits > 16:
        raise ValueError("packet quantization bits must lie in [1,16]")
    if np.any(integers < 0) or np.any(integers >= (1 << bits)):
        raise ValueError("packet integer lies outside the registered bit width")
    output = bytearray((integers.size * bits + 7) // 8)
    bit_cursor = 0
    for value in integers.tolist():
        for shift in range(bits - 1, -1, -1):
            if (int(value) >> shift) & 1:
                output[bit_cursor // 8] |= 1 << (7 - bit_cursor % 8)
            bit_cursor += 1
    return bytes(output)


def _unpack_unsigned(payload: bytes, *, count: int, bits: int) -> np.ndarray:
    if len(payload) * 8 < count * bits:
        raise ValueError("typed packet payload is truncated")
    values = np.zeros(count, dtype=np.int32)
    bit_cursor = 0
    for index in range(count):
        value = 0
        for _ in range(bits):
            value = (value << 1) | (
                (payload[bit_cursor // 8] >> (7 - bit_cursor % 8)) & 1
            )
            bit_cursor += 1
        values[index] = value
    if any(
        (payload[cursor // 8] >> (7 - cursor % 8)) & 1
        for cursor in range(count * bits, len(payload) * 8)
    ):
        raise ValueError("typed packet has nonzero byte-alignment padding")
    return values


@dataclass(frozen=True, slots=True)
class TypedEdgePacket:
    entity_count: int
    bits_per_coordinate: int
    quantized_values: tuple[int, ...]
    payload_hex: str
    checksum_sha256: str
    schema: str = PACKET_SCHEMA
    coordinate_order: str = "row_major_non_diagonal"
    quantizer: str = "uniform_clipped_logit[-8,8]"

    @property
    def coordinate_count(self) -> int:
        return self.entity_count * (self.entity_count - 1)

    @property
    def payload_bits(self) -> int:
        return self.coordinate_count * self.bits_per_coordinate

    def _header(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "entity_count": self.entity_count,
            "coordinate_order": self.coordinate_order,
            "bits_per_coordinate": self.bits_per_coordinate,
            "quantizer": self.quantizer,
            "payload_bits": self.payload_bits,
        }

    @property
    def framing_bits(self) -> int:
        return len(canonical_json_bytes(self._header())) * 8 + 256

    @property
    def serialized_bits(self) -> int:
        return self.payload_bits + self.framing_bits

    def validate(self) -> None:
        if self.schema != PACKET_SCHEMA:
            raise ValueError("typed packet schema changed")
        if self.entity_count not in {4, 6}:
            raise ValueError("typed packet entity count is outside the public registry")
        if self.coordinate_order != "row_major_non_diagonal":
            raise ValueError("typed packet coordinate order changed")
        if self.quantizer != "uniform_clipped_logit[-8,8]":
            raise ValueError("typed packet quantizer changed")
        values = np.asarray(self.quantized_values, dtype=np.int32)
        if values.shape != (self.coordinate_count,):
            raise ValueError("typed packet has the wrong coordinate count")
        payload = bytes.fromhex(self.payload_hex)
        if not np.array_equal(
            _unpack_unsigned(
                payload,
                count=self.coordinate_count,
                bits=self.bits_per_coordinate,
            ),
            values,
        ):
            raise ValueError("typed packet packed payload and coordinate values differ")
        checksum_input = canonical_json_bytes(self._header()) + payload
        if sha256_bytes(checksum_input) != self.checksum_sha256:
            raise ValueError("typed packet checksum mismatch")

    def probabilities(self) -> np.ndarray:
        self.validate()
        logits = dequantize_logits(
            np.asarray(self.quantized_values, dtype=np.int32),
            self.bits_per_coordinate,
            limit=8.0,
        )
        return logits_to_probabilities(logits)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **self._header(),
            "framing_bits": self.framing_bits,
            "serialized_bits": self.serialized_bits,
            "quantized_values": list(self.quantized_values),
            "payload_hex": self.payload_hex,
            "checksum_sha256": self.checksum_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TypedEdgePacket:
        packet = cls(
            entity_count=int(payload["entity_count"]),
            bits_per_coordinate=int(payload["bits_per_coordinate"]),
            quantized_values=tuple(int(value) for value in payload["quantized_values"]),
            payload_hex=str(payload["payload_hex"]),
            checksum_sha256=str(payload["checksum_sha256"]),
            schema=str(payload["schema"]),
            coordinate_order=str(payload["coordinate_order"]),
            quantizer=str(payload["quantizer"]),
        )
        packet.validate()
        if payload.get("payload_bits") != packet.payload_bits:
            raise ValueError("typed packet payload-bit accounting differs")
        if payload.get("framing_bits") != packet.framing_bits:
            raise ValueError("typed packet framing-bit accounting differs")
        if payload.get("serialized_bits") != packet.serialized_bits:
            raise ValueError("typed packet serialized-bit accounting differs")
        return packet


def encode_typed_edge_packet(
    probabilities: np.ndarray,
    *,
    entity_count: int,
    bits: int,
) -> TypedEdgePacket:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    expected = entity_count * (entity_count - 1)
    if values.shape != (expected,) or not np.all(np.isfinite(values)):
        raise ValueError("typed packet input has the wrong shape or non-finite values")
    quantized = quantize_logits(probabilities_to_logits(values, limit=8.0), bits, limit=8.0)
    packed = _pack_unsigned(quantized, bits)
    header = {
        "schema": PACKET_SCHEMA,
        "entity_count": entity_count,
        "coordinate_order": "row_major_non_diagonal",
        "bits_per_coordinate": bits,
        "quantizer": "uniform_clipped_logit[-8,8]",
        "payload_bits": expected * bits,
    }
    packet = TypedEdgePacket(
        entity_count=entity_count,
        bits_per_coordinate=bits,
        quantized_values=tuple(int(value) for value in quantized.tolist()),
        payload_hex=packed.hex(),
        checksum_sha256=sha256_bytes(canonical_json_bytes(header) + packed),
    )
    packet.validate()
    return packet


def encode_rate_matched_text_basis(
    parsed_binary_edges: np.ndarray,
    *,
    entity_count: int,
    bits: int = 4,
) -> TypedEdgePacket:
    """Compress a deterministic canonical-text parse into the primary wire budget.

    The parser is the information source; the wire form uses the same typed ABI
    and exact payload size as the activation packet. This is an oracle-like text
    ceiling, not evidence for an activation advantage over text.
    """

    values = np.asarray(parsed_binary_edges, dtype=np.float64).reshape(-1)
    if np.any((values != 0.0) & (values != 1.0)):
        raise ValueError("rate-matched text basis requires an exact binary parse")
    return encode_typed_edge_packet(values, entity_count=entity_count, bits=bits)


def _poisson_binomial(probabilities: np.ndarray) -> np.ndarray:
    distribution = np.asarray([1.0], dtype=np.float64)
    for probability in np.asarray(probabilities, dtype=np.float64).reshape(-1):
        probability = float(np.clip(probability, 0.0, 1.0))
        updated = np.zeros(distribution.size + 1, dtype=np.float64)
        updated[:-1] += distribution * (1.0 - probability)
        updated[1:] += distribution * probability
        distribution = updated
    return distribution


def _raw_density_and_mutual_count(world: RelationalWorld) -> tuple[int, int]:
    edge = world.edge_array()
    density_count = int(edge.sum())
    mutual_count = sum(
        int(edge[left, right] and edge[right, left])
        for left in range(world.n_entities)
        for right in range(left + 1, world.n_entities)
    )
    return density_count, mutual_count


def panel_control_thresholds(worlds: tuple[RelationalWorld, ...]) -> tuple[float, float]:
    if not worlds:
        raise ValueError("control thresholds require at least one world")
    values = np.asarray([_raw_density_and_mutual_count(world) for world in worlds])
    return float(np.median(values[:, 0])), float(np.median(values[:, 1]))


def execute_typed_basis(
    probabilities: np.ndarray,
    operation: OperationDefinition,
    *,
    entity_count: int,
    control_thresholds: tuple[float, float] | None = None,
) -> float:
    """Execute every frozen graph family with explicit consumer computation."""

    edge = edge_vector_to_matrix(probabilities, entity_count)
    if operation.family not in {"density", "reciprocity"}:
        return execute_public_basis(edge, operation)
    if control_thresholds is None:
        raise ValueError("density/reciprocity execution requires panel thresholds")
    if operation.family == "density":
        distribution = _poisson_binomial(edge[~np.eye(entity_count, dtype=bool)])
        threshold = control_thresholds[0]
    else:
        mutual_probabilities = [
            edge[left, right] * edge[right, left]
            for left in range(entity_count)
            for right in range(left + 1, entity_count)
        ]
        distribution = _poisson_binomial(np.asarray(mutual_probabilities))
        threshold = control_thresholds[1]
    probability = float(
        distribution[
            np.asarray([count > threshold for count in range(distribution.size)], dtype=bool)
        ].sum()
    )
    if operation.polarity < 0:
        probability = 1.0 - probability
    return float(np.clip(probability, 1e-7, 1.0 - 1e-7))


def packet_json_bytes(packet: TypedEdgePacket) -> bytes:
    """Return a deterministic display serialization, excluded from payload bits."""

    return json.dumps(packet.to_dict(), sort_keys=True, separators=(",", ":")).encode()
