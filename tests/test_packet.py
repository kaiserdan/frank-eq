import json

import numpy as np
import pytest

from frank_eq.packet import OperationalPacketV1, QueryConditionedSelector


def test_packet_roundtrip_and_checksum() -> None:
    packet = OperationalPacketV1.build(
        task_family="test",
        query_operation_id=2,
        fact_probabilities=np.asarray([0.1, 0.8, 0.5], dtype=np.float32),
        signature_probabilities=np.asarray([0.2, 0.4, 0.9, 0.7], dtype=np.float32),
        probe_ids=np.asarray([2, 1, 3], dtype=np.int32),
        quantization_bits=8,
        uncertainty=0.25,
    )
    recovered = OperationalPacketV1.deserialize(packet.serialize())
    assert recovered == packet
    assert recovered.decoded_fact_probabilities().shape == (3,)


def test_packet_tamper_fails() -> None:
    packet = OperationalPacketV1.build(
        task_family="test",
        query_operation_id=0,
        fact_probabilities=np.asarray([0.5], dtype=np.float32),
        signature_probabilities=np.asarray([0.9], dtype=np.float32),
        probe_ids=np.asarray([0], dtype=np.int32),
        quantization_bits=8,
        uncertainty=0.1,
    )
    payload = json.loads(packet.serialize())
    payload["probe_values"][0] += 1
    with pytest.raises(ValueError, match="checksum"):
        OperationalPacketV1.deserialize(json.dumps(payload).encode())


def test_query_selector_is_deterministic() -> None:
    descriptors = np.eye(5, dtype=np.float32)
    selector = QueryConditionedSelector(descriptors)
    selected = selector.select(3, 3)
    assert selected[0] == 3
    np.testing.assert_array_equal(selected, selector.select(3, 3))
