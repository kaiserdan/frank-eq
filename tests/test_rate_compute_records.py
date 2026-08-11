import torch

from frank_eq.rate_compute.backend import ProtocolScore, render_deliberation_query
from frank_eq.rate_compute.config import RateComputeRunConfig, ResponseProtocolConfig
from frank_eq.rate_compute.records import (
    _calibration_key,
    _PendingBranch,
    _score_pending_branches,
)
from frank_eq.schemas import OperationDefinition


def _response(item_id: int) -> dict[str, object]:
    return {
        "model_id": "model-a",
        "entity_count": 4,
        "kind": "basis",
        "item_id": item_id,
        "family": "edge",
        "protocol": "sequence",
    }


def test_calibration_is_coordinate_specific() -> None:
    assert _calibration_key(_response(0)) != _calibration_key(_response(1))


def test_direct_target_calibration_remains_family_specific() -> None:
    first = {
        **_response(0),
        "kind": "target",
        "family": "compose",
    }
    second = {
        **first,
        "item_id": 1,
    }

    assert _calibration_key(first) == _calibration_key(second)


def test_reasoning_query_reserves_semantic_answer_for_later_cue() -> None:
    protocols = ResponseProtocolConfig()
    operation = OperationDefinition(
        operation_id=0,
        family="compose",
        fact_args=(0, 3),
        residual_args=(1, 2),
        polarity=-1.0,
    )
    query = render_deliberation_query(operation, 4, protocols)

    assert "it is false that" in query
    assert protocols.reasoning_instruction in query
    assert query.endswith("Scratchpad:")
    assert "Reply with exactly" not in query


class _BatchRecordingAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    @staticmethod
    def _scores(query_ids: list[torch.Tensor]) -> list[ProtocolScore]:
        return [
            ProtocolScore(
                probability_true=float(query[0, 0]),
                log_odds_score=float(query[0, 0]),
                false_token_count=1,
                true_token_count=1,
                generated_token_count=0,
                generated_text="",
            )
            for query in query_ids
        ]

    def score_answer_token_batch(self, prefix_ids, prefix_cache, query_ids):
        self.calls.append(("answer_token", len(query_ids)))
        return self._scores(query_ids)

    def score_sequence_batch(self, prefix_ids, prefix_cache, query_ids, protocols):
        self.calls.append(("sequence", len(query_ids)))
        return self._scores(query_ids)

    def score_with_compute_batch(self, prefix_ids, prefix_cache, query_ids, protocols, *, mode):
        self.calls.append((mode, len(query_ids)))
        return self._scores(query_ids)


def test_pending_branches_batch_by_protocol_and_length_without_reordering() -> None:
    config = RateComputeRunConfig()
    config.capture.branch_batch_size = 2
    protocols = ["sequence", "answer_token", "sequence", "reason", "sequence", "pause"]
    lengths = [2, 1, 2, 3, 2, 3]
    pending = [
        _PendingBranch(
            row={"row_id": index},
            protocol=protocol,
            query_ids=torch.full((1, length), index + 1, dtype=torch.long),
        )
        for index, (protocol, length) in enumerate(zip(protocols, lengths, strict=True))
    ]
    adapter = _BatchRecordingAdapter()

    rows, stats = _score_pending_branches(
        adapter,
        torch.ones((1, 2), dtype=torch.long),
        object(),
        pending,
        config,
    )

    assert [row["row_id"] for row in rows] == list(range(6))
    assert [row["probability_true"] for row in rows] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert adapter.calls == [
        ("sequence", 2),
        ("sequence", 1),
        ("answer_token", 1),
        ("reason", 1),
        ("pause", 1),
    ]
    assert stats.response_batches == 5
    assert stats.max_observed_batch_size == 2
