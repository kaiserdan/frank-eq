from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from frank_eq.rate_compute.backend import RateComputeModelAdapter
from frank_eq.rate_compute.config import ResponseProtocolConfig


class _FakeCache:
    def __init__(self, state: torch.Tensor):
        self.state = state

    def __deepcopy__(self, memo):
        return _FakeCache(self.state.clone())

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.state = self.state.repeat_interleave(repeats, dim=0)


class _FakeCausalModel:
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: _FakeCache,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert attention_mask.shape[0] == input_ids.shape[0]
        assert return_dict is True
        cumulative = past_key_values.state[:, None] + input_ids.float().cumsum(dim=1)
        logits = torch.zeros((*input_ids.shape, 16), dtype=torch.float32)
        logits[:, :, 1] = -cumulative / 10.0
        logits[:, :, 2] = cumulative / 10.0
        logits[:, :, 3] = torch.remainder(cumulative, 5.0)
        updated = _FakeCache(past_key_values.state + input_ids.float().sum(dim=1))
        return SimpleNamespace(
            logits=logits,
            past_key_values=updated if use_cache else None,
        )


class _FakeTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_tensors: str,
        truncation: bool,
    ) -> dict[str, torch.Tensor]:
        assert add_special_tokens is False
        assert return_tensors == "pt"
        assert truncation is False
        mapping = {" false": [1], " true": [2]}
        return {"input_ids": torch.tensor([mapping.get(text, [4, 5])])}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [6]

    @staticmethod
    def decode(tokens: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return ",".join(str(token) for token in tokens)


class _FakeCapture:
    max_length = 128


def _adapter() -> RateComputeModelAdapter:
    adapter = object.__new__(RateComputeModelAdapter)
    adapter.capture = _FakeCapture()
    adapter.device = torch.device("cpu")
    adapter.model = _FakeCausalModel()
    adapter.tokenizer = _FakeTokenizer()
    adapter.answer_ids = (1, 2)
    return adapter


@pytest.mark.parametrize("protocol", ["answer_token", "sequence", "reason", "pause"])
def test_batched_protocols_match_query_exclusive_scalar_branches(protocol: str) -> None:
    adapter = _adapter()
    protocols = ResponseProtocolConfig(rationale_budget=3, pause_budget=3)
    prefix_ids = torch.tensor([[7, 8]])
    prefix_cache = _FakeCache(torch.tensor([15.0]))
    queries = [torch.tensor([[3, 4]]), torch.tensor([[5, 2]])]

    if protocol == "answer_token":
        batched = adapter.score_answer_token_batch(prefix_ids, prefix_cache, queries)
        scalar = [
            adapter.score_answer_token(prefix_ids, prefix_cache, query) for query in queries
        ]
    elif protocol == "sequence":
        batched = adapter.score_sequence_batch(
            prefix_ids, prefix_cache, queries, protocols
        )
        scalar = [
            adapter.score_sequence(prefix_ids, prefix_cache, query, protocols)
            for query in queries
        ]
    else:
        batched = adapter.score_with_compute_batch(
            prefix_ids,
            prefix_cache,
            queries,
            protocols,
            mode=protocol,
        )
        scalar = [
            adapter.score_with_compute(
                prefix_ids,
                prefix_cache,
                query,
                protocols,
                mode=protocol,
            )
            for query in queries
        ]

    assert prefix_cache.state.tolist() == [15.0]
    assert [score.probability_true for score in batched] == pytest.approx(
        [score.probability_true for score in scalar]
    )
    assert [score.log_odds_score for score in batched] == pytest.approx(
        [score.log_odds_score for score in scalar]
    )
    assert [score.generated_text for score in batched] == [
        score.generated_text for score in scalar
    ]
    assert [score.generated_token_count for score in batched] == [
        score.generated_token_count for score in scalar
    ]
