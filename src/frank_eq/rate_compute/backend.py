"""Frozen-model response protocols for the rate--compute audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from frank_eq.data.hf_backend import HFModelAdapter, clone_past_key_values
from frank_eq.data.real_panel import _base_operation_clause
from frank_eq.schemas import OperationDefinition

from .config import ResponseProtocolConfig


@dataclass(frozen=True, slots=True)
class ProtocolScore:
    probability_true: float
    log_odds_score: float
    false_token_count: int
    true_token_count: int
    generated_token_count: int
    generated_text: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _truth_clause(operation: OperationDefinition, n_entities: int) -> str:
    clause = _base_operation_clause(operation, n_entities)
    return clause if operation.polarity >= 0 else f"it is false that {clause}"


def render_sequence_query(
    operation: OperationDefinition,
    n_entities: int,
    protocols: ResponseProtocolConfig,
) -> str:
    false_display = protocols.candidate_false.strip()
    true_display = protocols.candidate_true.strip()
    return (
        "\nRegistered operation: decide whether the following statement is true: "
        f"{_truth_clause(operation, n_entities)}. "
        f"Reply with exactly {false_display} for false or {true_display} for true."
        f"{protocols.sequence_cue}"
    )


def render_deliberation_query(
    operation: OperationDefinition,
    n_entities: int,
    protocols: ResponseProtocolConfig,
) -> str:
    return (
        "\nRegistered operation: decide whether the following statement is true: "
        f"{_truth_clause(operation, n_entities)}. "
        f"{protocols.reasoning_instruction}\nScratchpad:"
    )


def render_final_cue(protocols: ResponseProtocolConfig) -> str:
    false_display = protocols.candidate_false.strip()
    true_display = protocols.candidate_true.strip()
    return (
        f"{protocols.final_cue} Reply with exactly {false_display} for false "
        f"or {true_display} for true. Answer:"
    )


class RateComputeModelAdapter(HFModelAdapter):
    """Extend the causal cache adapter with sequence and compute-aware readouts."""

    def _suffix_forward(
        self,
        cache: Any,
        input_ids: torch.Tensor,
        *,
        total_length: int,
        use_cache: bool = True,
    ) -> Any:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("suffix input_ids must have shape [1, length]")
        if total_length > self.capture.max_length:
            raise RuntimeError("branch sequence exceeds capture.max_length")
        attention_mask = torch.ones((1, total_length), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=use_cache,
                return_dict=True,
            )

    def _candidate_ids(self, candidate: str) -> torch.Tensor:
        ids = self.tokenizer(
            candidate,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        )["input_ids"].to(self.device)
        if ids.shape[1] < 1:
            raise RuntimeError(f"candidate {candidate!r} tokenized to an empty sequence")
        return ids

    def _candidate_log_likelihood(
        self,
        branch_output: Any,
        candidate_ids: torch.Tensor,
        *,
        prefix_length: int,
        normalize: bool,
    ) -> float:
        token_ids = candidate_ids[0]
        first_distribution = torch.log_softmax(branch_output.logits[0, -1].float(), dim=-1)
        total = float(first_distribution[int(token_ids[0])].item())
        if token_ids.numel() > 1:
            continuation = candidate_ids[:, :-1]
            cache = clone_past_key_values(branch_output.past_key_values)
            output = self._suffix_forward(
                cache,
                continuation,
                total_length=prefix_length + int(continuation.shape[1]),
                use_cache=False,
            )
            distributions = torch.log_softmax(output.logits[0].float(), dim=-1)
            for position, token_id in enumerate(token_ids[1:]):
                total += float(distributions[position, int(token_id)].item())
        if normalize:
            total /= float(token_ids.numel())
        return total

    def _score_candidates(
        self,
        branch_output: Any,
        *,
        branch_length: int,
        protocols: ResponseProtocolConfig,
        generated_ids: list[int] | None = None,
    ) -> ProtocolScore:
        false_ids = self._candidate_ids(protocols.candidate_false)
        true_ids = self._candidate_ids(protocols.candidate_true)
        false_score = self._candidate_log_likelihood(
            branch_output,
            false_ids,
            prefix_length=branch_length,
            normalize=protocols.normalize_sequence_log_likelihood,
        )
        true_score = self._candidate_log_likelihood(
            branch_output,
            true_ids,
            prefix_length=branch_length,
            normalize=protocols.normalize_sequence_log_likelihood,
        )
        log_odds = float(true_score - false_score)
        probability = float(1.0 / (1.0 + np.exp(-np.clip(log_odds, -50.0, 50.0))))
        generated = list(generated_ids or [])
        generated_text = ""
        if generated:
            generated_text = self.tokenizer.decode(generated, skip_special_tokens=False)
            generated_text = generated_text[: protocols.max_saved_reasoning_characters]
        return ProtocolScore(
            probability_true=probability,
            log_odds_score=log_odds,
            false_token_count=int(false_ids.shape[1]),
            true_token_count=int(true_ids.shape[1]),
            generated_token_count=len(generated),
            generated_text=generated_text,
        )

    def _query_output(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: torch.Tensor,
    ) -> tuple[Any, int]:
        cache = clone_past_key_values(prefix_cache)
        branch_length = int(prefix_ids.shape[1] + query_ids.shape[1])
        output = self._suffix_forward(
            cache,
            query_ids,
            total_length=branch_length,
            use_cache=True,
        )
        return output, branch_length

    def score_answer_token(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: torch.Tensor,
    ) -> ProtocolScore:
        output, _ = self._query_output(prefix_ids, prefix_cache, query_ids)
        probability = self._probability_from_logits(output.logits[0, -1])
        probability = float(np.clip(probability, 1e-7, 1.0 - 1e-7))
        return ProtocolScore(
            probability_true=probability,
            log_odds_score=float(np.log(probability / (1.0 - probability))),
            false_token_count=1,
            true_token_count=1,
            generated_token_count=0,
            generated_text="",
        )

    def score_sequence(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: torch.Tensor,
        protocols: ResponseProtocolConfig,
    ) -> ProtocolScore:
        output, branch_length = self._query_output(prefix_ids, prefix_cache, query_ids)
        return self._score_candidates(
            output,
            branch_length=branch_length,
            protocols=protocols,
        )

    def _greedy_compute(
        self,
        output: Any,
        *,
        branch_length: int,
        budget: int,
    ) -> tuple[Any, int, list[int]]:
        generated: list[int] = []
        current = output
        length = branch_length
        for _ in range(budget):
            token_id = int(torch.argmax(current.logits[0, -1]).item())
            generated.append(token_id)
            token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            length += 1
            current = self._suffix_forward(
                clone_past_key_values(current.past_key_values),
                token,
                total_length=length,
                use_cache=True,
            )
        return current, length, generated

    def _pause_compute(
        self,
        output: Any,
        *,
        branch_length: int,
        budget: int,
        pause_text: str,
    ) -> tuple[Any, int, list[int]]:
        seed_ids = self.tokenizer.encode(pause_text, add_special_tokens=False)
        if not seed_ids:
            raise RuntimeError("pause text tokenized to an empty sequence")
        repeated = [int(seed_ids[index % len(seed_ids)]) for index in range(budget)]
        pause_ids = torch.tensor([repeated], dtype=torch.long, device=self.device)
        length = branch_length + budget
        current = self._suffix_forward(
            clone_past_key_values(output.past_key_values),
            pause_ids,
            total_length=length,
            use_cache=True,
        )
        return current, length, repeated

    def score_with_compute(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: torch.Tensor,
        protocols: ResponseProtocolConfig,
        *,
        mode: str,
    ) -> ProtocolScore:
        output, branch_length = self._query_output(prefix_ids, prefix_cache, query_ids)
        if mode == "reason":
            current, branch_length, generated = self._greedy_compute(
                output,
                branch_length=branch_length,
                budget=protocols.rationale_budget,
            )
        elif mode == "pause":
            current, branch_length, generated = self._pause_compute(
                output,
                branch_length=branch_length,
                budget=protocols.pause_budget,
                pause_text=protocols.pause_text,
            )
        else:
            raise ValueError("compute mode must be reason or pause")

        cue_ids = self.tokenizer(
            render_final_cue(protocols),
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        )["input_ids"].to(self.device)
        branch_length += int(cue_ids.shape[1])
        final_output = self._suffix_forward(
            clone_past_key_values(current.past_key_values),
            cue_ids,
            total_length=branch_length,
            use_cache=True,
        )
        return self._score_candidates(
            final_output,
            branch_length=branch_length,
            protocols=protocols,
            generated_ids=generated,
        )
