"""Frozen-model response protocols for the rate--compute audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from frank_eq.data.hf_backend import (
    HFModelAdapter,
    clone_past_key_values,
    repeat_past_key_values,
)
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
        if input_ids.ndim != 2 or input_ids.shape[0] < 1:
            raise ValueError("suffix input_ids must have shape [batch, length]")
        if total_length > self.capture.max_length:
            raise RuntimeError("branch sequence exceeds capture.max_length")
        attention_mask = torch.ones(
            (int(input_ids.shape[0]), total_length),
            dtype=torch.long,
            device=self.device,
        )
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

    def candidate_metadata(self, protocols: ResponseProtocolConfig) -> dict[str, Any]:
        """Record the exact semantic candidate strings and tokenizer IDs."""

        return {
            "false": {
                "text": protocols.candidate_false,
                "token_ids": [
                    int(value)
                    for value in self._candidate_ids(protocols.candidate_false)[0].tolist()
                ],
            },
            "true": {
                "text": protocols.candidate_true,
                "token_ids": [
                    int(value)
                    for value in self._candidate_ids(protocols.candidate_true)[0].tolist()
                ],
            },
        }

    def _candidate_log_likelihood_batch(
        self,
        branch_output: Any,
        candidate_ids: torch.Tensor,
        *,
        prefix_length: int,
        normalize: bool,
    ) -> list[float]:
        token_ids = candidate_ids[0]
        batch_size = int(branch_output.logits.shape[0])
        first_distribution = torch.log_softmax(branch_output.logits[:, -1].float(), dim=-1)
        totals = first_distribution[:, int(token_ids[0])].clone()
        if token_ids.numel() > 1:
            continuation = candidate_ids[:, :-1].expand(batch_size, -1).contiguous()
            cache = clone_past_key_values(branch_output.past_key_values)
            output = self._suffix_forward(
                cache,
                continuation,
                total_length=prefix_length + int(continuation.shape[1]),
                use_cache=False,
            )
            distributions = torch.log_softmax(output.logits.float(), dim=-1)
            targets = token_ids[1:].view(1, -1, 1).expand(batch_size, -1, 1)
            totals += distributions.gather(2, targets).squeeze(-1).sum(dim=1)
        if normalize:
            totals /= float(token_ids.numel())
        return [float(value) for value in totals.detach().cpu().tolist()]

    def _score_candidates_batch(
        self,
        branch_output: Any,
        *,
        branch_length: int,
        protocols: ResponseProtocolConfig,
        generated_ids: list[list[int]] | None = None,
    ) -> list[ProtocolScore]:
        batch_size = int(branch_output.logits.shape[0])
        false_ids = self._candidate_ids(protocols.candidate_false)
        true_ids = self._candidate_ids(protocols.candidate_true)
        false_scores = self._candidate_log_likelihood_batch(
            branch_output,
            false_ids,
            prefix_length=branch_length,
            normalize=protocols.normalize_sequence_log_likelihood,
        )
        true_scores = self._candidate_log_likelihood_batch(
            branch_output,
            true_ids,
            prefix_length=branch_length,
            normalize=protocols.normalize_sequence_log_likelihood,
        )
        generated_rows = generated_ids or [[] for _ in range(batch_size)]
        if len(generated_rows) != batch_size:
            raise ValueError("generated token rows must match the response batch size")
        scores: list[ProtocolScore] = []
        for false_score, true_score, generated in zip(
            false_scores, true_scores, generated_rows, strict=True
        ):
            log_odds = float(true_score - false_score)
            probability = float(1.0 / (1.0 + np.exp(-np.clip(log_odds, -50.0, 50.0))))
            generated_text = ""
            if generated:
                generated_text = self.tokenizer.decode(generated, skip_special_tokens=False)
                generated_text = generated_text[: protocols.max_saved_reasoning_characters]
            scores.append(
                ProtocolScore(
                    probability_true=probability,
                    log_odds_score=log_odds,
                    false_token_count=int(false_ids.shape[1]),
                    true_token_count=int(true_ids.shape[1]),
                    generated_token_count=len(generated),
                    generated_text=generated_text,
                )
            )
        return scores

    @staticmethod
    def _stack_queries(query_ids: list[torch.Tensor]) -> torch.Tensor:
        if not query_ids:
            raise ValueError("response batch must contain at least one query")
        query_length = int(query_ids[0].shape[1])
        if any(
            query.ndim != 2 or query.shape[0] != 1 or int(query.shape[1]) != query_length
            for query in query_ids
        ):
            raise ValueError("batched query tensors must all have shape [1, same_length]")
        return torch.cat(query_ids, dim=0)

    def _query_output_batch(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: list[torch.Tensor],
    ) -> tuple[Any, int]:
        query_batch = self._stack_queries(query_ids)
        cache = repeat_past_key_values(prefix_cache, len(query_ids))
        branch_length = int(prefix_ids.shape[1] + query_batch.shape[1])
        output = self._suffix_forward(
            cache,
            query_batch,
            total_length=branch_length,
            use_cache=True,
        )
        return output, branch_length

    def score_answer_token_batch(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: list[torch.Tensor],
    ) -> list[ProtocolScore]:
        output, _ = self._query_output_batch(prefix_ids, prefix_cache, query_ids)
        pairs = output.logits[:, -1, list(self.answer_ids)].float()
        probabilities = torch.softmax(pairs, dim=-1)[:, 1]
        probabilities = probabilities.clamp(1e-7, 1.0 - 1e-7).detach().cpu().tolist()
        return [
            ProtocolScore(
                probability_true=float(probability),
                log_odds_score=float(np.log(probability / (1.0 - probability))),
                false_token_count=1,
                true_token_count=1,
                generated_token_count=0,
                generated_text="",
            )
            for probability in probabilities
        ]

    def score_answer_token(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: torch.Tensor,
    ) -> ProtocolScore:
        return self.score_answer_token_batch(prefix_ids, prefix_cache, [query_ids])[0]

    def score_sequence_batch(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: list[torch.Tensor],
        protocols: ResponseProtocolConfig,
    ) -> list[ProtocolScore]:
        output, branch_length = self._query_output_batch(prefix_ids, prefix_cache, query_ids)
        return self._score_candidates_batch(
            output,
            branch_length=branch_length,
            protocols=protocols,
        )

    def score_sequence(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: torch.Tensor,
        protocols: ResponseProtocolConfig,
    ) -> ProtocolScore:
        return self.score_sequence_batch(prefix_ids, prefix_cache, [query_ids], protocols)[0]

    def _greedy_compute_batch(
        self,
        output: Any,
        *,
        branch_length: int,
        budget: int,
    ) -> tuple[Any, int, list[list[int]]]:
        generated_steps: list[torch.Tensor] = []
        current = output
        length = branch_length
        for _ in range(budget):
            token = torch.argmax(current.logits[:, -1], dim=-1).to(torch.long).unsqueeze(1)
            generated_steps.append(token[:, 0])
            length += 1
            current = self._suffix_forward(
                clone_past_key_values(current.past_key_values),
                token,
                total_length=length,
                use_cache=True,
            )
        if generated_steps:
            generated = torch.stack(generated_steps, dim=1).detach().cpu().tolist()
        else:
            generated = [[] for _ in range(int(output.logits.shape[0]))]
        return current, length, generated

    def _pause_compute_batch(
        self,
        output: Any,
        *,
        branch_length: int,
        budget: int,
        pause_text: str,
    ) -> tuple[Any, int, list[list[int]]]:
        seed_ids = self.tokenizer.encode(pause_text, add_special_tokens=False)
        if not seed_ids:
            raise RuntimeError("pause text tokenized to an empty sequence")
        repeated = [int(seed_ids[index % len(seed_ids)]) for index in range(budget)]
        batch_size = int(output.logits.shape[0])
        pause_ids = (
            torch.tensor([repeated], dtype=torch.long, device=self.device)
            .expand(batch_size, -1)
            .contiguous()
        )
        length = branch_length + budget
        current = self._suffix_forward(
            clone_past_key_values(output.past_key_values),
            pause_ids,
            total_length=length,
            use_cache=True,
        )
        return current, length, [list(repeated) for _ in range(batch_size)]

    def score_with_compute_batch(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: list[torch.Tensor],
        protocols: ResponseProtocolConfig,
        *,
        mode: str,
    ) -> list[ProtocolScore]:
        output, branch_length = self._query_output_batch(prefix_ids, prefix_cache, query_ids)
        if mode == "reason":
            current, branch_length, generated = self._greedy_compute_batch(
                output,
                branch_length=branch_length,
                budget=protocols.rationale_budget,
            )
        elif mode == "pause":
            current, branch_length, generated = self._pause_compute_batch(
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
        cue_batch = cue_ids.expand(len(query_ids), -1).contiguous()
        branch_length += int(cue_ids.shape[1])
        final_output = self._suffix_forward(
            clone_past_key_values(current.past_key_values),
            cue_batch,
            total_length=branch_length,
            use_cache=True,
        )
        return self._score_candidates_batch(
            final_output,
            branch_length=branch_length,
            protocols=protocols,
            generated_ids=generated,
        )

    def score_with_compute(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: torch.Tensor,
        protocols: ResponseProtocolConfig,
        *,
        mode: str,
    ) -> ProtocolScore:
        return self.score_with_compute_batch(
            prefix_ids,
            prefix_cache,
            [query_ids],
            protocols,
            mode=mode,
        )[0]
