"""Hugging Face backend for causally ordered real-model Stage-A caches."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from frank_eq.contracts import FutureBranchRecord, FutureSignatureRecord, StateCaptureRecord
from frank_eq.real_config import CaptureConfig, RealModelSpec
from frank_eq.utils import sha256_bytes

from .real_panel import RealPanel, render_operation_query, render_world_prefix

# Frozen v2-1 system turn. Historical ``prompt_format=chat`` ended the cached
# prefix at an assistant generation header, after which the operation text was
# appended as assistant content. That path is retained for exact
# reproducibility. New qualification protocols use ``chat_turn`` below.
CHAT_SYSTEM_CONTRACT = (
    "You are a careful reasoner over a small described world. "
    "Answer questions about the world based only on the description given."
)
CHAT_ACKNOWLEDGEMENT = (
    "I have stored the world description. I will answer the next registered operation."
)


@dataclass(slots=True)
class CapturedModelRows:
    """All renderer/world rows extracted from one frozen checkpoint."""

    hidden: np.ndarray
    world_ids: np.ndarray
    renderer_ids: np.ndarray
    branch_signatures: np.ndarray
    records: list[FutureSignatureRecord]
    hidden_dim: int
    layer_indices: list[int]
    answer_labels: tuple[str, str]
    branch_mode_counts: dict[str, int]
    parity_audit: dict[str, Any]
    model_revision: str | None


def resolve_layer_indices(num_hidden_layers: int, normalized_depths: list[float]) -> list[int]:
    """Map normalized depths to Transformer hidden-state tuple indices."""

    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be positive")
    indices = [
        max(1, min(num_hidden_layers, int(round(depth * num_hidden_layers))))
        for depth in normalized_depths
    ]
    if len(indices) != len(set(indices)):
        raise ValueError(
            f"normalized depths collapse to duplicate layer indices {indices}; choose a wider grid"
        )
    return indices


def resolve_torch_dtype(name: str) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return aliases[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported capture dtype: {name}") from error


def choose_answer_token_pair(
    tokenizer: Any,
    candidates: list[list[str]],
) -> tuple[tuple[str, str], tuple[int, int]]:
    """Choose a false/true label pair that is exactly one token for a tokenizer."""

    for pair in candidates:
        if len(pair) != 2:
            continue
        false_ids = tokenizer.encode(pair[0], add_special_tokens=False)
        true_ids = tokenizer.encode(pair[1], add_special_tokens=False)
        if len(false_ids) == 1 and len(true_ids) == 1 and false_ids[0] != true_ids[0]:
            return (str(pair[0]), str(pair[1])), (int(false_ids[0]), int(true_ids[0]))
    raise RuntimeError(
        "no registered false/true answer pair is single-token under this tokenizer"
    )


def _clone_tensor_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    return copy.deepcopy(value)


def clone_past_key_values(cache: Any) -> Any:
    """Clone tuple or modern Transformers cache objects for independent branches."""

    if cache is None:
        raise ValueError("cannot clone an absent KV cache")
    if isinstance(cache, (tuple, list, dict)):
        return _clone_tensor_tree(cache)
    if hasattr(cache, "to_legacy_cache"):
        legacy = _clone_tensor_tree(cache.to_legacy_cache())
        factory = getattr(type(cache), "from_legacy_cache", None)
        if callable(factory):
            return factory(legacy)
        return legacy
    return copy.deepcopy(cache)


def _safe_model_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


class HFModelAdapter:
    """Sequentially extract one checkpoint without exposing any future query at capture."""

    def __init__(self, spec: RealModelSpec, capture: CaptureConfig):
        self.spec = spec
        self.capture = capture
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "real-model extraction requires `pip install -e '.[real]'`"
            ) from error

        tokenizer_id = spec.tokenizer_id or spec.hf_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=capture.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = resolve_torch_dtype(capture.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=capture.local_files_only,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.device = torch.device(capture.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("capture.device requests CUDA but CUDA is unavailable")
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        num_layers = int(getattr(self.model.config, "num_hidden_layers", 0))
        if num_layers < 1:
            raise RuntimeError("checkpoint config does not expose num_hidden_layers")
        self.layer_indices = resolve_layer_indices(num_layers, capture.normalized_depths)
        self.answer_labels, self.answer_ids = choose_answer_token_pair(
            self.tokenizer, capture.answer_token_pairs
        )

    def _apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            raise RuntimeError(
                f"capture.prompt_format requires a chat template but {self.spec.model_id} "
                "tokenizer has none"
            )
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **dict(self.capture.chat_template_kwargs),
            )
        except Exception as error:
            raise RuntimeError(
                f"chat template application failed for {self.spec.model_id}: {error}"
            ) from error

    @staticmethod
    def _base_chat_messages(world_statement: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": CHAT_SYSTEM_CONTRACT},
            {"role": "user", "content": world_statement},
        ]

    @staticmethod
    def _chat_turn_messages(world_statement: str) -> list[dict[str, str]]:
        """Candidate conversation for ``prompt_format=chat_turn``.

        The query-blind world statement is placed in the system message and the
        fixed acknowledgement is the only assistant message, so the cached
        prefix contains no assistant turn that follows a user turn. Qwen3's
        chat template renders such turns context-dependently (a final post-query
        assistant message gains a ``<think>`` wrapper that disappears once a
        later user message exists), which would break exact-prefix continuity;
        this construction renders identically in the prefix and in the
        full-conversation reveal for every frozen checkpoint.
        """

        return [
            {
                "role": "system",
                "content": f"{CHAT_SYSTEM_CONTRACT}\n\n{world_statement}",
            },
            {"role": "assistant", "content": CHAT_ACKNOWLEDGEMENT},
        ]

    def _format_prefix(self, prefix: str) -> str:
        if self.capture.prompt_format == "raw":
            return prefix
        if self.capture.prompt_format == "chat":
            # Historical v2-1 contract. The operation suffix was appended after
            # this assistant header, so it became assistant content rather than
            # a new user turn. Retain only for reproducing the adopted result.
            return self._apply_chat_template(
                self._base_chat_messages(prefix),
                add_generation_prompt=True,
            )
        if self.capture.prompt_format == "chat_turn":
            return self._apply_chat_template(
                self._chat_turn_messages(prefix),
                add_generation_prompt=False,
            )
        raise ValueError(f"unsupported capture.prompt_format: {self.capture.prompt_format}")

    def _tokenize(self, text: str, *, add_special_tokens: bool | None = None) -> torch.Tensor:
        if add_special_tokens is None:
            # Chat-formatted strings carry their own opening and turn markers.
            add_special_tokens = self.capture.prompt_format == "raw"
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            truncation=False,
        )["input_ids"]
        if encoded.shape[1] > self.capture.max_length:
            raise RuntimeError(
                f"sequence length {encoded.shape[1]} exceeds max_length={self.capture.max_length}"
            )
        return encoded.to(self.device)

    def _query_ids(
        self,
        query: str,
        *,
        world_statement: str | None = None,
        prefix_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.capture.prompt_format != "chat_turn":
            encoded = self.tokenizer(
                query,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=False,
            )["input_ids"]
            return encoded.to(self.device)

        if world_statement is None or prefix_ids is None:
            raise ValueError("chat_turn suffix construction requires world_statement and prefix_ids")
        messages = self._chat_turn_messages(world_statement)
        messages.append({"role": "user", "content": query})
        full_text = self._apply_chat_template(messages, add_generation_prompt=True)
        full_ids = self._tokenize(full_text, add_special_tokens=False)
        prefix_length = int(prefix_ids.shape[1])
        if full_ids.shape[1] <= prefix_length:
            raise RuntimeError("chat_turn operation suffix is empty")
        if not torch.equal(full_ids[:, :prefix_length], prefix_ids):
            raise RuntimeError(
                "chat_turn template violates exact prefix continuity; operation reveal would "
                "change the cached token history"
            )
        return full_ids[:, prefix_length:]

    def _probability_from_logits(self, logits: torch.Tensor) -> float:
        pair = logits[list(self.answer_ids)].float()
        return float(torch.softmax(pair, dim=0)[1].item())

    def _branch_exact_replay(self, prefix_ids: torch.Tensor, query_ids: torch.Tensor) -> float:
        input_ids = torch.cat([prefix_ids, query_ids], dim=1)
        if input_ids.shape[1] > self.capture.max_length:
            raise RuntimeError("prefix plus operation query exceeds capture.max_length")
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, use_cache=False, return_dict=True)
        return self._probability_from_logits(output.logits[0, -1])

    def _branch_kv_reuse(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: torch.Tensor,
    ) -> float:
        cache = clone_past_key_values(prefix_cache)
        total_length = prefix_ids.shape[1] + query_ids.shape[1]
        if total_length > self.capture.max_length:
            raise RuntimeError("prefix plus operation query exceeds capture.max_length")
        attention_mask = torch.ones((1, total_length), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(
                input_ids=query_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=False,
                return_dict=True,
            )
        return self._probability_from_logits(output.logits[0, -1])

    def capture_panel(self, panel: RealPanel, split_by_world: dict[int, str]) -> CapturedModelRows:
        hidden_rows: list[np.ndarray] = []
        world_ids: list[int] = []
        renderer_ids: list[int] = []
        branch_rows: list[np.ndarray] = []
        records: list[FutureSignatureRecord] = []
        mode_counts = {"kv_reuse": 0, "exact_prefix_replay": 0}
        parity_audit: dict[str, Any] = {
            "sample_size": self.capture.parity_sample_size,
            "prompt_format": self.capture.prompt_format,
            "entries": [],
        }
        false_label, true_label = self.answer_labels
        model_slug = _safe_model_slug(self.spec.model_id)
        hidden_dim: int | None = None

        for world in panel.worlds:
            split = split_by_world[world.world_id]
            for renderer_id in range(int(panel.config["n_renderers"])):
                world_statement = render_world_prefix(world, renderer_id)
                prefix = self._format_prefix(world_statement)
                prefix_ids = self._tokenize(prefix)
                request_cache = self.capture.branch_mode in {"auto", "kv_reuse"}
                with torch.inference_mode():
                    prefix_output = self.model(
                        input_ids=prefix_ids,
                        output_hidden_states=True,
                        use_cache=request_cache,
                        return_dict=True,
                    )
                if prefix_output.hidden_states is None:
                    raise RuntimeError("checkpoint did not return hidden states")
                selected = torch.stack(
                    [prefix_output.hidden_states[index][0, -1] for index in self.layer_indices],
                    dim=0,
                ).float().cpu().numpy()
                hidden_dim = int(selected.shape[1])
                state_id = f"w{world.world_id:06d}-{model_slug}-r{renderer_id}"
                hidden_digest = sha256_bytes(np.asarray(selected, dtype=np.float32).tobytes())
                capture_record = StateCaptureRecord(
                    state_id=state_id,
                    world_id=str(world.world_id),
                    model_id=self.spec.model_id,
                    renderer_id=str(renderer_id),
                    split=split,
                    prefix_sha256=sha256_bytes(prefix.encode("utf-8")),
                    hidden_artifact_sha256=hidden_digest,
                    captured_before_operation=True,
                    capture_step=int(prefix_ids.shape[1] - 1),
                )

                probabilities: list[float] = []
                branches: list[FutureBranchRecord] = []
                for operation in panel.operations:
                    query = render_operation_query(
                        operation.definition,
                        panel.n_entities,
                        false_label,
                        true_label,
                    )
                    query_ids = self._query_ids(
                        query,
                        world_statement=world_statement,
                        prefix_ids=prefix_ids,
                    )
                    selected_mode = self.capture.branch_mode
                    if selected_mode == "auto":
                        selected_mode = "kv_reuse"
                    try:
                        if selected_mode == "kv_reuse":
                            probability = self._branch_kv_reuse(
                                prefix_ids, prefix_output.past_key_values, query_ids
                            )
                        else:
                            probability = self._branch_exact_replay(prefix_ids, query_ids)
                    except Exception:
                        if (
                            selected_mode != "kv_reuse"
                            or not self.capture.allow_exact_replay_fallback
                        ):
                            raise
                        selected_mode = "exact_prefix_replay"
                        probability = self._branch_exact_replay(prefix_ids, query_ids)
                    mode_counts[selected_mode] += 1
                    probability = float(np.clip(probability, 1e-7, 1.0 - 1e-7))
                    probabilities.append(probability)
                    if (
                        len(parity_audit["entries"]) < self.capture.parity_sample_size
                        and selected_mode == "kv_reuse"
                    ):
                        try:
                            replay_probability = self._branch_exact_replay(prefix_ids, query_ids)
                        except Exception:
                            # Parity is stack telemetry on an already-verified KV
                            # path; a replay-side failure does not alter that path.
                            pass
                        else:
                            parity_audit["entries"].append(
                                {
                                    "state_id": state_id,
                                    "operation_id": str(operation.definition.operation_id),
                                    "kv_probability": probability,
                                    "replay_probability": float(
                                        np.clip(replay_probability, 1e-7, 1.0 - 1e-7)
                                    ),
                                }
                            )
                    branches.append(
                        FutureBranchRecord(
                            state_id=state_id,
                            operation_id=str(operation.definition.operation_id),
                            operation_descriptor_sha256=operation.descriptor_sha256,
                            outcome_probabilities=(1.0 - probability, probability),
                            branch_seed=self.capture.branch_seed,
                            operation_reveal_step=int(prefix_ids.shape[1]),
                        )
                    )
                record = FutureSignatureRecord(capture=capture_record, branches=tuple(branches))
                record.validate(
                    {str(operation.definition.operation_id) for operation in panel.operations}
                )
                records.append(record)
                hidden_rows.append(selected)
                world_ids.append(world.world_id)
                renderer_ids.append(renderer_id)
                branch_rows.append(np.asarray(probabilities, dtype=np.float32))

        if hidden_dim is None:
            raise RuntimeError("panel extraction produced no hidden rows")
        revision = getattr(self.model.config, "_commit_hash", None) or self.spec.revision
        return CapturedModelRows(
            hidden=np.stack(hidden_rows, axis=0).astype(np.float32),
            world_ids=np.asarray(world_ids, dtype=np.int64),
            renderer_ids=np.asarray(renderer_ids, dtype=np.int64),
            branch_signatures=np.stack(branch_rows, axis=0).astype(np.float32),
            records=records,
            hidden_dim=hidden_dim,
            layer_indices=list(self.layer_indices),
            answer_labels=self.answer_labels,
            branch_mode_counts=mode_counts,
            parity_audit=parity_audit,
            model_revision=None if revision is None else str(revision),
        )
