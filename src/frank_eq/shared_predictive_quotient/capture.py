"""Query-blind SPQ0 capture surfaces and categorical future forecasting."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from frank_eq.data.hf_backend import CHAT_ACKNOWLEDGEMENT, CHAT_SYSTEM_CONTRACT
from frank_eq.rate_compute.backend import RateComputeModelAdapter
from frank_eq.real_config import CaptureConfig, RealModelSpec
from frank_eq.telemetry import WandbTelemetry
from frank_eq.utils import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file

from .automaton import ControlledSystem, SharedPredictiveBasis
from .config import SPQModelSpec, SPQRunConfig
from .panel import (
    RENDERER_IDS,
    ROLE_IDS,
    SYSTEM_ROLE_IDS,
    SPQPanel,
    render_prefix,
    render_probability_query,
)


@dataclass(frozen=True, slots=True)
class CategoricalScore:
    log_likelihoods: tuple[float, ...]


def _compat_capture(config: SPQRunConfig) -> CaptureConfig:
    capture = config.capture
    return CaptureConfig(
        normalized_depths=list(capture.normalized_depths),
        max_length=capture.max_length,
        dtype=capture.dtype,
        device=capture.device,
        branch_mode=capture.branch_mode,
        allow_exact_replay_fallback=capture.allow_exact_replay_fallback,
        answer_token_pairs=[[" A", " B"], ["A", "B"], [" No", " Yes"]],
        branch_seed=0,
        branch_batch_size=capture.branch_batch_size,
        local_files_only=capture.local_files_only,
        prompt_format=capture.prompt_format,
        chat_template_kwargs=dict(capture.chat_template_kwargs),
        parity_sample_size=0,
        parity_max_abs_diff=0.01,
    )


class SPQModelAdapter(RateComputeModelAdapter):
    """Frozen checkpoint adapter exposing categorical probability forecasts."""

    def __init__(self, model: SPQModelSpec, config: SPQRunConfig):
        spec = RealModelSpec(
            model_id=model.model_id,
            hf_id=model.hf_id,
            role=model.role,
            tokenizer_id=model.tokenizer_id,
            revision=model.revision,
            trust_remote_code=model.trust_remote_code,
        )
        super().__init__(spec, _compat_capture(config))
        self.spq_model = model
        self.spq_config = config
        self._candidate_ids_cache = [
            self._candidate_ids(label)
            for label in config.probability_protocol.candidate_labels
        ]

    @staticmethod
    def _spq_prefix_messages(world_statement: str) -> list[dict[str, str]]:
        """Use a Qwen/Mistral-compatible prefix that ends at a user turn.

        Mistral requires strict user/assistant alternation after an optional
        system message. Qwen renders a trailing assistant message differently
        once a later user message exists. A single first user turn avoids both
        context dependencies; the fixed acknowledgement is part of every
        post-capture branch and still precedes the future-test reveal.
        """

        return [
            {
                "role": "user",
                "content": (
                    f"{CHAT_SYSTEM_CONTRACT}\n\n{world_statement}\n\n"
                    "Store this complete controlled-system history. The future test remains "
                    "unselected; wait for the next user turn before forecasting."
                ),
            }
        ]

    def _format_prefix(self, prefix: str) -> str:
        if self.capture.prompt_format != "chat_turn":
            return super()._format_prefix(prefix)
        return self._apply_chat_template(
            self._spq_prefix_messages(prefix),
            add_generation_prompt=False,
        )

    def _query_ids(
        self,
        query: str,
        *,
        world_statement: str | None = None,
        prefix_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.capture.prompt_format != "chat_turn":
            return super()._query_ids(
                query,
                world_statement=world_statement,
                prefix_ids=prefix_ids,
            )
        if world_statement is None or prefix_ids is None:
            raise ValueError(
                "SPQ0 chat_turn suffix construction requires a world statement and prefix IDs"
            )
        messages = self._spq_prefix_messages(world_statement)
        messages.extend(
            [
                {"role": "assistant", "content": CHAT_ACKNOWLEDGEMENT},
                {"role": "user", "content": query},
            ]
        )
        full_text = self._apply_chat_template(messages, add_generation_prompt=True)
        full_ids = self._tokenize(full_text, add_special_tokens=False)
        prefix_length = int(prefix_ids.shape[1])
        if full_ids.shape[1] <= prefix_length:
            raise RuntimeError("SPQ0 chat_turn future-test suffix is empty")
        if not torch.equal(full_ids[:, :prefix_length], prefix_ids):
            raise RuntimeError(
                "SPQ0 chat_turn template violates exact prefix continuity"
            )
        return full_ids[:, prefix_length:]

    def categorical_candidate_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "label": label,
                "probability_bin": float(probability),
                "token_ids": [int(value) for value in ids[0].detach().cpu().tolist()],
                "token_count": int(ids.shape[1]),
            }
            for label, probability, ids in zip(
                self.spq_config.probability_protocol.candidate_labels,
                self.spq_config.probability_protocol.bins,
                self._candidate_ids_cache,
                strict=True,
            )
        ]

    def score_categorical_batch(
        self,
        prefix_ids: torch.Tensor,
        prefix_cache: Any,
        query_ids: list[torch.Tensor],
    ) -> list[CategoricalScore]:
        """Score all registered bins from one query branch per future test."""

        branch_output, branch_length = self._query_output_batch(
            prefix_ids,
            prefix_cache,
            query_ids,
        )
        candidate_scores = [
            self._candidate_log_likelihood_batch(
                branch_output,
                candidate_ids,
                prefix_length=branch_length,
                normalize=(
                    self.spq_config.probability_protocol.normalize_candidate_log_likelihoods
                ),
            )
            for candidate_ids in self._candidate_ids_cache
        ]
        result: list[CategoricalScore] = []
        for row in range(len(query_ids)):
            result.append(
                CategoricalScore(
                    log_likelihoods=tuple(
                        float(candidate[row]) for candidate in candidate_scores
                    )
                )
            )
        return result


def _score_queries(
    adapter: SPQModelAdapter,
    prefix_ids: torch.Tensor,
    prefix_cache: Any,
    query_ids: list[torch.Tensor],
    *,
    batch_size: int,
) -> list[CategoricalScore]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, query in enumerate(query_ids):
        groups[int(query.shape[1])].append(index)
    result: list[CategoricalScore | None] = [None] * len(query_ids)
    for query_length in sorted(groups):
        indices = groups[query_length]
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset : offset + batch_size]
            scores = adapter.score_categorical_batch(
                prefix_ids,
                prefix_cache,
                [query_ids[index] for index in selected],
            )
            for index, score in zip(selected, scores, strict=True):
                result[index] = score
    if any(score is None for score in result):
        raise RuntimeError("categorical response batching left an unscored future test")
    return [score for score in result if score is not None]


def _formatted_event_token_indices(
    adapter: SPQModelAdapter,
    prefix_text: str,
    markers: Iterable[str],
    prefix_ids: torch.Tensor,
) -> np.ndarray:
    """Locate event ends in the exact formatted prefix using tokenizer offsets."""

    try:
        encoded = adapter.tokenizer(
            prefix_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"{adapter.spq_model.model_id} tokenizer cannot expose exact event offsets: {error}"
        ) from error
    observed_ids = encoded["input_ids"]
    if not torch.equal(observed_ids.cpu(), prefix_ids.detach().cpu()):
        raise RuntimeError("event-offset tokenization differs from the captured prefix")
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("SPQ0 event-boundary capture requires a fast tokenizer offset map")
    offset_rows = [(int(start), int(end)) for start, end in offsets[0].tolist()]
    indices: list[int] = []
    cursor = 0
    for marker in markers:
        marker_start = prefix_text.find(marker, cursor)
        if marker_start < 0 or prefix_text.find(marker, marker_start + 1) >= 0:
            raise RuntimeError("event marker is absent or non-unique in the formatted prefix")
        marker_end = marker_start + len(marker)
        candidates = [
            index
            for index, (start, end) in enumerate(offset_rows)
            if end > start and start < marker_end and end <= marker_end
        ]
        if not candidates:
            candidates = [
                index
                for index, (start, end) in enumerate(offset_rows)
                if end > start and start < marker_end <= end
            ]
        if not candidates:
            raise RuntimeError("could not locate an event end in tokenizer offsets")
        token_index = max(candidates)
        if indices and token_index <= indices[-1]:
            raise RuntimeError("event token indices are not strictly increasing")
        indices.append(token_index)
        cursor = marker_end
    return np.asarray(indices, dtype=np.int64)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)
    return sha256_file(path)


def _pad_token_rows(
    token_rows: list[np.ndarray],
    boundary_rows: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_tokens = max(len(row) for row in token_rows)
    max_events = max(len(row) for row in boundary_rows)
    token_ids = np.zeros((len(token_rows), max_tokens), dtype=np.int64)
    attention_mask = np.zeros((len(token_rows), max_tokens), dtype=np.bool_)
    event_indices = np.full((len(token_rows), max_events), -1, dtype=np.int64)
    for index, (tokens, boundaries) in enumerate(
        zip(token_rows, boundary_rows, strict=True)
    ):
        token_ids[index, : len(tokens)] = tokens
        attention_mask[index, : len(tokens)] = True
        event_indices[index, : len(boundaries)] = boundaries
    return token_ids, attention_mask, event_indices


def capture_model(
    config: SPQRunConfig,
    model_spec: SPQModelSpec,
    systems: tuple[ControlledSystem, ...],
    basis: SharedPredictiveBasis,
    panels: dict[str, SPQPanel],
    root: Path,
    telemetry: WandbTelemetry,
) -> dict[str, Any]:
    """Capture every registered surface and categorical future signature."""

    adapter = SPQModelAdapter(model_spec, config)
    system_by_id = {system.system_id: system for system in systems}
    tests = (*basis.public_tests, *basis.target_tests)
    final_features: list[np.ndarray] = []
    event_boundary_features: list[np.ndarray] = []
    all_token_features: list[np.ndarray] = []
    embedding_features: list[np.ndarray] = []
    token_rows: list[np.ndarray] = []
    boundary_rows: list[np.ndarray] = []
    history_ids: list[int] = []
    role_ids: list[int] = []
    system_ids: list[int] = []
    system_role_ids: list[int] = []
    renderer_ids: list[int] = []
    lengths: list[int] = []
    last_observations: list[int] = []
    observation_frequencies: list[np.ndarray] = []
    semantic_public: list[np.ndarray] = []
    semantic_core: list[np.ndarray] = []
    semantic_targets: list[np.ndarray] = []
    categorical_log_likelihoods: list[np.ndarray] = []
    prefix_hashes: list[str] = []
    exact_prefix_checks = 0
    exact_boundary_checks = 0
    response_branches = 0
    response_batches = 0
    system_registry = {system.system_id: index for index, system in enumerate(systems)}

    for role in ("calibration", "selection", "validation"):
        panel = panels[role]
        for history in panel.histories:
            system = system_by_id[history.system_id]
            for renderer_name in panel.renderers:
                rendered = render_prefix(system, history, renderer_name)
                prefix_text = adapter._format_prefix(rendered.text)
                prefix_ids = adapter._tokenize(prefix_text)
                event_indices = _formatted_event_token_indices(
                    adapter,
                    prefix_text,
                    rendered.event_end_markers,
                    prefix_ids,
                )
                exact_boundary_checks += len(event_indices)
                with torch.inference_mode():
                    prefix_output = adapter.model(
                        input_ids=prefix_ids,
                        output_hidden_states=True,
                        use_cache=True,
                        return_dict=True,
                    )
                if prefix_output.hidden_states is None or prefix_output.past_key_values is None:
                    raise RuntimeError("SPQ0 checkpoint did not return hidden states and a KV cache")
                selected_layers = [
                    prefix_output.hidden_states[layer][0].float()
                    for layer in adapter.layer_indices
                ]
                final = torch.stack([hidden[-1] for hidden in selected_layers], dim=0)
                boundary = torch.stack(
                    [hidden[event_indices.tolist()] for hidden in selected_layers],
                    dim=0,
                )
                boundary_summary = torch.cat(
                    [boundary.mean(dim=1), boundary.amax(dim=1)],
                    dim=1,
                )
                all_token_summary = torch.stack(
                    [
                        torch.cat([hidden.mean(dim=0), hidden.amax(dim=0)], dim=0)
                        for hidden in selected_layers
                    ],
                    dim=0,
                )
                embedding = prefix_output.hidden_states[0][0].float().mean(dim=0)

                query_ids: list[torch.Tensor] = []
                for test in tests:
                    query = render_probability_query(
                        system,
                        test,
                        bins=config.probability_protocol.bins,
                        candidate_labels=config.probability_protocol.candidate_labels,
                    )
                    query_ids.append(
                        adapter._query_ids(
                            query,
                            world_statement=rendered.text,
                            prefix_ids=prefix_ids,
                        )
                    )
                    exact_prefix_checks += 1
                scores = _score_queries(
                    adapter,
                    prefix_ids,
                    prefix_output.past_key_values,
                    query_ids,
                    batch_size=config.capture.branch_batch_size,
                )
                response_branches += len(scores)
                grouped_lengths: dict[int, int] = defaultdict(int)
                for query in query_ids:
                    grouped_lengths[int(query.shape[1])] += 1
                response_batches += sum(
                    (count + config.capture.branch_batch_size - 1)
                    // config.capture.branch_batch_size
                    for count in grouped_lengths.values()
                )

                final_features.append(final.cpu().numpy())
                event_boundary_features.append(boundary_summary.cpu().numpy())
                all_token_features.append(all_token_summary.cpu().numpy())
                embedding_features.append(embedding.cpu().numpy())
                token_rows.append(
                    prefix_ids[0].detach().cpu().numpy().astype(np.int64, copy=False)
                )
                boundary_rows.append(event_indices)
                history_ids.append(history.history_id)
                role_ids.append(ROLE_IDS[role])
                system_ids.append(system_registry[history.system_id])
                system_role_ids.append(SYSTEM_ROLE_IDS[history.system_role])
                renderer_ids.append(RENDERER_IDS[renderer_name])
                lengths.append(history.length)
                last_observations.append(int(history.observations[-1]))
                observation_frequencies.append(
                    np.bincount(
                        np.asarray(history.observations, dtype=np.int64),
                        minlength=system.n_observations,
                    ).astype(np.float64)
                    / float(history.length)
                )
                semantic_public.append(
                    np.asarray(history.public_probabilities, dtype=np.float64)
                )
                semantic_core.append(
                    np.asarray(history.core_probabilities, dtype=np.float64)
                )
                semantic_targets.append(
                    np.asarray(history.target_probabilities, dtype=np.float64)
                )
                categorical_log_likelihoods.append(
                    np.asarray(
                        [score.log_likelihoods for score in scores],
                        dtype=np.float64,
                    )
                )
                prefix_hashes.append(sha256_bytes(prefix_text.encode("utf-8")))

    token_ids, attention_mask, event_token_indices = _pad_token_rows(
        token_rows,
        boundary_rows,
    )
    serialized_dtype = np.float32
    arrays = {
        "final_token_residual": np.stack(final_features).astype(serialized_dtype),
        "event_boundary_summary": np.stack(event_boundary_features).astype(
            serialized_dtype
        ),
        "all_token_summary": np.stack(all_token_features).astype(serialized_dtype),
        "mean_input_embedding": np.stack(embedding_features).astype(serialized_dtype),
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "event_token_indices": event_token_indices,
        "history_ids": np.asarray(history_ids, dtype=np.int64),
        "role_ids": np.asarray(role_ids, dtype=np.int8),
        "system_ids": np.asarray(system_ids, dtype=np.int8),
        "system_role_ids": np.asarray(system_role_ids, dtype=np.int8),
        "renderer_ids": np.asarray(renderer_ids, dtype=np.int8),
        "lengths": np.asarray(lengths, dtype=np.int16),
        "last_observations": np.asarray(last_observations, dtype=np.int8),
        "observation_frequencies": np.stack(observation_frequencies).astype(np.float64),
        "semantic_public": np.stack(semantic_public).astype(np.float64),
        "semantic_core": np.stack(semantic_core).astype(np.float64),
        "semantic_targets": np.stack(semantic_targets).astype(np.float64),
        "categorical_log_likelihoods": np.stack(categorical_log_likelihoods).astype(
            np.float64
        ),
    }
    capture_path = root / "captures" / f"{model_spec.model_id}.npz"
    capture_sha = _write_npz(capture_path, arrays)
    observed_revision = getattr(adapter.model.config, "_commit_hash", None) or model_spec.revision
    if str(observed_revision) != model_spec.revision:
        raise RuntimeError(
            f"{model_spec.model_id} loaded revision {observed_revision!r}, expected "
            f"{model_spec.revision!r}"
        )
    metadata = {
        "schema": "frank_eq_spq0_capture_v1",
        "model_id": model_spec.model_id,
        "family": model_spec.family,
        "hf_id": model_spec.hf_id,
        "revision_requested": model_spec.revision,
        "revision_observed": str(observed_revision),
        "layer_indices": list(adapter.layer_indices),
        "hidden_width": int(arrays["final_token_residual"].shape[-1]),
        "surface_dimensions": {
            "final_token_residual": int(arrays["final_token_residual"].shape[-1]),
            "event_boundary_residuals": int(arrays["event_boundary_summary"].shape[-1]),
            "all_token_summary": int(arrays["all_token_summary"].shape[-1]),
            "mean_input_embedding": int(arrays["mean_input_embedding"].shape[-1]),
        },
        "rows": int(arrays["history_ids"].size),
        "max_tokens": int(token_ids.shape[1]),
        "maximum_events": int(event_token_indices.shape[1]),
        "public_tests": len(basis.public_tests),
        "core_tests": basis.exact_rank,
        "target_tests": len(basis.target_tests),
        "categorical_bins": len(config.probability_protocol.bins),
        "candidate_metadata": adapter.categorical_candidate_metadata(),
        "chat_turn_shape": config.capture.chat_turn_shape,
        "exact_prefix_continuity_checks": exact_prefix_checks,
        "exact_event_boundary_checks": exact_boundary_checks,
        "response_branches": response_branches,
        "response_batches": response_batches,
        "branch_execution": {
            "literal_kv_reuse": True,
            "exclusive_cache_batching": True,
            "exact_replay_response_branches": 0,
        },
        "capture_surface_contract": list(config.capture.surfaces),
        "selected_kv_surface_enabled": False,
        "prefix_hashes_sha256": sha256_bytes(canonical_json_bytes(prefix_hashes)),
        "array": str(capture_path.relative_to(root)),
        "array_sha256": capture_sha,
    }
    metadata_path = root / "captures" / f"{model_spec.model_id}.json"
    atomic_write_json(metadata_path, metadata)
    telemetry.log(
        {
            "capture": {
                "model": model_spec.model_id,
                "rows": metadata["rows"],
                "response_branches": response_branches,
            }
        }
    )
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "metadata": str(metadata_path.relative_to(root)),
        "metadata_sha256": sha256_file(metadata_path),
        "array": str(capture_path.relative_to(root)),
        "array_sha256": capture_sha,
    }


def load_capture(
    root: Path,
    entry: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    array_path = root / str(entry["array"])
    metadata_path = root / str(entry["metadata"])
    if sha256_file(array_path) != entry["array_sha256"]:
        raise ValueError("SPQ0 capture array hash mismatch")
    if sha256_file(metadata_path) != entry["metadata_sha256"]:
        raise ValueError("SPQ0 capture metadata hash mismatch")
    with np.load(array_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    import json

    return arrays, json.loads(metadata_path.read_text())
