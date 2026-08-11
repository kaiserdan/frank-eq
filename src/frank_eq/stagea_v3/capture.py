"""Causally ordered all-token capture and frozen source teachers for Stage-A v3."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from frank_eq.data.hf_backend import HFModelAdapter
from frank_eq.rate_compute.backend import (
    ProtocolScore,
    RateComputeModelAdapter,
    render_deliberation_query,
    render_sequence_query,
)
from frank_eq.rate_compute.config import ResponseProtocolConfig
from frank_eq.rate_compute.logic import ordered_edges, render_basis_query
from frank_eq.real_config import CaptureConfig, RealModelSpec
from frank_eq.utils import canonical_json_bytes, sha256_bytes, sha256_file

from .config import StageAV3Config, StageAV3ModelSpec
from .panel import V3Panel, render_v3_world_prefix


@dataclass(slots=True)
class V3CaptureShard:
    """One model/role/complexity capture shard with padded all-token states."""

    model_id: str
    role: str
    entity_count: int
    layer_indices: tuple[int, ...]
    hidden_width: int
    residuals: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    world_ids: torch.Tensor
    renderer_ids: torch.Tensor
    semantic_targets: torch.Tensor
    behavioral_targets: torch.Tensor
    behavioral_log_odds: torch.Tensor
    operation_targets: torch.Tensor
    operation_targets_hard: torch.Tensor
    direct_probabilities: torch.Tensor
    direct_log_odds: torch.Tensor
    direct_generated_tokens: torch.Tensor
    prefix_metadata: list[dict[str, Any]]
    capture_summary: dict[str, Any]
    schema: str = "frank_eq_stagea_v3_capture_shard_v1"

    @property
    def rows(self) -> int:
        return int(self.residuals.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.residuals.shape[2])

    @property
    def coordinate_count(self) -> int:
        return self.entity_count * (self.entity_count - 1)

    def validate(self) -> None:
        if self.schema != "frank_eq_stagea_v3_capture_shard_v1":
            raise ValueError("unsupported Stage-A v3 capture-shard schema")
        if self.role not in {"train", "validation", "test"}:
            raise ValueError("capture shard has an invalid role")
        if self.entity_count not in {4, 6}:
            raise ValueError("capture shard has an invalid entity count")
        if self.residuals.ndim != 4 or self.residuals.dtype != torch.float32:
            raise ValueError("capture residuals must be float32 [rows,depths,tokens,width]")
        rows, depths, tokens, width = self.residuals.shape
        if depths != len(self.layer_indices) or width != self.hidden_width:
            raise ValueError("capture residual dimensions disagree with shard metadata")
        if self.token_ids.shape != (rows, tokens) or self.token_ids.dtype != torch.long:
            raise ValueError("capture token IDs have the wrong shape or dtype")
        if self.attention_mask.shape != (rows, tokens) or self.attention_mask.dtype != torch.bool:
            raise ValueError("capture attention mask has the wrong shape or dtype")
        if torch.any(self.attention_mask.sum(dim=1) < 1):
            raise ValueError("capture shard contains an empty prefix")
        if torch.any(self.token_ids[~self.attention_mask] != 0):
            raise ValueError("padded capture token IDs must be zero")
        coordinate_shape = (rows, self.coordinate_count)
        if self.semantic_targets.shape != coordinate_shape:
            raise ValueError("semantic target matrix has the wrong shape")
        if self.behavioral_targets.shape != coordinate_shape:
            raise ValueError("behavioral target matrix has the wrong shape")
        if self.behavioral_log_odds.shape != coordinate_shape:
            raise ValueError("behavioral score matrix has the wrong shape")
        if self.world_ids.shape != (rows,) or self.renderer_ids.shape != (rows,):
            raise ValueError("capture row identifiers have the wrong shape")
        if self.operation_targets.ndim != 2 or self.operation_targets.shape[0] != rows:
            raise ValueError("operation target matrix has the wrong shape")
        if self.operation_targets_hard.shape != self.operation_targets.shape:
            raise ValueError("hard operation target matrix has the wrong shape")
        expected_direct = (*self.operation_targets.shape, 3)
        if self.direct_probabilities.shape != expected_direct:
            raise ValueError("direct probability tensor has the wrong shape")
        if self.direct_log_odds.shape != expected_direct:
            raise ValueError("direct score tensor has the wrong shape")
        if self.direct_generated_tokens.shape != expected_direct:
            raise ValueError("direct generated-token tensor has the wrong shape")
        if len(self.prefix_metadata) != rows:
            raise ValueError("capture prefix metadata row count differs")
        for tensor in (
            self.semantic_targets,
            self.behavioral_targets,
            self.behavioral_log_odds,
            self.operation_targets,
            self.direct_probabilities,
            self.direct_log_odds,
        ):
            if not torch.all(torch.isfinite(tensor)):
                raise ValueError("capture shard contains non-finite targets or scores")
        if torch.any((self.semantic_targets < 0) | (self.semantic_targets > 1)):
            raise ValueError("semantic targets lie outside [0,1]")
        if torch.any((self.behavioral_targets <= 0) | (self.behavioral_targets >= 1)):
            raise ValueError("behavioral targets must lie strictly inside (0,1)")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "model_id": self.model_id,
            "role": self.role,
            "entity_count": self.entity_count,
            "layer_indices": list(self.layer_indices),
            "hidden_width": self.hidden_width,
            "residuals": self.residuals,
            "token_ids": self.token_ids,
            "attention_mask": self.attention_mask,
            "world_ids": self.world_ids,
            "renderer_ids": self.renderer_ids,
            "semantic_targets": self.semantic_targets,
            "behavioral_targets": self.behavioral_targets,
            "behavioral_log_odds": self.behavioral_log_odds,
            "operation_targets": self.operation_targets,
            "operation_targets_hard": self.operation_targets_hard,
            "direct_probabilities": self.direct_probabilities,
            "direct_log_odds": self.direct_log_odds,
            "direct_generated_tokens": self.direct_generated_tokens,
            "prefix_metadata": self.prefix_metadata,
            "capture_summary": self.capture_summary,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> V3CaptureShard:
        shard = cls(
            model_id=str(payload["model_id"]),
            role=str(payload["role"]),
            entity_count=int(payload["entity_count"]),
            layer_indices=tuple(int(value) for value in payload["layer_indices"]),
            hidden_width=int(payload["hidden_width"]),
            residuals=payload["residuals"],
            token_ids=payload["token_ids"],
            attention_mask=payload["attention_mask"],
            world_ids=payload["world_ids"],
            renderer_ids=payload["renderer_ids"],
            semantic_targets=payload["semantic_targets"],
            behavioral_targets=payload["behavioral_targets"],
            behavioral_log_odds=payload["behavioral_log_odds"],
            operation_targets=payload["operation_targets"],
            operation_targets_hard=payload["operation_targets_hard"],
            direct_probabilities=payload["direct_probabilities"],
            direct_log_odds=payload["direct_log_odds"],
            direct_generated_tokens=payload["direct_generated_tokens"],
            prefix_metadata=list(payload["prefix_metadata"]),
            capture_summary=dict(payload["capture_summary"]),
            schema=str(payload["schema"]),
        )
        shard.validate()
        return shard


def write_capture_shard(path: str | Path, shard: V3CaptureShard) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(shard.to_payload(), temporary)
    os.replace(temporary, target)
    return sha256_file(target)


def load_capture_shard(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> V3CaptureShard:
    source = Path(path)
    if expected_sha256 is not None and sha256_file(source) != expected_sha256:
        raise ValueError(f"capture shard hash mismatch: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("capture shard payload is not a mapping")
    return V3CaptureShard.from_payload(payload)


def build_capture_config(config: StageAV3Config) -> CaptureConfig:
    capture = config.section("capture")
    return CaptureConfig(
        normalized_depths=list(capture["normalized_depths"]),
        max_length=int(capture["max_length"]),
        dtype=str(capture["dtype"]),
        device=str(capture["device"]),
        branch_mode=str(capture["branch_mode"]),
        allow_exact_replay_fallback=bool(capture["allow_exact_replay_fallback"]),
        answer_token_pairs=[list(pair) for pair in capture["answer_token_pairs"]],
        branch_seed=int(capture["branch_seed"]),
        branch_batch_size=int(capture["branch_batch_size"]),
        local_files_only=bool(capture["local_files_only"]),
        prompt_format=str(capture["prompt_format"]),
        chat_template_kwargs=dict(capture["chat_template_kwargs"]),
        parity_sample_size=0,
    )


def build_response_config(config: StageAV3Config) -> ResponseProtocolConfig:
    teacher = config.section("teacher_protocol")
    return ResponseProtocolConfig(
        candidate_false=str(teacher["candidate_false"]),
        candidate_true=str(teacher["candidate_true"]),
        basis_protocol=str(teacher["basis_protocol"]),
        target_protocols=list(teacher["direct_protocols"]),
        compute_families=list(config.section("panel")["operation_families"]),
        rationale_budget=int(teacher["rationale_budget"]),
        pause_budget=int(teacher["pause_budget"]),
        pause_text=str(teacher["pause_text"]),
        reasoning_instruction=str(teacher["reasoning_instruction"]),
        final_cue=str(teacher["final_cue"]),
        sequence_cue=str(teacher["sequence_cue"]),
        normalize_sequence_log_likelihood=bool(
            teacher["normalize_sequence_log_likelihood"]
        ),
    )


def build_model_spec(spec: StageAV3ModelSpec) -> RealModelSpec:
    return RealModelSpec(
        model_id=spec.model_id,
        hf_id=spec.hf_id,
        role=spec.role,
        revision=spec.revision,
    )


def _semantic_basis_query(
    source: int,
    target: int,
    n_entities: int,
    protocols: ResponseProtocolConfig,
) -> str:
    false_display = protocols.candidate_false.strip()
    true_display = protocols.candidate_true.strip()
    cue = (
        f" Reply with exactly {false_display} for false or {true_display} for true."
        f"{protocols.sequence_cue}"
    )
    return render_basis_query(source, target, n_entities, final_cue=cue)


def _score_query_groups(
    adapter: RateComputeModelAdapter,
    prefix_ids: torch.Tensor,
    prefix_cache: Any,
    queries: list[tuple[str, torch.Tensor]],
    protocols: ResponseProtocolConfig,
    *,
    batch_size: int,
) -> tuple[list[ProtocolScore], dict[str, int]]:
    if batch_size < 1:
        raise ValueError("branch batch size must be positive")
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (mode, query_ids) in enumerate(queries):
        grouped[(mode, int(query_ids.shape[1]))].append(index)
    scores: list[ProtocolScore | None] = [None] * len(queries)
    response_batches = 0
    max_batch = 0
    for (mode, _), indices in grouped.items():
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            query_batch = [queries[index][1] for index in batch_indices]
            if mode == "sequence":
                batch_scores = adapter.score_sequence_batch(
                    prefix_ids,
                    prefix_cache,
                    query_batch,
                    protocols,
                )
            elif mode in {"reason", "pause"}:
                batch_scores = adapter.score_with_compute_batch(
                    prefix_ids,
                    prefix_cache,
                    query_batch,
                    protocols,
                    mode=mode,
                )
            else:
                raise ValueError(f"unsupported Stage-A v3 teacher protocol: {mode}")
            if len(batch_scores) != len(batch_indices):
                raise RuntimeError("teacher response batch returned the wrong number of scores")
            for index, score in zip(batch_indices, batch_scores, strict=True):
                scores[index] = score
            response_batches += 1
            max_batch = max(max_batch, len(batch_indices))
    if any(score is None for score in scores):
        raise RuntimeError("teacher response capture left an unscored branch")
    return (
        [score for score in scores if score is not None],
        {"response_batches": response_batches, "max_observed_batch_size": max_batch},
    )


def _token_offsets(adapter: HFModelAdapter, prefix_text: str, prefix_ids: torch.Tensor) -> list[list[int]]:
    try:
        encoded = adapter.tokenizer(
            prefix_text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
            return_offsets_mapping=True,
        )
    except Exception as error:
        raise RuntimeError("Stage-A v3 requires exact tokenizer offset mappings") from error
    encoded_ids = encoded["input_ids"].to(prefix_ids.device)
    if not torch.equal(encoded_ids, prefix_ids):
        raise RuntimeError("offset-mapping tokenization differs from captured prefix IDs")
    offsets = encoded["offset_mapping"][0].detach().cpu().tolist()
    return [[int(start), int(stop)] for start, stop in offsets]


def capture_panel_shard(
    config: StageAV3Config,
    model_spec: StageAV3ModelSpec,
    panel: V3Panel,
    *,
    adapter: RateComputeModelAdapter | None = None,
) -> V3CaptureShard:
    """Capture one frozen model/panel shard with no operation before prefix state."""

    owned_adapter = adapter is None
    if adapter is None:
        adapter = RateComputeModelAdapter(
            build_model_spec(model_spec),
            build_capture_config(config),
        )
    protocols = build_response_config(config)
    capture = config.section("capture")
    panel_config = config.section("panel")
    renderer_registry = [*panel_config["renderers"]["fit"]]
    if panel.role == "test":
        renderer_registry.extend(panel_config["renderers"]["test_unseen"])
    if len(renderer_registry) != len(set(renderer_registry)):
        raise ValueError("Stage-A v3 renderer registry contains duplicates")

    residual_rows: list[torch.Tensor] = []
    token_rows: list[torch.Tensor] = []
    semantic_rows: list[torch.Tensor] = []
    behavioral_rows: list[torch.Tensor] = []
    behavioral_score_rows: list[torch.Tensor] = []
    operation_rows: list[torch.Tensor] = []
    operation_hard_rows: list[torch.Tensor] = []
    direct_probability_rows: list[torch.Tensor] = []
    direct_score_rows: list[torch.Tensor] = []
    direct_generated_rows: list[torch.Tensor] = []
    world_ids: list[int] = []
    renderer_ids: list[int] = []
    metadata: list[dict[str, Any]] = []
    total_response_batches = 0
    max_observed_batch = 0
    logical_queries = 0

    for world in panel.panel.worlds:
        semantic_target = torch.from_numpy(world.fact_vector()).to(torch.float32)
        operation_target = torch.tensor(
            panel.panel.oracle_signatures[world.world_id], dtype=torch.float32
        )
        operation_hard = (operation_target >= 0.5).to(torch.int8)
        for renderer_name in renderer_registry:
            renderer_id = panel.renderer_names.index(renderer_name)
            world_statement = render_v3_world_prefix(world, renderer_name)
            prefix_text = adapter._format_prefix(world_statement)
            prefix_ids = adapter._tokenize(prefix_text)
            with torch.inference_mode():
                prefix_output = adapter.model(
                    input_ids=prefix_ids,
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True,
                )
            if prefix_output.hidden_states is None:
                raise RuntimeError("checkpoint did not return residual hidden states")
            if prefix_output.past_key_values is None:
                raise RuntimeError("checkpoint did not return a KV cache")
            selected = torch.stack(
                [prefix_output.hidden_states[index][0] for index in adapter.layer_indices],
                dim=0,
            ).detach().float().cpu()
            if selected.shape[1] > int(capture["compiler_max_tokens"]):
                raise RuntimeError(
                    "captured prefix exceeds compiler_max_tokens; truncation is forbidden"
                )
            cpu_ids = prefix_ids[0].detach().to(torch.long).cpu()
            offsets = _token_offsets(adapter, prefix_text, prefix_ids)

            basis_queries: list[tuple[str, torch.Tensor]] = []
            for source, target in ordered_edges(panel.entity_count):
                query = _semantic_basis_query(
                    source,
                    target,
                    panel.entity_count,
                    protocols,
                )
                basis_queries.append(
                    (
                        "sequence",
                        adapter._query_ids(
                            query,
                            world_statement=world_statement,
                            prefix_ids=prefix_ids,
                        ),
                    )
                )
            basis_scores, basis_stats = _score_query_groups(
                adapter,
                prefix_ids,
                prefix_output.past_key_values,
                basis_queries,
                protocols,
                batch_size=int(capture["branch_batch_size"]),
            )

            direct_queries: list[tuple[str, torch.Tensor]] = []
            for operation in panel.panel.operations:
                for protocol in protocols.target_protocols:
                    if protocol == "sequence":
                        query = render_sequence_query(
                            operation.definition,
                            panel.entity_count,
                            protocols,
                        )
                    else:
                        query = render_deliberation_query(
                            operation.definition,
                            panel.entity_count,
                            protocols,
                        )
                    direct_queries.append(
                        (
                            protocol,
                            adapter._query_ids(
                                query,
                                world_statement=world_statement,
                                prefix_ids=prefix_ids,
                            ),
                        )
                    )
            direct_scores, direct_stats = _score_query_groups(
                adapter,
                prefix_ids,
                prefix_output.past_key_values,
                direct_queries,
                protocols,
                batch_size=int(capture["branch_batch_size"]),
            )
            operation_count = len(panel.panel.operations)
            protocol_count = len(protocols.target_protocols)
            direct_probability = torch.tensor(
                [score.probability_true for score in direct_scores], dtype=torch.float32
            ).reshape(operation_count, protocol_count)
            direct_score = torch.tensor(
                [score.log_odds_score for score in direct_scores], dtype=torch.float32
            ).reshape(operation_count, protocol_count)
            direct_generated = torch.tensor(
                [score.generated_token_count for score in direct_scores], dtype=torch.int32
            ).reshape(operation_count, protocol_count)

            residual_rows.append(selected)
            token_rows.append(cpu_ids)
            semantic_rows.append(semantic_target)
            behavioral_rows.append(
                torch.tensor(
                    [score.probability_true for score in basis_scores], dtype=torch.float32
                )
            )
            behavioral_score_rows.append(
                torch.tensor([score.log_odds_score for score in basis_scores], dtype=torch.float32)
            )
            operation_rows.append(operation_target)
            operation_hard_rows.append(operation_hard)
            direct_probability_rows.append(direct_probability)
            direct_score_rows.append(direct_score)
            direct_generated_rows.append(direct_generated)
            world_ids.append(world.world_id)
            renderer_ids.append(renderer_id)

            residual_digest = sha256_bytes(selected.numpy().tobytes())
            token_digest = sha256_bytes(cpu_ids.numpy().tobytes())
            metadata.append(
                {
                    "local_world_id": world.world_id,
                    "public_world_id": panel.public_world_id(world.world_id),
                    "renderer_id": renderer_id,
                    "renderer_name": renderer_name,
                    "prefix_utf8_hex": prefix_text.encode("utf-8").hex(),
                    "prefix_sha256": sha256_bytes(prefix_text.encode("utf-8")),
                    "token_ids_sha256": token_digest,
                    "token_offsets": offsets,
                    "token_count": int(cpu_ids.numel()),
                    "layer_indices": list(adapter.layer_indices),
                    "hidden_width": int(selected.shape[-1]),
                    "residual_sha256": residual_digest,
                    "captured_before_operation": True,
                    "operation_reveal_step": int(cpu_ids.numel()),
                    "direct_generated_text": [
                        [
                            direct_scores[operation_index * protocol_count + protocol_index]
                            .generated_text
                            for protocol_index in range(protocol_count)
                        ]
                        for operation_index in range(operation_count)
                    ],
                }
            )
            logical_queries += len(basis_queries) + len(direct_queries)
            total_response_batches += (
                basis_stats["response_batches"] + direct_stats["response_batches"]
            )
            max_observed_batch = max(
                max_observed_batch,
                basis_stats["max_observed_batch_size"],
                direct_stats["max_observed_batch_size"],
            )
            if len(token_rows) % 8 == 0 or len(token_rows) == len(panel.panel.worlds) * len(
                renderer_registry
            ):
                print(
                    "stagea-v3 capture progress "
                    f"model={model_spec.model_id} role={panel.role} "
                    f"n={panel.entity_count} rows={len(token_rows)}/"
                    f"{len(panel.panel.worlds) * len(renderer_registry)} "
                    f"logical_queries={logical_queries}",
                    flush=True,
                )

    max_tokens = max(int(row.shape[0]) for row in token_rows)
    rows = len(token_rows)
    depths = len(adapter.layer_indices)
    hidden_width = int(residual_rows[0].shape[-1])
    padded_residuals = torch.zeros(
        (rows, depths, max_tokens, hidden_width), dtype=torch.float32
    )
    padded_ids = torch.zeros((rows, max_tokens), dtype=torch.long)
    attention_mask = torch.zeros((rows, max_tokens), dtype=torch.bool)
    for index, (residual, ids) in enumerate(zip(residual_rows, token_rows, strict=True)):
        length = int(ids.numel())
        if residual.shape != (depths, length, hidden_width):
            raise RuntimeError("captured residual/token lengths disagree")
        padded_residuals[index, :, :length] = residual
        padded_ids[index, :length] = ids
        attention_mask[index, :length] = True

    observed_revision = getattr(adapter.model.config, "_commit_hash", None) or model_spec.revision
    if str(observed_revision) != model_spec.revision:
        raise RuntimeError(
            f"observed model revision {observed_revision} differs from {model_spec.revision}"
        )
    summary = {
        "schema": "frank_eq_stagea_v3_capture_summary_v1",
        "model_id": model_spec.model_id,
        "revision_requested": model_spec.revision,
        "revision_observed": str(observed_revision),
        "role": panel.role,
        "entity_count": panel.entity_count,
        "rows": rows,
        "prefix_forwards": rows,
        "logical_post_capture_source_queries": logical_queries,
        "kv_cloned_response_branches": logical_queries,
        "exact_replay_response_branches": 0,
        "exact_prefix_continuity_checks": logical_queries,
        "response_batches": total_response_batches,
        "configured_branch_batch_size": int(capture["branch_batch_size"]),
        "max_observed_batch_size": max_observed_batch,
        "exclusive_cache_batching": True,
        "allow_exact_replay_fallback": False,
        "primary_compiler_post_capture_source_queries": 0,
        "candidate_metadata": adapter.candidate_metadata(protocols),
        "direct_protocol_order": list(protocols.target_protocols),
        "operation_registry_sha256": panel.operation_registry_sha256,
        "prefix_manifest_sha256": sha256_bytes(
            canonical_json_bytes(
                [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "direct_generated_text"
                    }
                    for row in metadata
                ]
            )
        ),
    }
    shard = V3CaptureShard(
        model_id=model_spec.model_id,
        role=panel.role,
        entity_count=panel.entity_count,
        layer_indices=tuple(int(value) for value in adapter.layer_indices),
        hidden_width=hidden_width,
        residuals=padded_residuals,
        token_ids=padded_ids,
        attention_mask=attention_mask,
        world_ids=torch.tensor(world_ids, dtype=torch.long),
        renderer_ids=torch.tensor(renderer_ids, dtype=torch.long),
        semantic_targets=torch.stack(semantic_rows),
        behavioral_targets=torch.stack(behavioral_rows),
        behavioral_log_odds=torch.stack(behavioral_score_rows),
        operation_targets=torch.stack(operation_rows),
        operation_targets_hard=torch.stack(operation_hard_rows),
        direct_probabilities=torch.stack(direct_probability_rows),
        direct_log_odds=torch.stack(direct_score_rows),
        direct_generated_tokens=torch.stack(direct_generated_rows),
        prefix_metadata=metadata,
        capture_summary=summary,
    )
    shard.validate()
    if owned_adapter:
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return shard
