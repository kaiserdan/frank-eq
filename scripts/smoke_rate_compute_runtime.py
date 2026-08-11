#!/usr/bin/env python3
"""Exercise the production RC0 model/KV paths without running an experiment panel."""

from __future__ import annotations

import argparse
import gc
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from frank_eq.rate_compute import load_rate_compute_config
from frank_eq.rate_compute.backend import RateComputeModelAdapter
from frank_eq.rate_compute.records import _basis_query
from frank_eq.rate_compute.workflow import _timestamp as workflow_timestamp
from frank_eq.utils import atomic_write_json, sha256_file


def _finite_probability(value: float, *, label: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise RuntimeError(f"{label} returned invalid probability {value!r}")
    return value


def smoke_runtime(config_path: Path, output_path: Path) -> dict[str, Any]:
    """Load every frozen model and exercise prefix, cloned-KV, and compute protocols."""

    config = load_rate_compute_config(config_path)
    if config.capture.prompt_format != "chat_turn":
        raise RuntimeError("RC0 runtime smoke requires prompt_format=chat_turn")
    if config.capture.branch_mode != "kv_reuse":
        raise RuntimeError("RC0 runtime smoke requires branch_mode=kv_reuse")
    if config.capture.allow_exact_replay_fallback:
        raise RuntimeError("RC0 runtime smoke forbids replay fallback")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("RC0 runtime smoke requires exactly one visible CUDA accelerator")

    world_statement = (
        "There are four entities: A, B, C, and D. The directed graph is closed-world. "
        "A points to B. B points to C. No other directed edges are present."
    )
    models: list[dict[str, Any]] = []
    started = time.time()
    for spec in config.models:
        model_started = time.time()
        adapter = RateComputeModelAdapter(spec, config.capture)
        prefix_text = adapter._format_prefix(world_statement)
        prefix_ids = adapter._tokenize(prefix_text)
        with torch.inference_mode():
            prefix_output = adapter.model(
                input_ids=prefix_ids,
                use_cache=True,
                return_dict=True,
            )
        if prefix_output.past_key_values is None:
            raise RuntimeError(f"{spec.model_id} did not return a KV cache")

        query = _basis_query(0, 1, 4, config)
        query_ids = adapter._query_ids(
            query,
            world_statement=world_statement,
            prefix_ids=prefix_ids,
        )
        batch_size = int(config.capture.branch_batch_size)
        query_batch = [query_ids.clone() for _ in range(batch_size)]

        scalar_started = time.perf_counter()
        sequence = adapter.score_sequence(
            prefix_ids,
            prefix_output.past_key_values,
            query_ids,
            config.protocols,
        )
        reason = adapter.score_with_compute(
            prefix_ids,
            prefix_output.past_key_values,
            query_ids,
            config.protocols,
            mode="reason",
        )
        pause = adapter.score_with_compute(
            prefix_ids,
            prefix_output.past_key_values,
            query_ids,
            config.protocols,
            mode="pause",
        )
        scalar_seconds = time.perf_counter() - scalar_started

        batch_started = time.perf_counter()
        batch_sequence = adapter.score_sequence_batch(
            prefix_ids,
            prefix_output.past_key_values,
            query_batch,
            config.protocols,
        )
        batch_reason = adapter.score_with_compute_batch(
            prefix_ids,
            prefix_output.past_key_values,
            query_batch,
            config.protocols,
            mode="reason",
        )
        batch_pause = adapter.score_with_compute_batch(
            prefix_ids,
            prefix_output.past_key_values,
            query_batch,
            config.protocols,
            mode="pause",
        )
        batch_seconds = time.perf_counter() - batch_started
        if reason.generated_token_count != config.protocols.rationale_budget:
            raise RuntimeError(
                f"{spec.model_id} generated {reason.generated_token_count} reasoning tokens; "
                f"expected {config.protocols.rationale_budget}"
            )
        if pause.generated_token_count != config.protocols.pause_budget:
            raise RuntimeError(
                f"{spec.model_id} consumed {pause.generated_token_count} pause tokens; "
                f"expected {config.protocols.pause_budget}"
            )
        for protocol, batch_scores, expected_tokens in (
            ("sequence", batch_sequence, 0),
            ("reason", batch_reason, config.protocols.rationale_budget),
            ("pause", batch_pause, config.protocols.pause_budget),
        ):
            if len(batch_scores) != batch_size:
                raise RuntimeError(
                    f"{spec.model_id}/{protocol} returned {len(batch_scores)} batch rows; "
                    f"expected {batch_size}"
                )
            for index, score in enumerate(batch_scores):
                _finite_probability(
                    score.probability_true,
                    label=f"{spec.model_id}/{protocol}/batch/{index}",
                )
                if score.generated_token_count != expected_tokens:
                    raise RuntimeError(
                        f"{spec.model_id}/{protocol}/batch/{index} used "
                        f"{score.generated_token_count} generated tokens; "
                        f"expected {expected_tokens}"
                    )

        scalar_by_protocol = {
            "sequence": sequence,
            "reason": reason,
            "pause": pause,
        }
        batch_by_protocol = {
            "sequence": batch_sequence,
            "reason": batch_reason,
            "pause": batch_pause,
        }
        parity = {
            protocol: {
                "probability_abs_diff": abs(
                    scalar_by_protocol[protocol].probability_true
                    - batch_by_protocol[protocol][0].probability_true
                ),
                "generated_text_exact": (
                    scalar_by_protocol[protocol].generated_text
                    == batch_by_protocol[protocol][0].generated_text
                ),
            }
            for protocol in scalar_by_protocol
        }
        if any(item["probability_abs_diff"] > 0.02 for item in parity.values()):
            raise RuntimeError(f"{spec.model_id} scalar/batch probability parity exceeded 0.02")
        if not all(item["generated_text_exact"] for item in parity.values()):
            raise RuntimeError(f"{spec.model_id} scalar/batch generated text differs")
        observed_revision = getattr(adapter.model.config, "_commit_hash", None)
        if observed_revision and observed_revision != spec.revision:
            raise RuntimeError(
                f"{spec.model_id} revision mismatch: {observed_revision} != {spec.revision}"
            )
        models.append(
            {
                "model_id": spec.model_id,
                "hf_id": spec.hf_id,
                "revision_requested": spec.revision,
                "revision_observed": observed_revision or spec.revision,
                "candidate_metadata": adapter.candidate_metadata(config.protocols),
                "prefix_tokens": int(prefix_ids.shape[1]),
                "query_tokens": int(query_ids.shape[1]),
                "batch_execution": {
                    "configured_size": batch_size,
                    "observed_size": len(batch_sequence),
                    "scalar_seconds": scalar_seconds,
                    "batch_seconds": batch_seconds,
                    "effective_speedup": scalar_seconds / (batch_seconds / float(batch_size)),
                    "scalar_batch_parity": parity,
                },
                "probabilities": {
                    "sequence": _finite_probability(
                        sequence.probability_true, label=f"{spec.model_id}/sequence"
                    ),
                    "reason": _finite_probability(
                        reason.probability_true, label=f"{spec.model_id}/reason"
                    ),
                    "pause": _finite_probability(
                        pause.probability_true, label=f"{spec.model_id}/pause"
                    ),
                },
                "generated_tokens": {
                    "reason": reason.generated_token_count,
                    "pause": pause.generated_token_count,
                },
                "elapsed_seconds": time.time() - model_started,
            }
        )
        del prefix_output, query_batch, batch_sequence, batch_reason, batch_pause, adapter
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "schema": "frank_eq_rate_compute_runtime_smoke_v1",
        "status": "passed",
        "engineering_only": True,
        "scientific_result": False,
        "workflow_timestamp": workflow_timestamp(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "deployment": {
            "project_version": os.environ.get("PROJECT_VERSION"),
            "source_sha256": os.environ.get("FRANK_EQ_SOURCE_SHA256"),
            "runtime_image": os.environ.get("FRANK_EQ_RUNTIME_IMAGE"),
            "runtime_image_sha256": os.environ.get("FRANK_EQ_IMAGE_SHA256"),
        },
        "environment": {
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "accelerator": torch.cuda.get_device_name(0),
        },
        "models": models,
        "elapsed_seconds": time.time() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = smoke_runtime(args.config, args.out)
    print(f"runtime smoke {payload['status']}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
