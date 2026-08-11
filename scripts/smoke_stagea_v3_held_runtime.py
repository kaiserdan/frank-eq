#!/usr/bin/env python3
"""Task-blind exact-revision runtime smoke for the Stage-A v3 held model."""

from __future__ import annotations

import argparse
import gc
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from frank_eq.data.hf_backend import clone_past_key_values
from frank_eq.rate_compute.backend import RateComputeModelAdapter
from frank_eq.stagea_v3.capture import build_capture_config, build_model_spec
from frank_eq.stagea_v3.config import load_stagea_v3_config
from frank_eq.utils import atomic_write_json, sha256_bytes


def smoke(config_path: Path, output_path: Path) -> dict[str, object]:
    config = load_stagea_v3_config(config_path)
    held = config.held_model
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("held runtime smoke requires exactly one visible CUDA device")

    started = time.time()
    adapter = RateComputeModelAdapter(
        build_model_spec(held),
        build_capture_config(config),
    )
    neutral_statement = (
        "Engineering-only cache continuity check. No experiment world, graph, "
        "operation, target, candidate answer, or label is present."
    )
    prefix_text = adapter._format_prefix(neutral_statement)
    prefix_ids = adapter._tokenize(prefix_text)
    with torch.inference_mode():
        output = adapter.model(
            input_ids=prefix_ids,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
    if output.hidden_states is None or output.past_key_values is None:
        raise RuntimeError("held model did not return hidden states and a KV cache")
    cloned_cache = clone_past_key_values(output.past_key_values)
    suffix_ids = adapter._query_ids(
        "Runtime continuity marker only.",
        world_statement=neutral_statement,
        prefix_ids=prefix_ids,
    )
    observed_revision = getattr(adapter.model.config, "_commit_hash", None) or held.revision
    if str(observed_revision) != held.revision:
        raise RuntimeError("held model resolved to a different revision")

    result: dict[str, object] = {
        "schema": "frank_eq_stagea_v3_held_runtime_smoke_v1",
        "status": "passed",
        "config_sha256": config.config_sha256,
        "model_id": held.model_id,
        "repository": held.hf_id,
        "revision_requested": held.revision,
        "revision_observed": str(observed_revision),
        "prompt_format": config.section("capture")["prompt_format"],
        "prefix_sha256": sha256_bytes(prefix_text.encode("utf-8")),
        "prefix_tokens": int(prefix_ids.shape[1]),
        "continuity_suffix_tokens": int(suffix_ids.shape[1]),
        "hidden_state_count": len(output.hidden_states),
        "selected_layers": list(adapter.layer_indices),
        "hidden_width": int(output.hidden_states[-1].shape[-1]),
        "kv_cache_type": type(cloned_cache).__name__,
        "registered_worlds_loaded": 0,
        "registered_operations_scored": 0,
        "answers_scored": 0,
        "test_access_count": 0,
        "model_inference_executed": True,
        "inference_scope": "neutral_prefix_only",
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "source_sha256": os.environ.get("FRANK_EQ_SOURCE_SHA256"),
        "runtime_image": os.environ.get("FRANK_EQ_RUNTIME_IMAGE"),
        "runtime_image_sha256": os.environ.get("FRANK_EQ_IMAGE_SHA256"),
        "elapsed_seconds": time.time() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }
    atomic_write_json(output_path, result)
    del output, cloned_cache, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = smoke(args.config, args.out)
    print(
        {
            key: result[key]
            for key in (
                "status",
                "model_id",
                "revision_observed",
                "registered_worlds_loaded",
                "registered_operations_scored",
                "answers_scored",
                "test_access_count",
                "inference_scope",
            )
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
