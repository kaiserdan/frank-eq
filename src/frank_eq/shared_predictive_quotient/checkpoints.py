"""Model-free checkpoint hashing and reserved-checkpoint non-access receipts."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from frank_eq.utils import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file

from .config import SPQRunConfig


def preflight_active_tokenizer_contract(
    config: SPQRunConfig,
    systems: tuple[Any, ...],
    basis: Any,
    panels: dict[str, Any],
    checkpoint_receipt: dict[str, Any],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify active chat templates and candidate tokens before model construction."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("SPQ0 tokenizer preflight requires Transformers") from error

    from .capture import SPQModelAdapter, _formatted_event_token_indices
    from .panel import render_prefix, render_probability_query

    system_by_id = {system.system_id: system for system in systems}
    tests = (*basis.public_tests, *basis.target_tests)
    active: dict[str, Any] = {}
    for model in config.models:
        checkpoint = checkpoint_receipt["active"][model.model_id]
        snapshot = Path(checkpoint["snapshot"])
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            trust_remote_code=model.trust_remote_code,
            local_files_only=True,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError(f"{model.model_id} requires a fast tokenizer for event offsets")
        adapter = object.__new__(SPQModelAdapter)
        adapter.spec = SimpleNamespace(model_id=model.model_id)
        adapter.spq_model = model
        adapter.capture = config.capture
        adapter.tokenizer = tokenizer
        adapter.device = "cpu"

        prefix_checks = 0
        branch_checks = 0
        boundary_checks = 0
        maximum_prefix_tokens = 0
        maximum_total_tokens = 0
        strata: list[dict[str, Any]] = []
        for role in ("calibration", "selection", "validation"):
            panel = panels[role]
            for system_id in panel.system_ids:
                system = system_by_id[system_id]
                for length in panel.lengths:
                    history = next(
                        row
                        for row in panel.histories
                        if row.system_id == system_id and row.length == length
                    )
                    for renderer in panel.renderers:
                        rendered = render_prefix(system, history, renderer)
                        prefix_text = adapter._format_prefix(rendered.text)
                        prefix_ids = adapter._tokenize(prefix_text)
                        boundaries = _formatted_event_token_indices(
                            adapter,
                            prefix_text,
                            rendered.event_end_markers,
                            prefix_ids,
                        )
                        prefix_checks += 1
                        boundary_checks += len(boundaries)
                        maximum_prefix_tokens = max(
                            maximum_prefix_tokens,
                            int(prefix_ids.shape[1]),
                        )
                        for test in tests:
                            query = render_probability_query(
                                system,
                                test,
                                bins=config.probability_protocol.bins,
                                candidate_labels=(
                                    config.probability_protocol.candidate_labels
                                ),
                            )
                            suffix = adapter._query_ids(
                                query,
                                world_statement=rendered.text,
                                prefix_ids=prefix_ids,
                            )
                            branch_checks += 1
                            maximum_total_tokens = max(
                                maximum_total_tokens,
                                int(prefix_ids.shape[1] + suffix.shape[1]),
                            )
                        strata.append(
                            {
                                "role": role,
                                "system_id": system_id,
                                "length": length,
                                "renderer": renderer,
                            }
                        )
        candidate_ids = [
            tokenizer.encode(label, add_special_tokens=False)
            for label in config.probability_protocol.candidate_labels
        ]
        if any(not values for values in candidate_ids) or len(
            {tuple(values) for values in candidate_ids}
        ) != len(candidate_ids):
            raise RuntimeError(f"{model.model_id} categorical candidates are empty or collide")
        if maximum_total_tokens > config.capture.max_length:
            raise RuntimeError(f"{model.model_id} tokenizer preflight exceeds max_length")
        active[model.model_id] = {
            "hf_id": model.hf_id,
            "revision": model.revision,
            "snapshot_content_sha256": checkpoint["snapshot_content_sha256"],
            "tokenizer_class": type(tokenizer).__name__,
            "fast_tokenizer": True,
            "chat_turn_shape": config.capture.chat_turn_shape,
            "prefix_strata_checked": prefix_checks,
            "future_test_branches_checked": branch_checks,
            "event_boundaries_checked": boundary_checks,
            "maximum_prefix_tokens": maximum_prefix_tokens,
            "maximum_prefix_plus_query_tokens": maximum_total_tokens,
            "candidate_token_ids": candidate_ids,
            "candidate_token_counts": [len(values) for values in candidate_ids],
            "candidate_ids_unique": True,
            "strata_sha256": sha256_bytes(canonical_json_bytes(strata)),
            "model_loaded": False,
            "inference_executed": False,
        }
    receipt = {
        "schema": "frank_eq_spq0_tokenizer_preflight_v1",
        "status": "passed",
        "active": active,
        "active_tokenizer_contract_sha256": sha256_bytes(canonical_json_bytes(active)),
        "reserved_snapshot_resolution_attempts": 0,
        "reserved_files_opened": 0,
        "reserved_tokenizer_loads": 0,
        "reserved_model_loads": 0,
        "reserved_inference_calls": 0,
    }
    if output_path is not None:
        atomic_write_json(output_path, receipt)
    return receipt


def preflight_checkpoint_contract(
    config: SPQRunConfig,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Resolve and hash active snapshots without importing or loading model weights."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("SPQ0 checkpoint preflight requires huggingface-hub") from error

    hf_home = Path(os.environ.get("HF_HOME", "")).expanduser()
    if not str(hf_home) or str(hf_home) == ".":
        raise RuntimeError("SPQ0 checkpoint preflight requires an explicit HF_HOME")
    cache_dir = hf_home / "hub"
    active: dict[str, Any] = {}
    for model in config.models:
        snapshot = Path(
            snapshot_download(
                repo_id=model.hf_id,
                revision=model.revision,
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
        ).resolve()
        if snapshot.name != model.revision:
            raise RuntimeError(
                f"{model.model_id} resolved snapshot {snapshot.name!r}, expected "
                f"{model.revision!r}"
            )
        files: dict[str, dict[str, Any]] = {}
        for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
            relative = path.relative_to(snapshot).as_posix()
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        required = {"config.json"}
        if not required <= set(files) or not any(
            name.endswith((".safetensors", ".bin")) for name in files
        ):
            raise RuntimeError(f"{model.model_id} snapshot lacks config or weight files")
        active[model.model_id] = {
            "hf_id": model.hf_id,
            "family": model.family,
            "revision_requested": model.revision,
            "revision_resolved": snapshot.name,
            "snapshot": str(snapshot),
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(row["bytes"] for row in files.values()),
            "snapshot_content_sha256": sha256_bytes(canonical_json_bytes(files)),
            "model_loaded": False,
            "inference_executed": False,
        }
    reserved = [
        {
            "model_id": model.model_id,
            "hf_id": model.hf_id,
            "family": model.family,
            "revision": model.revision,
            "access": "reserved_unopened",
            "snapshot_resolution_attempted": False,
            "files_opened": 0,
            "model_adapter_instantiated": False,
            "model_loaded": False,
            "inference_executed": False,
        }
        for model in config.reserved_unopened_models
    ]
    receipt = {
        "schema": "frank_eq_spq0_checkpoint_preflight_v1",
        "status": "passed",
        "local_files_only": True,
        "active": active,
        "reserved_unopened": reserved,
        "active_snapshot_content_sha256": sha256_bytes(canonical_json_bytes(active)),
        "reserved_snapshot_resolution_attempts": 0,
        "reserved_files_opened": 0,
        "reserved_model_adapter_instantiations": 0,
        "reserved_model_loads": 0,
        "reserved_inference_calls": 0,
    }
    atomic_write_json(output_path, receipt)
    return receipt
