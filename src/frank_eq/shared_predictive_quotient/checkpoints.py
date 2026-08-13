"""Model-free checkpoint hashing and reserved-checkpoint non-access receipts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from frank_eq.utils import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file

from .config import SPQRunConfig


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
