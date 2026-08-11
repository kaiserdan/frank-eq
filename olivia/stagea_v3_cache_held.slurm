#!/usr/bin/env bash
#SBATCH --job-name=frank-eq-cache-qwen3-14b
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

: "${REMOTE_VERSION_DIR:?missing immutable remote version directory}"

module --quiet purge
module load NRIS/GPU

container_image="/cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif"
image_sha256="a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1"
observed_image_sha256="$(sha256sum "$container_image" | awk '{print $1}')"
[[ "$observed_image_sha256" == "$image_sha256" ]] || {
    echo "held-cache staging image SHA-256 mismatch" >&2
    exit 2
}

export HF_HOME="/cluster/projects/nn12027k/hf-cache"
export http_proxy="http://10.63.2.48:3128/"
export https_proxy="http://10.63.2.48:3128/"
output_root="$REMOTE_VERSION_DIR/outputs/$SLURM_JOB_ID"
mkdir -p "$output_root"

for key in HF_HOME http_proxy https_proxy HF_TOKEN HUGGING_FACE_HUB_TOKEN; do
    value="${!key:-}"
    if [[ -n "$value" ]]; then
        export "APPTAINERENV_${key}=$value"
    fi
done
export APPTAINERENV_OUTPUT_ROOT="$output_root"
export APPTAINERENV_HF_HUB_OFFLINE=0
export APPTAINERENV_TRANSFORMERS_OFFLINE=0

apptainer exec --nv --bind "$HF_HOME,$output_root" "$container_image" python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen3-14B"
revision = "40c069824f4251a91eefaf281ebe4c544efd3e18"
snapshot = Path(
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(Path(os.environ["HF_HOME"]) / "hub"),
        local_files_only=False,
        max_workers=8,
    )
).resolve()
if snapshot.name != revision:
    raise RuntimeError(f"resolved snapshot {snapshot.name} differs from {revision}")

required = ("config.json", "tokenizer.json", "model.safetensors.index.json")
missing = [name for name in required if not (snapshot / name).is_file()]
if missing:
    raise RuntimeError("held snapshot lacks required files: " + ", ".join(missing))

files = []
for path in sorted(item for item in snapshot.iterdir() if item.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    files.append(
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )

receipt = {
    "schema": "frank_eq_stagea_v3_held_cache_receipt_v1",
    "status": "passed",
    "repository": repo_id,
    "revision_requested": revision,
    "revision_resolved": snapshot.name,
    "snapshot": str(snapshot),
    "task_prompts_executed": 0,
    "model_inference_executed": False,
    "files": files,
    "file_count": len(files),
    "total_bytes": sum(row["bytes"] for row in files),
    "python": platform.python_version(),
    "completed_at": datetime.now(UTC).isoformat(),
}
target = Path(os.environ["OUTPUT_ROOT"]) / "held_cache_receipt.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, target)
print(json.dumps({key: receipt[key] for key in (
    "status", "repository", "revision_resolved", "task_prompts_executed",
    "model_inference_executed", "file_count", "total_bytes"
)}, sort_keys=True))
PY
