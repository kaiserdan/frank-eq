#!/bin/bash -l
#SBATCH --job-name=frank-eq-rc0-runtime-smoke
#SBATCH --account=nn12027k
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:20:00

set -euo pipefail

: "${REMOTE_CODE_DIR:?submit through run-on-olivia/scripts/submit.sh}"
: "${REMOTE_VERSION_DIR:?submit through run-on-olivia/scripts/submit.sh}"
: "${PROJECT_VERSION:?missing immutable source version}"

module --quiet purge
module load NRIS/GPU

SIF_IMAGE="${SIF_IMAGE:-/cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif}"
EXPECTED_IMAGE_SHA256="${FRANK_EQ_IMAGE_SHA256:-a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1}"
HF_HOME="${FRANK_EQ_HF_HOME:-/cluster/projects/nn12027k/hf-cache}"
RUN_DIR="$REMOTE_VERSION_DIR/outputs/$SLURM_JOB_ID"
TMPDIR="${SCRATCH:-$RUN_DIR/tmp}"
mkdir -p "$RUN_DIR" "$TMPDIR"

observed_image_sha256="$(sha256sum "$SIF_IMAGE" | awk '{print $1}')"
if [[ "$observed_image_sha256" != "$EXPECTED_IMAGE_SHA256" ]]; then
  echo "runtime image SHA-256 mismatch" >&2
  exit 2
fi
source_sha256="$(<"$REMOTE_VERSION_DIR/source.sha256")"

export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_TRANSFORMERS_CACHE="$HF_HOME"
export APPTAINERENV_TMPDIR="$TMPDIR"
export APPTAINERENV_PYTHONPATH="$REMOTE_CODE_DIR/src"
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_PROJECT_VERSION="$PROJECT_VERSION"
export APPTAINERENV_FRANK_EQ_SOURCE_SHA256="$source_sha256"
export APPTAINERENV_FRANK_EQ_RUNTIME_IMAGE="$SIF_IMAGE"
export APPTAINERENV_FRANK_EQ_IMAGE_SHA256="$observed_image_sha256"

cd "$REMOTE_CODE_DIR"
srun apptainer exec --nv \
  --bind "$REMOTE_CODE_DIR,$RUN_DIR,$HF_HOME,$TMPDIR" \
  "$SIF_IMAGE" \
  python3 scripts/smoke_rate_compute_runtime.py \
    --config configs/rate_compute/real_olivia_rc0.yaml \
    --out "$RUN_DIR/smoke.json"

sha256sum "$RUN_DIR/smoke.json"
