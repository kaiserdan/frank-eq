#!/usr/bin/env bash
#SBATCH --job-name=frank-eq-stagea-v3-held-smoke
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00

set -euo pipefail

: "${REMOTE_CODE_DIR:?submit through the immutable generic Olivia runner}"
: "${REMOTE_VERSION_DIR:?missing immutable remote version directory}"

module --quiet purge
module load NRIS/GPU

image="/cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif"
image_sha256="a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1"
observed_image_sha256="$(sha256sum "$image" | awk '{print $1}')"
[[ "$observed_image_sha256" == "$image_sha256" ]] || {
    echo "held runtime smoke image SHA-256 mismatch" >&2
    exit 2
}

hf_home="/cluster/projects/nn12027k/hf-cache"
run_root="$REMOTE_VERSION_DIR/outputs/$SLURM_JOB_ID"
tmp_root="${SCRATCH:-$run_root/tmp}"
mkdir -p "$run_root" "$tmp_root"
source_sha256="$(tr -d '[:space:]' < "$REMOTE_VERSION_DIR/source.sha256")"

export APPTAINERENV_HF_HOME="$hf_home"
export APPTAINERENV_TRANSFORMERS_CACHE="$hf_home"
export APPTAINERENV_TMPDIR="$tmp_root"
export APPTAINERENV_PYTHONPATH="$REMOTE_CODE_DIR/src"
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_FRANK_EQ_SOURCE_SHA256="$source_sha256"
export APPTAINERENV_FRANK_EQ_RUNTIME_IMAGE="$image"
export APPTAINERENV_FRANK_EQ_IMAGE_SHA256="$image_sha256"

cd "$REMOTE_CODE_DIR"
srun --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 \
  apptainer exec --nv \
  --bind "$REMOTE_CODE_DIR,$run_root,$hf_home,$tmp_root" \
  "$image" \
  python3 scripts/smoke_stagea_v3_held_runtime.py \
    --config configs/stagea_v3/real_olivia_v3.yaml \
    --out "$run_root/held_runtime_smoke.json"

sha256sum "$run_root/held_runtime_smoke.json"
