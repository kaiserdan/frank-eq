#!/usr/bin/env bash
#SBATCH --job-name=frank-eq-stagea
#SBATCH --account=nn12027k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

PROJECT_ROOT="$(readlink -f "${SLURM_SUBMIT_DIR:-$PWD}")"
cd "$PROJECT_ROOT"
mkdir -p logs "${FRANK_EQ_RUN_ROOT:-runs}"

export HF_HOME="${FRANK_EQ_HF_HOME:-/cluster/projects/${SLURM_JOB_ACCOUNT}/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TMPDIR="${TMPDIR:-/tmp/frank-eq-${SLURM_JOB_ID}}"
mkdir -p "$HF_HOME" "$TMPDIR"

CONTAINER_IMAGE="${FRANK_EQ_OLIVIA_IMAGE:-/cluster/home/dakai5365/project/frank/scratch_pytorch_gcc_updated.sif}"
BIND_PATHS="$PROJECT_ROOT,$HF_HOME,/cluster/projects/${SLURM_JOB_ACCOUNT}"

for key in FRANK_EQ_CLUSTER FRANK_EQ_JOB_NAME FRANK_EQ_CONFIG FRANK_EQ_RUN_ROOT FRANK_EQ_PROFILE FRANK_EQ_STAGES FRANK_EQ_SOURCE_SHA256 FRANK_EQ_GIT_COMMIT HF_HOME TRANSFORMERS_CACHE TMPDIR HUGGING_FACE_HUB_TOKEN HF_TOKEN; do
    value="${!key:-}"
    if [[ -n "$value" ]]; then
        export "APPTAINERENV_${key}=$value"
    fi
done
export APPTAINERENV_FRANK_EQ_PROJECT_ROOT="$PROJECT_ROOT"
export APPTAINERENV_PYTHONPATH="$PROJECT_ROOT/src"
export APPTAINERENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export APPTAINERENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export APPTAINERENV_TOKENIZERS_PARALLELISM=false

srun --ntasks=1 --gpus-per-node=1 --cpus-per-task=32 \
  apptainer exec --nv --writable-tmpfs --bind "$BIND_PATHS" "$CONTAINER_IMAGE" \
  bash "$PROJECT_ROOT/olivia/quickstart.sh"
