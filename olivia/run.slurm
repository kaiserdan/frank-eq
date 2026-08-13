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

module --quiet purge
module load NRIS/GPU

PROJECT_ROOT="$(readlink -f "${SLURM_SUBMIT_DIR:-$PWD}")"
cd "$PROJECT_ROOT"
mkdir -p logs "${FRANK_EQ_RUN_ROOT:-runs}"

export HF_HOME="${FRANK_EQ_HF_HOME:-/cluster/projects/${SLURM_JOB_ACCOUNT}/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TMPDIR="${TMPDIR:-${SCRATCH:-/cluster/work/projects/${SLURM_JOB_ACCOUNT}/dakai5365/frank-eq/tmp}/${FRANK_EQ_JOB_NAME:-stagea}-${SLURM_JOB_ID}}"
mkdir -p "$HF_HOME" "$TMPDIR"

CONTAINER_IMAGE="${FRANK_EQ_OLIVIA_IMAGE:-/cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif}"
BIND_PATHS="$PROJECT_ROOT,$HF_HOME,$TMPDIR,/cluster/projects/${SLURM_JOB_ACCOUNT}"
if [[ -n "${FRANK_EQ_RECOVERY_SOURCE_RUN:-}" ]]; then
    BIND_PATHS="$BIND_PATHS,$FRANK_EQ_RECOVERY_SOURCE_RUN"
fi

if [[ "${FRANK_EQ_CONFIG:-}" == configs/rate_compute/* || "${FRANK_EQ_CONFIG:-}" == configs/spq0/* ]]; then
    : "${FRANK_EQ_IMAGE_SHA256:?registered audits require an inspected image SHA-256}"
fi
if [[ -n "${FRANK_EQ_IMAGE_SHA256:-}" ]]; then
    observed_image_sha256="$(sha256sum "$CONTAINER_IMAGE" | awk '{print $1}')"
    if [[ "$observed_image_sha256" != "$FRANK_EQ_IMAGE_SHA256" ]]; then
        echo "runtime image SHA-256 mismatch" >&2
        exit 2
    fi
fi

wandb_env="${FRANK_EQ_WANDB_ENV:-$HOME/.config/codex-hpc/wandb.env}"
if [[ -f "$wandb_env" ]]; then
    if [[ "$(stat -c '%a' "$wandb_env")" != "600" ]]; then
        echo "refusing W&B credential file without mode 0600: $wandb_env" >&2
        exit 2
    fi
    set -a
    source "$wandb_env"
    set +a
fi
export WANDB_DIR="${WANDB_DIR:-${FRANK_EQ_RUN_ROOT:-runs}/wandb}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${FRANK_EQ_JOB_NAME:-frank-eq}-${SLURM_JOB_ID}}"

for key in FRANK_EQ_CLUSTER FRANK_EQ_JOB_NAME FRANK_EQ_CONFIG FRANK_EQ_RUN_ROOT FRANK_EQ_PROFILE FRANK_EQ_STAGES FRANK_EQ_SOURCE_SHA256 FRANK_EQ_GIT_COMMIT FRANK_EQ_GIT_DIRTY FRANK_EQ_IMAGE_SHA256 FRANK_EQ_SPQ0_PLAN FRANK_EQ_RECOVERY_SOURCE_RUN FRANK_EQ_RECOVERY_MANIFEST FRANK_EQ_RECOVERY_MANIFEST_SHA256 HF_HOME TRANSFORMERS_CACHE TMPDIR HUGGING_FACE_HUB_TOKEN HF_TOKEN FRANK_EQ_ALLOW_PIP_INSTALL FRANK_EQ_PIP_FIND_LINKS WANDB_API_KEY WANDB_ENTITY WANDB_MODE WANDB_DIR WANDB_RUN_NAME WANDB_BASE_URL; do
    value="${!key:-}"
    if [[ -n "$value" ]]; then
        export "APPTAINERENV_${key}=$value"
    fi
done
export APPTAINERENV_FRANK_EQ_PROJECT_ROOT="$PROJECT_ROOT"
export APPTAINERENV_FRANK_EQ_RUNTIME_IMAGE="$CONTAINER_IMAGE"
export APPTAINERENV_PYTHONPATH="$PROJECT_ROOT/src"
export APPTAINERENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export APPTAINERENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export APPTAINERENV_TOKENIZERS_PARALLELISM=false

srun --ntasks=1 --gpus-per-node=1 --cpus-per-task=32 \
  apptainer exec --nv --bind "$BIND_PATHS" "$CONTAINER_IMAGE" \
  bash "$PROJECT_ROOT/olivia/quickstart.sh"
