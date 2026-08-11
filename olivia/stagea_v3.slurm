#!/usr/bin/env bash
#SBATCH --job-name=frank-eq-stagea-v3
#SBATCH --account=nn12027k
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

module --quiet purge
module load NRIS/GPU

if [[ -n "${REMOTE_CODE_DIR:-}" && -n "${REMOTE_VERSION_DIR:-}" ]]; then
    project_root="$REMOTE_CODE_DIR"
    version_root="$REMOTE_VERSION_DIR"
    source "$version_root/manifest.env"
    source_sha256="$(tr -d '[:space:]' < "$version_root/source.sha256")"
    [[ "$source_sha256" == "$SOURCE_ARCHIVE_SHA256" ]] || {
        echo "deployed source archive SHA-256 differs from its immutable manifest" >&2
        exit 2
    }
    project_version="${PROJECT_VERSION:?missing immutable project version}"
    job_root="$version_root/outputs/$SLURM_JOB_ID"
    run_root="$job_root/run"
else
    project_root="$(readlink -f "${SLURM_SUBMIT_DIR:-$PWD}")"
    version_root="$project_root"
    source_sha256="${FRANK_EQ_SOURCE_SHA256:?missing source archive SHA-256}"
    project_version="${PROJECT_VERSION:-${FRANK_EQ_JOB_NAME:?missing immutable job name}}"
    job_root="$project_root"
    run_root="${FRANK_EQ_RUN_ROOT:-runs}"
fi
[[ "${GIT_DIRTY:-${FRANK_EQ_GIT_DIRTY:-unknown}}" == "false" ]] || {
    echo "Stage-A v3 refuses a dirty source deployment" >&2
    exit 2
}

mkdir -p "$job_root"
[[ ! -e "$run_root" ]] || {
    echo "Stage-A v3 run root already exists: $run_root" >&2
    exit 2
}

container_image="/cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif"
image_sha256="a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1"
observed_image_sha256="$(sha256sum "$container_image" | awk '{print $1}')"
[[ "$observed_image_sha256" == "$image_sha256" ]] || {
    echo "Stage-A v3 runtime image SHA-256 mismatch" >&2
    exit 2
}

export HF_HOME="/cluster/projects/nn12027k/hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TMPDIR="${SCRATCH:-$job_root/tmp}"
mkdir -p "$TMPDIR"

wandb_env="$HOME/.config/codex-hpc/wandb.env"
if [[ -f "$wandb_env" ]]; then
    [[ "$(stat -c '%a' "$wandb_env")" == "600" ]] || {
        echo "refusing W&B credential file without mode 0600" >&2
        exit 2
    }
    set -a
    source "$wandb_env"
    set +a
fi
export WANDB_DIR="$job_root/wandb"
export WANDB_RUN_NAME="frank-eq-stagea-v3-${project_version}-${SLURM_JOB_ID}"

export FRANK_EQ_CLUSTER=olivia
export FRANK_EQ_CONFIG=configs/stagea_v3/real_olivia_v3.yaml
export FRANK_EQ_STAGES=prepare,founder_fit,freeze,held_onboard,evaluate
export FRANK_EQ_RUN_ROOT="$run_root"
export FRANK_EQ_STAGEA_V3_PLAN=configs/stagea_v3/inspected_plan.json
export FRANK_EQ_SOURCE_SHA256="$source_sha256"
export FRANK_EQ_GIT_COMMIT="${GIT_COMMIT:-${FRANK_EQ_GIT_COMMIT:?missing git commit}}"
export FRANK_EQ_GIT_DIRTY="${GIT_DIRTY:-${FRANK_EQ_GIT_DIRTY:?missing git state}}"
export FRANK_EQ_RUNTIME_IMAGE="$container_image"
export FRANK_EQ_IMAGE_SHA256="$image_sha256"
export FRANK_EQ_ALLOW_PIP_INSTALL=0
export PROJECT_VERSION="$project_version"

for key in PROJECT_VERSION HF_HOME TRANSFORMERS_CACHE TMPDIR WANDB_API_KEY WANDB_ENTITY WANDB_MODE WANDB_DIR WANDB_RUN_NAME WANDB_BASE_URL FRANK_EQ_CLUSTER FRANK_EQ_CONFIG FRANK_EQ_STAGES FRANK_EQ_RUN_ROOT FRANK_EQ_STAGEA_V3_PLAN FRANK_EQ_SOURCE_SHA256 FRANK_EQ_GIT_COMMIT FRANK_EQ_GIT_DIRTY FRANK_EQ_RUNTIME_IMAGE FRANK_EQ_IMAGE_SHA256 FRANK_EQ_ALLOW_PIP_INSTALL; do
    value="${!key:-}"
    if [[ -n "$value" ]]; then
        export "APPTAINERENV_${key}=$value"
    fi
done
export APPTAINERENV_FRANK_EQ_PROJECT_ROOT="$project_root"
export APPTAINERENV_PYTHONPATH="$project_root/src"
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false

bind_paths="$version_root,$HF_HOME,$TMPDIR,/cluster/projects/${SLURM_JOB_ACCOUNT}"
srun --ntasks=1 --gpus-per-node=1 --cpus-per-task=32 \
  apptainer exec --nv --bind "$bind_paths" "$container_image" \
  bash "$project_root/olivia/quickstart.sh"
