#!/usr/bin/env bash
#SBATCH --job-name=frank-eq-stagea
#SBATCH --account=project_465002861
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

module --quiet --force purge
module load LUMI/25.09
module load partition/G
module load rocm/6.4.4
module load lumi-CrayPath
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

PROJECT_ROOT="$(readlink -f "${SLURM_SUBMIT_DIR:-$PWD}")"
cd "$PROJECT_ROOT"
mkdir -p logs "${FRANK_EQ_RUN_ROOT:-runs}"

export HF_HOME="${FRANK_EQ_HF_HOME:-/scratch/project_465002861/kaiserda/frank/hf-cache}"
export TMPDIR="${TMPDIR:-/scratch/project_465002861/kaiserda/frank-eq/tmp/${FRANK_EQ_JOB_NAME:-stagea}-${SLURM_JOB_ID}}"
mkdir -p "$HF_HOME" "$TMPDIR"
CONTAINER_IMAGE="${FRANK_EQ_LUMI_IMAGE:-/scratch/project_465002861/kaiserda/frank/build_env/usae-deps.sif}"

for key in FRANK_EQ_CLUSTER FRANK_EQ_JOB_NAME FRANK_EQ_CONFIG FRANK_EQ_RUN_ROOT FRANK_EQ_PROFILE FRANK_EQ_STAGES FRANK_EQ_SOURCE_SHA256 FRANK_EQ_GIT_COMMIT HF_HOME TMPDIR HUGGING_FACE_HUB_TOKEN HF_TOKEN; do
    value="${!key:-}"
    if [[ -n "$value" ]]; then
        export "SINGULARITYENV_${key}=$value"
    fi
done
export SINGULARITYENV_FRANK_EQ_PROJECT_ROOT="$PROJECT_ROOT"
export SINGULARITYENV_PYTHONPATH="$PROJECT_ROOT/src"
export SINGULARITYENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SINGULARITYENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
unset BASH_ENV || true
unset ENV || true
export SINGULARITYENV_BASH_ENV=/dev/null
export SINGULARITYENV_ENV=""

srun --ntasks=1 --gpus-per-node=1 --cpus-per-task=32 \
  singularity exec --bind "$PROJECT_ROOT,$HF_HOME,$TMPDIR" "$CONTAINER_IMAGE" \
  bash "$PROJECT_ROOT/lumi/quickstart.sh"
