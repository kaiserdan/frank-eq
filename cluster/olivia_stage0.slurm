#!/bin/bash
#SBATCH --job-name=frank-eq-stage0
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

CONFIG="${1:-configs/stage0/synthetic_full.yaml}"
RUN_ROOT="${RUN_ROOT:-runs/olivia-stage0-${SLURM_JOB_ID}}"
VENV_ROOT="${TMPDIR:-/tmp}/frank-eq-${SLURM_JOB_ID}"

python -m venv "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -m compileall -q src scripts
pytest -q
python scripts/validate_repo.py
frank-eq run-stage0 --config "$CONFIG" --out "$RUN_ROOT"
