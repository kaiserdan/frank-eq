#!/usr/bin/env bash
set -euo pipefail

cd "${FRANK_EQ_PROJECT_ROOT:-$PWD}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [[ "${FRANK_EQ_ALLOW_PIP_INSTALL:-0}" == "1" ]]; then
    runtime_root="${TMPDIR:-/tmp}/frank-eq-runtime-${SLURM_JOB_ID:-local}"
    python -m venv --system-site-packages "$runtime_root"
    source "$runtime_root/bin/activate"
    python -m pip install -e '.[real]' --no-build-isolation
fi

python - <<'PY'
import importlib.util
missing = [name for name in ("torch", "transformers", "yaml", "numpy") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "container is missing real Stage-A dependencies: " + ", ".join(missing)
        + ". Rebuild the image or set FRANK_EQ_ALLOW_PIP_INSTALL=1."
    )
PY

python -m frank_eq.cli validate-real-config --config "${FRANK_EQ_CONFIG:?missing FRANK_EQ_CONFIG}"
python -m frank_eq.cli run-real-stagea \
  --config "$FRANK_EQ_CONFIG" \
  --out "${FRANK_EQ_RUN_ROOT:-runs}" \
  --stages "${FRANK_EQ_STAGES:-cache,validate,train,eval}"
