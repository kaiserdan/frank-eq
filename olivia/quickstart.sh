#!/usr/bin/env bash
set -euo pipefail

cd "${FRANK_EQ_PROJECT_ROOT:-$PWD}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if command -v python >/dev/null 2>&1; then
    python_bin="python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
else
    echo "container exposes neither python nor python3" >&2
    exit 2
fi

if [[ "${FRANK_EQ_ALLOW_PIP_INSTALL:-0}" == "1" ]]; then
    runtime_root="${TMPDIR:-/tmp}/frank-eq-runtime-${SLURM_JOB_ID:-local}"
    "$python_bin" -m venv --system-site-packages "$runtime_root"
    source "$runtime_root/bin/activate"
    python_bin="python"
    pip_extra=()
    if [[ -n "${FRANK_EQ_PIP_FIND_LINKS:-}" ]]; then
        pip_extra=(--no-index --find-links "$FRANK_EQ_PIP_FIND_LINKS")
    fi
    "$python_bin" -m pip install -e '.[real]' --no-build-isolation "${pip_extra[@]}"
fi

"$python_bin" - <<'PY'
import importlib.util
missing = [name for name in ("torch", "transformers", "yaml", "numpy") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "container is missing real Frank-EQ dependencies: " + ", ".join(missing)
        + ". Rebuild the image or set FRANK_EQ_ALLOW_PIP_INSTALL=1."
    )
PY

config_arg="${FRANK_EQ_CONFIG:?missing FRANK_EQ_CONFIG}"
stages_arg="${FRANK_EQ_STAGES:-cache,validate,train,eval}"
stages_arg="${stages_arg//+/ ,}"
stages_arg="${stages_arg// /}"

case "$config_arg" in
    configs/predictive_state/*)
        if [[ "$stages_arg" != "audit" ]]; then
            echo "PSR0 requires --stages audit; received: $stages_arg" >&2
            exit 2
        fi
        plan_file="${FRANK_EQ_PSR0_PLAN:-configs/predictive_state/inspected_plan.json}"
        if [[ ! -f "$plan_file" ]]; then
            echo "missing inspected PSR0 plan: $plan_file" >&2
            exit 2
        fi
        plan_sha256="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_sha256"])' "$plan_file")"
        [[ "$plan_sha256" =~ ^[0-9a-f]{64}$ ]] || {
            echo "inspected PSR0 plan has no valid internal SHA-256" >&2
            exit 2
        }
        "$python_bin" scripts/predictive_state_cli.py validate --config "$config_arg"
        "$python_bin" scripts/predictive_state_cli.py run \
          --config "$config_arg" \
          --out "${FRANK_EQ_RUN_ROOT:?missing FRANK_EQ_RUN_ROOT}" \
          --stages "$stages_arg" \
          --plan "$plan_file" \
          --inspected-plan-sha256 "$plan_sha256"
        "$python_bin" scripts/predictive_state_cli.py verify \
          --config "$config_arg" \
          --run "$FRANK_EQ_RUN_ROOT"
        ;;
    configs/stagea_v3/*)
        expected_stages="prepare,founder_fit,freeze,held_onboard,evaluate"
        if [[ "$stages_arg" != "$expected_stages" ]]; then
            echo "Stage-A v3 requires the complete frozen sequence; received: $stages_arg" >&2
            exit 2
        fi
        plan_file="${FRANK_EQ_STAGEA_V3_PLAN:-configs/stagea_v3/inspected_plan.json}"
        if [[ ! -f "$plan_file" ]]; then
            echo "missing inspected Stage-A v3 plan: $plan_file" >&2
            exit 2
        fi
        plan_sha256="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_sha256"])' "$plan_file")"
        [[ "$plan_sha256" =~ ^[0-9a-f]{64}$ ]] || {
            echo "inspected Stage-A v3 plan has no valid internal SHA-256" >&2
            exit 2
        }
        "$python_bin" -m frank_eq.cli validate-stagea-v3-config --config "$config_arg"
        "$python_bin" -m frank_eq.cli run-stagea-v3 \
          --config "$config_arg" \
          --out "${FRANK_EQ_RUN_ROOT:?missing FRANK_EQ_RUN_ROOT}" \
          --stages "$stages_arg" \
          --plan "$plan_file" \
          --inspected-plan-sha256 "$plan_sha256"
        "$python_bin" -m frank_eq.cli verify-stagea-v3 \
          --config "$config_arg" \
          --run "$FRANK_EQ_RUN_ROOT"
        ;;
    configs/rate_compute/*)
        if [[ "$stages_arg" != "audit" ]]; then
            echo "rate--compute configs require --stages audit; received: $stages_arg" >&2
            exit 2
        fi
        "$python_bin" -m frank_eq.cli validate-rate-compute-config --config "$config_arg"
        if [[ -n "${FRANK_EQ_RECOVERY_SOURCE_RUN:-}" ]]; then
            : "${FRANK_EQ_RECOVERY_MANIFEST:?recovery requires a manifest path}"
            : "${FRANK_EQ_RECOVERY_MANIFEST_SHA256:?recovery requires a manifest SHA-256}"
            "$python_bin" -m frank_eq.cli recover-rate-compute-audit \
              --config "$config_arg" \
              --source-run "$FRANK_EQ_RECOVERY_SOURCE_RUN" \
              --recovery-manifest "$FRANK_EQ_RECOVERY_MANIFEST" \
              --recovery-manifest-sha256 "$FRANK_EQ_RECOVERY_MANIFEST_SHA256" \
              --out "${FRANK_EQ_RUN_ROOT:-runs}"
        else
            "$python_bin" -m frank_eq.cli run-rate-compute-audit \
              --config "$config_arg" \
              --out "${FRANK_EQ_RUN_ROOT:-runs}"
        fi
        ;;
    *)
        "$python_bin" -m frank_eq.cli validate-real-config --config "$config_arg"
        "$python_bin" -m frank_eq.cli run-real-stagea \
          --config "$config_arg" \
          --out "${FRANK_EQ_RUN_ROOT:-runs}" \
          --stages "$stages_arg"
        ;;
esac
