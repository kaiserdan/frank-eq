# Olivia runbook

## Current authority

Stage R / RC0 completed on Olivia and is adopted as a development pass:

```text
capture:  frank-eq-rc0-rate-compute-olivia-20260811c  Slurm 1874736
recovery: frank-eq-rc0-rate-compute-olivia-20260811d-recovery  Slurm 1891471
result:   PUBLIC_BASIS_COMPOSITION_SUPPORTED
```

Stage-A v3-2 is now frozen and the user authorized one sequential representation
run after separate registration and implementation commits, full validation,
held-checkpoint staging, and an inspected content-addressed dry run. See
`docs/20_STAGEA_V3_PROTOCOL.md`.

This does not authorize an RC0 rerun, receiver execution, receiver-world access,
or a paper claim. V3 test panels must not be created before founder and held
compiler freeze manifests.

## Runtime contract

Olivia `accel` nodes are ARM64 NVIDIA Grace Hopper nodes. The inspected runtime
for RC0 is:

```text
image:  /cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif
sha256: a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1
cache:  /cluster/projects/nn12027k/hf-cache
work:   /cluster/work/projects/nn12027k/dakai5365/frank-eq
```

It exposes `python3`, CUDA-enabled PyTorch, Transformers, NumPy, PyYAML, and
W&B on `aarch64`. Do not use the historical AMD64 image or the existing CPython
x86_64 wheelhouse, and do not enable runtime package installation for RC0.

W&B remains fail-open telemetry. Its credential is sourced from
`$HOME/.config/codex-hpc/wandb.env` only when that file has mode `0600`; the
credential itself must never enter the repository, source archive, or Slurm
submission record.

## Checkpoint preflight

Both exact revisions must already be present in the shared offline cache:

```text
Qwen/Qwen3-4B  1cfa9a7208912126459214e8b04321603b3df60c
Qwen/Qwen3-8B  b968826d9c46dd6066d109eabc6255188de91218
```

No unpinned head or network fallback is allowed.

## Immutable development smoke

After local validation, create a fresh immutable deployment with the generic
Olivia runner. Record the returned version and archive SHA-256:

```bash
~/.codex/skills/run-on-olivia/scripts/deploy_version.sh \
  --project frank-eq \
  --account nn12027k \
  --source "$PWD"
```

Then run the engineering-only `devel` smoke. It loads both frozen models and
exercises exact chat-prefix continuity, cloned-KV semantic sequence scoring,
32 generated reasoning tokens, and 32 matched pause tokens. It does not open an
RC0 panel or produce a scientific result.

```bash
~/.codex/skills/run-on-olivia/scripts/submit.sh \
  --project frank-eq \
  --account nn12027k \
  --version <immutable-version> \
  --job-file olivia/rc0_runtime_smoke.slurm \
  -- --qos=devel
```

Do not launch RC0 unless the smoke job is `COMPLETED` and its `smoke.json`
reports `status: passed`, the exact two revisions, one visible CUDA device, and
matching source/config/image provenance.

## RC0 dry run and submission

Keep the local source unchanged between the inspected dry run and submission:

```bash
export FRANK_EQ_OLIVIA_HOST=olivia
export FRANK_EQ_OLIVIA_ROOT=/cluster/work/projects/nn12027k/dakai5365/frank-eq
export FRANK_EQ_OLIVIA_IMAGE=/cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif
export FRANK_EQ_IMAGE_SHA256=a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1
export FRANK_EQ_HF_HOME=/cluster/projects/nn12027k/hf-cache

python olivia/cli.py submit \
  --job-name <fresh-job-name> \
  --config configs/rate_compute/real_olivia_rc0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

Audit the content-addressed source archive, then remove only `--dry-run` and
submit the same frozen plan.

If the exact-source `devel` smoke is queued behind the per-user QoS cap, the
full job may itself be queued with a recorded fail-closed dependency:

```bash
export FRANK_EQ_SLURM_DEPENDENCY=afterok:<smoke-job-id>
```

Only this single numeric `afterok` form is accepted. Slurm must report the
dependency as unfulfilled until the smoke completes successfully; a failed or
canceled smoke prevents the audit from starting.

## Status, fetch, and verify

```bash
python olivia/cli.py status --job-name <job> --json
python olivia/cli.py fetch  --job-name <job> --json
python olivia/cli.py verify --job-name <job> --json
python scripts/verify_rate_compute_run.py \
  --run .agents/state/olivia/<job>/remote/runs/<run-directory>
```

Local launcher state remains under `.agents/state/olivia/<job>/` and must never
be committed. Adopt only a compact, hash-verified evidence package after the
machine decision and response strata have been independently audited.

If an audit fails only after writing the complete raw and calibrated capture,
and no compiled prediction, metric, or decision exists, preserve that job and use
the fail-closed artifact-only recovery procedure in
`docs/19_STAGE_R_CLUSTER_RUNBOOK.md`. The recovery must use a fresh job name, the
same frozen config, exact `--stages audit`, and `--recover-from-job <failed-job>`.
It must not load either model or overwrite the failed run.

Slurm `COMPLETED` means engineering completion, not scientific success. A
well-formed negative machine decision is a valid RC0 result.
