# Olivia runbook

## Local environment

```bash
export FRANK_EQ_OLIVIA_HOST=olivia
export FRANK_EQ_OLIVIA_ROOT=/cluster/home/dakai5365/project/frank-eq
```

Authentication remains outside the repository.

## Current authority

Real Stage-A v1 already has a valid negative outcome. Do not submit a v1
recapture or rerun merely to localize it. The preferred next action is local
analysis of the fetched cache:

```bash
frank-eq diagnose-real-cache \
  --cache .agents/state/olivia/<job>/remote/runs/cache \
  --out runs/diagnostics/<job>
```

The diagnostic uses training/validation worlds only and is non-promotional.

## Runtime environment

The `accel` partition is H200/x86_64. The default runtime container is
`pytorch-2.5.1-cuda12.4-runtime-amd64.sif`. The job may install `.[real]`
extras when explicitly allowed:

```bash
export FRANK_EQ_ALLOW_PIP_INSTALL=1
export FRANK_EQ_PIP_FIND_LINKS=/cluster/projects/nn12027k/frank-eq-wheels
export FRANK_EQ_OLIVIA_IMAGE=/cluster/home/dakai5365/project/frank/pytorch-2.5.1-cuda12.4-runtime-amd64.sif
```

W&B is fail-open telemetry only. Credentials remain in the environment.

## Checkpoint preflight

Real configs use the cluster Hugging Face cache. Future v2 configs must pin
exact checkpoint revisions. Verify all snapshots before submission.

Default:

```text
/cluster/projects/nn12027k/hf-cache
```

## Diagnostic dry run

A new cache plus non-promotional diagnostic can be tested with:

```bash
python olivia/cli.py submit \
  --job-name frank-eq-stagea-diagnostic \
  --config configs/stage0/real_olivia.yaml \
  --profile full \
  --stages cache,validate,diagnose \
  --dry-run --json
```

This is an operational smoke only. It does not authorize reusing the v1 test
role or adopting a v2 architecture.

## Status, fetch, verify

```bash
python olivia/cli.py status --job-name <job> --json
python olivia/cli.py fetch --job-name <job> --json
python olivia/cli.py verify --job-name <job> --json
```

Local state is under:

```text
.agents/state/olivia/<job>/
```

This path is ignored and must never be committed. Adopted evidence is copied
selectively into `evidence/<run>/` with a hash manifest.

## Workflow stages

```text
cache,validate,diagnose,train,eval
```

`diagnose` is optional. Historical `cache,validate,train,eval` stage lists
remain valid. A diagnostic result is not a scientific gate.

## Result interpretation

- Slurm `COMPLETED` plus workflow `completed` means engineering integrity.
- A failing `eval/decision.json` is a valid negative result.
- A diagnostic recommendation has no authorization semantics.
- Missing cache hashes or causal-boundary failures prohibit all downstream work.
- Receiver execution remains locked until a fresh, prospective Stage-A v2 gate
  passes.
