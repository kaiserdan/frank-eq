# LUMI runbook

## Local environment

```bash
export FRANK_EQ_LUMI_HOST=lumi
export FRANK_EQ_LUMI_ROOT=/scratch/project_465002861/kaiserda/frank-eq
```

The default Slurm surface uses one MI250X GCD. Operational resource changes do
not change the scientific contract.

## Current authority

There is no authorized LUMI execution. LUMI jobs `20942127` and `20952565`
produced the adopted Stage-A v1 and v2 negatives. RC0 and Stage-A v3-2 later
completed on Olivia; v3-2's terminal decision is
`ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED`. Do not rerun any of these registrations,
launch receiver work, or treat a cluster move as fresh scientific authority.
Stage M0 subsequently completed on Olivia with
`OPERATION_CLOSED_EVENTS_NOT_READABLE`; it cannot be rerun or moved to LUMI.
There is no Stage M0 LUMI port and no current LUMI execution authority.

The historical v1 fetched cache can still be analyzed without model execution
using:

```bash
frank-eq diagnose-real-cache \
  --cache .agents/state/lumi/frank-eq-stagea-devg-v2/remote/runs/cache \
  --out runs/diagnostics/frank-eq-stagea-devg-v2
```

This uses train/validation worlds only and authorizes nothing.

## Stage-A v2-1 (falsified; adopted negative)

Protocol `docs/14_STAGEA_V2_PROTOCOL.md`, config `configs/stage0/real_lumi_v2.yaml`.
Chat-templated capture, world seed 20260810, revision pins, parity audit,
native-competence gate. Ran on dev-g (`frank-eq-stagea-lumi-v2`, Slurm
20952565): workflow completed, decision `STOP_OR_REVISE_STAGE0` — native
competence −0.0521, prompt surface not the bottleneck. Evidence:
`evidence/real_stagea_lumi_v2/`. Engineering amendments during the run:
parity tolerance calibrated from a 32-branch measurement
(`parity_max_abs_diff: 0.33` = max(0.05, 3× max measured)), and
`branch_mode: kv_reuse` is exclusive (fallback disabled) because exact-replay
differs from KV-reuse by up to ~0.11 probability on this stack. Do not reuse
the v2-1 test role.

## Runtime

Default container:

```text
/scratch/project_465002861/kaiserda/frank/build_env/usae-deps.sif
```

Default Hugging Face cache:

```text
/scratch/project_465002861/kaiserda/frank/hf-cache
```

Future v2 configs must pin exact model revisions.

## Historical diagnostic dry run

Retained for provenance only; do not submit without a separately authorized
fresh registration:

```bash
python lumi/cli.py submit \
  --job-name frank-eq-stagea-diagnostic-lumi \
  --config configs/stage0/real_lumi.yaml \
  --profile full \
  --stages cache,validate,diagnose \
  --dry-run --json
```

## Status, fetch, verify

```bash
python lumi/cli.py status --job-name <job> --json
python lumi/cli.py fetch --job-name <job> --json
python lumi/cli.py verify --job-name <job> --json
```

Local state is stored under `.agents/state/lumi/<job>/`. It is ignored and must
not be committed. Adopt only small hash-verified evidence packages.

## Workflow stages

```text
cache,validate,diagnose,train,eval
```

The `diagnose` stage is non-promotional and may be omitted. Existing stage lists
remain backward compatible.

## ROCm and cache cautions

- Transformers receives `device: cuda`; ROCm PyTorch exposes the CUDA API.
- Keep extraction sequential by checkpoint.
- Missing snapshots or unsupported kernels are engineering failures.
- KV reuse counts are insufficient for a scientific cache-equivalence claim;
  v2 requires a frozen sample-wise KV-versus-replay parity audit.
- A valid negative scientific decision is not a scheduler failure.
