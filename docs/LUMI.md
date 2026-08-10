# LUMI runbook

## Local environment

```bash
export FRANK_EQ_LUMI_HOST=lumi
export FRANK_EQ_LUMI_ROOT=/scratch/project_465002861/kaiserda/frank-eq
```

The default Slurm surface uses one MI250X GCD. Operational resource changes do
not change the scientific contract.

## Current authority

LUMI job `20942127` produced the adopted Stage-A v1 negative. Do not rerun its
test role. Analyze its fetched cache with:

```bash
frank-eq diagnose-real-cache \
  --cache .agents/state/lumi/frank-eq-stagea-devg-v2/remote/runs/cache \
  --out runs/diagnostics/frank-eq-stagea-devg-v2
```

This uses train/validation worlds only and authorizes nothing.

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

## Diagnostic dry run and submit

For a fresh operational smoke, not a v1 scientific rerun:

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
