---
name: lumi-cluster-runner
description: Package, submit, monitor, fetch, and verify Frank-EQ Stage-A jobs on LUMI-G.
---

# LUMI cluster runner

Use this skill for Frank-EQ jobs on LUMI. The operator surface is `lumi/cli.py`. It shares the same content-addressed and fail-closed workflow as Olivia.

## Workflow

1. Read `AGENTS.md`, `HANDOFF.md`, `docs/11_REAL_STAGEA_IMPLEMENTATION.md`, and `docs/LUMI.md`.
2. Run local tests and config validation.
3. Dry-run:

   ```bash
   python lumi/cli.py submit \
     --job-name <immutable-name> \
     --config configs/stage0/real_lumi.yaml \
     --profile smoke \
     --stages cache,validate,train,eval \
     --dry-run --json
   ```

4. Submit without `--dry-run`; record the returned source hash and Slurm ID.
5. Poll, fetch, and verify using the same job name.
6. Treat a complete negative scientific decision as a valid result.

## Commands

```bash
python lumi/cli.py status --job-name <name> --json
python lumi/cli.py fetch --job-name <name> --json
python lumi/cli.py verify --job-name <name> --json
```

## LUMI constraints

- The job loads the pinned LUMI/ROCm modules in `lumi/run.slurm` and uses the configured Singularity image.
- Keep caches and temporary compilation state under project scratch; never under `$HOME`.
- Start with one GPU and the smoke panel. Scale only after cache validation and memory telemetry are complete.
- Set `local_files_only: true` for claim-bearing jobs and pre-stage exact checkpoint revisions.
- Do not mix source archives, model revisions, or operation manifests across reruns.

See `references/contract.md` for artifact and failure requirements.
