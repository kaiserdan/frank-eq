---
name: lumi-cluster-runner
description: Package, submit, monitor, fetch, and verify Frank-EQ jobs on LUMI-G, including the development-only RC0 audit.
---

# LUMI cluster runner

Use `lumi/cli.py` for content-addressed Frank-EQ submissions. Read `AGENTS.md`,
`HANDOFF.md`, `docs/LUMI.md`, and the protocol governing the selected config.

## Current authority

There is no authorized LUMI execution. RC0 completed on Olivia and is adopted
under `evidence/real_stage_r_olivia_rc0/`. Stage-A v3-2 subsequently completed
on Olivia with the terminal negative decision
`ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED`, adopted under
`evidence/real_stagea_v3_olivia/`.

Do not rerun RC0, v3-2, a historical Stage-A registration, or Stage Q. New
cluster work requires separate scientific justification and a fresh frozen
registration.

## Historical RC0 command surface

The frozen RC0 contract was:

```text
config: configs/rate_compute/real_lumi_rc0.yaml
stages: audit
role: development-only
```

Historical dry run (retained for provenance, not authorization):

```bash
python lumi/cli.py submit \
  --job-name frank-eq-rc0-rate-compute \
  --config configs/rate_compute/real_lumi_rc0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

## Historical operator commands

```bash
python lumi/cli.py status --job-name frank-eq-rc0-rate-compute --json
python lumi/cli.py fetch  --job-name frank-eq-rc0-rate-compute --json
python lumi/cli.py verify --job-name frank-eq-rc0-rate-compute --json
```

Then run the RC0-specific verifier against the fetched run root:

```bash
python scripts/verify_rate_compute_run.py --run <fetched-run-root>
```

## Mandatory checks

- Qwen3-4B and Qwen3-8B exact revisions are present in the shared HF cache.
- `local_files_only: true` remains set.
- The submitted stages string is exactly `audit`.
- Both entity-count panels and both renderers complete.
- All branches use cloned KV reuse; no replay fallback occurs.
- A scientific negative remains a scheduler-successful result.
- Generated runs remain under project scratch and outside Git.

Do not launch historical Stage-A or Stage-Q configs as an adaptive follow-up,
resubmit v3-2, or launch receiver work. Do not reuse RC0 or v3-2 development,
test, model, or renderer roles for later confirmation.

See `AGENTS.md`, `HANDOFF.md`, `docs/19_STAGE_R_CLUSTER_RUNBOOK.md`, and
`evidence/real_stagea_v3_olivia/AUDIT.md` for the current authority and artifact
contract.
