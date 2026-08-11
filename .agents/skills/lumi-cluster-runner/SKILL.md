---
name: lumi-cluster-runner
description: Package, submit, monitor, fetch, and verify Frank-EQ jobs on LUMI-G, including the development-only RC0 audit.
---

# LUMI cluster runner

Use `lumi/cli.py` for content-addressed Frank-EQ submissions. Read `AGENTS.md`,
`HANDOFF.md`, `docs/LUMI.md`, and the protocol governing the selected config.

## Current authorized run

The only authorized new scientific execution is RC0:

```text
config: configs/rate_compute/real_lumi_rc0.yaml
stages: audit
role: development-only
```

Dry run:

```bash
python lumi/cli.py submit \
  --job-name frank-eq-rc0-rate-compute \
  --config configs/rate_compute/real_lumi_rc0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

After checking the source hash and exact model revisions, submit without
`--dry-run`.

## Operator commands

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

Do not launch historical Stage-A or Stage-Q configs as an adaptive follow-up.
Do not reuse RC0 development worlds or exposed models for a later held or
confirmation role.

See `docs/19_STAGE_R_CLUSTER_RUNBOOK.md` and `references/contract.md` for the
full artifact contract.
