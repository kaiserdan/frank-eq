---
name: olivia-cluster-runner
description: Package, submit, monitor, fetch, and verify Frank-EQ jobs on Olivia, including the development-only RC0 audit.
---

# Olivia cluster runner

Use `olivia/cli.py`; do not create mutable remote worktrees or ad-hoc copy
procedures. Read `AGENTS.md`, `HANDOFF.md`, `docs/OLIVIA.md`, and the selected
protocol first.

## Current authorized run

```text
config: configs/rate_compute/real_olivia_rc0.yaml
stages: audit
role: development-only
```

Local validation:

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
```

Dry run:

```bash
python olivia/cli.py submit \
  --job-name frank-eq-rc0-rate-compute \
  --config configs/rate_compute/real_olivia_rc0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

Submit only after checking the deterministic source SHA-256 and confirming both
exact model revisions are available on Olivia.

## Operator commands

```bash
python olivia/cli.py status --job-name frank-eq-rc0-rate-compute --json
python olivia/cli.py fetch  --job-name frank-eq-rc0-rate-compute --json
python olivia/cli.py verify --job-name frank-eq-rc0-rate-compute --json
python scripts/verify_rate_compute_run.py --run <fetched-run-root>
```

## Invariants

- Stages must equal `audit`.
- No model revision substitution or network-resolved unpinned head.
- Corrected `chat_turn`, exclusive KV reuse, no replay fallback.
- Both complexity panels and renderer views complete.
- Scheduler completion and scientific promotion remain separate.
- Generated caches, responses, source archives, and `.agents/state/` stay out of
  Git.
- RC0 development worlds and exposed models cannot become later confirmation or
  held roles.

Do not launch receiver execution or another Stage-Q scale screen. See
`docs/19_STAGE_R_CLUSTER_RUNBOOK.md` and `references/contract.md` for the full
artifact and failure contract.
