---
name: olivia-cluster-runner
description: Package, submit, monitor, fetch, and verify the frozen Frank-EQ PSR0 development audit on Olivia.
---

# Olivia cluster runner

Use `olivia/cli.py`; do not create mutable remote worktrees or ad-hoc copy
procedures. Read `AGENTS.md`, `HANDOFF.md`,
`docs/22_PREDICTIVE_STATE_PSR0.md`, and
`docs/23_PSR0_OLIVIA_RUNBOOK.md` first.

## Current authorized run

```text
job name: frank-eq-psr0-olivia-20260812a
config:   configs/predictive_state/real_olivia_psr0.yaml
plan:     configs/predictive_state/inspected_plan.json
stages:   audit
role:     development-only
```

Historical RC0 and Stage-A v3-2 jobs are consumed. Do not resubmit or tune them.
PSR0 contains no held sender, claim-bearing test role, receiver, or claim.

## Local validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
python scripts/validate_predictive_state.py
```

## Dry run

```bash
python olivia/cli.py submit \
  --job-name frank-eq-psr0-olivia-20260812a \
  --config configs/predictive_state/real_olivia_psr0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

Confirm the source SHA, clean Git state, exact checkpoint snapshots, config hash,
and internal plan SHA before submitting the same package without `--dry-run`.

## Monitor and verify

```bash
python olivia/cli.py status --job-name frank-eq-psr0-olivia-20260812a --json
python olivia/cli.py fetch  --job-name frank-eq-psr0-olivia-20260812a --json
python olivia/cli.py verify --job-name frank-eq-psr0-olivia-20260812a --json
python scripts/predictive_state_cli.py verify \
  --config configs/predictive_state/real_olivia_psr0.yaml \
  --run <fetched-run-root>
```

A scientific failure is a valid completed run. Do not equate scheduler or
engineering success with promotion.

## Invariants

- Checkpoints are pinned Qwen3-4B and Qwen3-8B.
- Stages equal `audit`.
- The committed inspected plan is immutable.
- `chat_turn`, exact prefix continuity, exclusive KV reuse, and no replay
  fallback remain active.
- Future tests are revealed only after prefix capture.
- Models load sequentially.
- Train and validation histories remain disjoint.
- Symbolic grammar and history length 32 remain validation-only.
- No held/test/receiver role is opened.
- Runtime future-test queries are tomography, not message rate.
- Generated runs, captures, source archives, and `.agents/state/` remain outside
  Git.

Only `PUBLIC_PREDICTIVE_STATE_CANDIDATE_SUPPORTED` permits drafting a new
protocol, never executing it automatically.
