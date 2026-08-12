---
name: olivia-cluster-runner
description: Package, submit, monitor, fetch, and verify Frank-EQ jobs on Olivia, including the development-only Stage-M0 operation-closed basis audit.
---

# Olivia cluster runner

Use `olivia/cli.py`; do not create mutable remote worktrees or ad-hoc copy
procedures. Read `STAGE_M_HANDOFF.md`, the selected protocol, and the matching
experiment skill before acting.

## Current authority

The historical Stage-A v3-2 job is consumed and terminal:

```text
job:      frank-eq-stagea-v3-2-olivia-20260812b / Slurm 1899057
decision: ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED
evidence: evidence/real_stagea_v3_olivia/
```

Do not resubmit, repair, recover, or tune it. RC0 and Stage Q also remain closed.

The only newly prepared experiment is the development-only Stage M0 audit:

```text
config:  configs/moment_compute/real_olivia_m0.yaml
profile: full
stages:  audit
```

It has no held sender, claim-bearing test role, receiver access, or scientific
claim authority.

## Validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
python scripts/validate_moment_compute.py
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
```

## Stage M0 dry run

```bash
python olivia/cli.py submit \
  --job-name frank-eq-moment-compute-m0 \
  --config configs/moment_compute/real_olivia_m0.yaml \
  --profile full --stages audit --dry-run --json
```

Inspect the deterministic source hash, config hash, clean Git state, exact model
revisions, and checkpoint availability. Submit the identical command without
`--dry-run` only after those checks.

## Monitor, fetch, verify

```bash
python olivia/cli.py status --job-name frank-eq-moment-compute-m0 --json
python olivia/cli.py fetch  --job-name frank-eq-moment-compute-m0 --json
python olivia/cli.py verify --job-name frank-eq-moment-compute-m0 --json
python scripts/verify_moment_compute_run.py \
  --run .agents/state/olivia/frank-eq-moment-compute-m0/remote/runs
```

## Invariants

- Corrected `chat_turn`, literal cloned-KV branching, no replay fallback.
- Calibration, protocol-selection, and validation worlds are disjoint.
- Event calibration is not event-ID specific.
- The event registry is generated from the full four-entity operation grammar.
- The historical marginal executor remains a frozen control.
- Exact public algebra is validated before model inference.
- Scheduler completion and scientific promotion remain separate.
- Generated runs, source archives, checkpoints, `.agents/state/`, and credentials
  remain outside Git.

Only `OPERATION_CLOSED_MOMENT_BASIS_SUPPORTED` permits drafting a fresh
successor protocol. It does not authorize that run, a held sender, receiver work,
or a paper claim.
