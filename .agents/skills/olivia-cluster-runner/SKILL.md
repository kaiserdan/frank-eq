---
name: olivia-cluster-runner
description: Package, dry-run, submit only with authority, monitor, fetch, and verify Frank-EQ jobs on Olivia, including the prospective development-only SPQ0 census.
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
Stage M0 completed and is adopted:

```text
job:       frank-eq-moment-compute-m0 / Slurm 1970800
decision:  OPERATION_CLOSED_EVENTS_NOT_READABLE
evidence:  evidence/real_stage_m_olivia_m0/
```

It has no held sender, claim-bearing test role, receiver access, successor draft
authority, or scientific claim authority. Do not resubmit it. There is no
current graph execution authority.

SPQ0 is the sole prospective Olivia workflow. Implementation and deterministic
dry-run inspection are authorized, but this PR must not launch it. A later
separately authorized submission requires exactly:

```text
configs/spq0/real_olivia_spq0.yaml
--profile full --stages audit
```

It has no recovery, test, held, receiver, or claim path. OLMo2 and Granite are
reserved unopened. Follow `.agents/skills/spq0-runner/SKILL.md` and
`docs/26_SPQ0_OLIVIA_RUNBOOK.md`.

## Validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
python scripts/validate_moment_compute.py
python scripts/validate_spq0.py
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
```

## Historical Stage M0 record

```bash
python olivia/cli.py status --job-name frank-eq-moment-compute-m0 --json
python olivia/cli.py fetch --job-name frank-eq-moment-compute-m0 --json
python olivia/cli.py verify --job-name frank-eq-moment-compute-m0 --json
```

The completed immutable binding is source SHA-256 `352215a6...27668`, config
SHA-256 `f181fece...d51d`, and image SHA-256 `a3ca46f0...aa3b1`. These commands
are inspection only, not resubmission authority.

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

The observed Stage M diagnosis closes the graph/source line. SPQ0 does not
change that conclusion because it asks a fresh stochastic-system question.
Never launch it automatically, access its reserved checkpoints, or treat its
prospective plan as evidence.
