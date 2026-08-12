---
name: moment-compute-runner
description: Validate, submit, monitor, fetch, and verify the development-only Frank-EQ Stage-M0 operation-closed basis audit on Olivia.
---

# Frank-EQ Stage M0 runner

Read, in order:

1. `docs/22_MAIN_RESULTS_AUDIT_AND_STAGE_M.md`
2. `docs/23_STAGE_M_OPERATION_CLOSED_BASIS.md`
3. `docs/24_STAGE_M_OLIVIA_RUNBOOK.md`
4. `evidence/real_stage_m_olivia_m0/AUDIT.md`
5. `evidence/real_stagea_v3_olivia/AUDIT.md`
6. `AGENTS.md`

## Current authority

The historical Stage-A v3-2 registration is consumed and cannot be retried or
tuned. Stage M0 is also complete and adopted:

```text
job:       frank-eq-moment-compute-m0 / Slurm 1970800
decision:  OPERATION_CLOSED_EVENTS_NOT_READABLE
evidence:  evidence/real_stage_m_olivia_m0/
```

It has no held sender, test role, receiver access, claim authority, successor
draft authority, or rerun authority. There is no current Stage-M command
surface for execution.

## Required invariants

- State formation precedes every public-event and target query.
- Use corrected `chat_turn` formatting and literal cloned-KV branches.
- Do not allow replay fallback or mixed branch modes.
- Calibration, direct-protocol selection, and validation worlds are disjoint.
- Fit event calibration by model, event kind, and event order; never by event ID.
- Public event coordinates are generated from the full operation grammar.
- Preserve the historical marginal executor as a control.
- Exact binary event truth must reproduce the formal executor before launch.
- Bootstrap worlds, not individual response rows.
- A valid negative is scheduler-successful and scientifically preserved.
- Never commit model caches, raw runs, `.agents/state/`, credentials, or W&B keys.

## Historical retrieval commands

```bash
python olivia/cli.py status --job-name frank-eq-moment-compute-m0 --json
python olivia/cli.py fetch --job-name frank-eq-moment-compute-m0 --json
python olivia/cli.py verify --job-name frank-eq-moment-compute-m0 --json
python scripts/verify_moment_compute_run.py \
  --run .agents/state/olivia/frank-eq-moment-compute-m0/remote/runs
```

These commands inspect the immutable completed job. They do not authorize
resubmission or a modified verifier contract.

## Decision boundary

The observed diagnosis is `OPERATION_CLOSED_EVENTS_NOT_READABLE`, so the frozen
stop rule closes the current graph/source line. Do not draft or train a Stage M
compiler, enlarge the basis, tune readout, or rerun M0. A future task requires a
new scientific question and separately frozen authority.
