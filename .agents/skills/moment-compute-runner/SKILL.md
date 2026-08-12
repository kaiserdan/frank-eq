---
name: moment-compute-runner
description: Validate, submit, monitor, fetch, and verify the development-only Frank-EQ Stage-M0 operation-closed basis audit on Olivia.
---

# Frank-EQ Stage M0 runner

Read, in order:

1. `docs/22_MAIN_RESULTS_AUDIT_AND_STAGE_M.md`
2. `docs/23_STAGE_M_OPERATION_CLOSED_BASIS.md`
3. `docs/24_STAGE_M_OLIVIA_RUNBOOK.md`
4. `evidence/real_stagea_v3_olivia/AUDIT.md`
5. `AGENTS.md`

## Current authority

The historical Stage-A v3-2 registration is consumed and cannot be retried or
tuned. Stage M0 is a new development-only question. It has no held sender, test
role, receiver access, or claim authority.

The only Stage-M command surface is:

```text
config: configs/moment_compute/real_olivia_m0.yaml
stage:  audit
```

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

## Commands

```bash
python scripts/validate_moment_compute.py
export FRANK_EQ_IMAGE_SHA256=a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1
python olivia/cli.py submit \
  --job-name frank-eq-moment-compute-m0 \
  --config configs/moment_compute/real_olivia_m0.yaml \
  --profile full --stages audit --dry-run --json
```

Reject a plan with a null or different image hash. After source, image, remote
target, and checkpoint inspection, remove `--dry-run`. Fetch and verify with
`olivia/cli.py`, then run `scripts/verify_moment_compute_run.py`.

## Decision boundary

Only `OPERATION_CLOSED_MOMENT_BASIS_SUPPORTED` permits drafting a separately
frozen one-shot compiler protocol. It does not authorize that run. All other
diagnoses stop or redirect the graph/source line according to
`docs/23_STAGE_M_OPERATION_CLOSED_BASIS.md`.
