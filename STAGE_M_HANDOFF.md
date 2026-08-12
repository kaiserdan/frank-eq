# Stage M handoff

Stage M0 was merged into `main` by merge commit `659c120`. This handoff is now
the live continuation guide; it does not alter the immutable Stage-A v3-2
evidence.

## Current authority

Stage-A v3-2 remains consumed with
`ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED`. Do not rerun, tune, or use it to authorize
receiver work.

The newly prepared next experiment is Stage M0:

```text
question: can an operation-closed joint-event basis explain the gap between
          readable atomic facts and failed nonlinear composition?
config:   configs/moment_compute/real_olivia_m0.yaml
cluster:  Olivia
stage:    audit
role:     development-only
```

Stage M0 has no held sender, claim-bearing test role, receiver access, or claim
authority. A pass authorizes only drafting a separately frozen one-shot compiler
protocol.

Static contract validation currently reports 64 four-entity worlds, 32
operations, 318 typed event coordinates, two pinned founder models, zero exact
executor mismatches, and all protected authorizations closed.

## Why Stage M exists

The historical executor applies nonlinear graph operations to first-order edge
marginals under an independence assumption. Nonlinear future operations require
joint events in general. Stage M queries the sparse conjunction and joint-degree
events needed to close the frozen four-entity operation algebra, then compares
its exact executor with both the marginal executor and a cross-fitted direct
response baseline.

## Reading order

1. `docs/22_MAIN_RESULTS_AUDIT_AND_STAGE_M.md`
2. `docs/23_STAGE_M_OPERATION_CLOSED_BASIS.md`
3. `docs/24_STAGE_M_OLIVIA_RUNBOOK.md`
4. `.agents/skills/moment-compute-runner/SKILL.md`
5. `evidence/real_stagea_v3_olivia/AUDIT.md`

## Next action

Run the complete local validation contract, commit any documentation-only
reconciliation so the source tree is clean, and execute exactly:

```bash
export FRANK_EQ_IMAGE_SHA256=a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1
python olivia/cli.py submit \
  --job-name frank-eq-moment-compute-m0 \
  --config configs/moment_compute/real_olivia_m0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

Inspect the resulting source/config hashes, exact revisions, runtime image,
resources, checkpoint availability, and fresh remote path. Only then may the
identical command without `--dry-run` be submitted. Do not launch another
architecture, held sender, receiver, or confirmation experiment before the M0
machine decision is fetched, independently verified, and adopted.
