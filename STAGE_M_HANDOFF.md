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

The content-addressed dry run was inspected and the identical frozen command
without `--dry-run` was submitted as Olivia Slurm job `1970800` from source
commit `d4e64bb78b6cce8740a0856bb971a9492fa662e1`:

```text
job:            frank-eq-moment-compute-m0
Slurm:          1970800
source SHA-256: 352215a620e2b9147140a719c8c4ad8666a3cacf51d5d73d1454b5c2ccc27668
config SHA-256: f181fece5f47078c9aae3a04195fd44156efa917648f6f0591192e948d80d51d
image SHA-256:  a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1
```

The job started on `gpu-1-58`; the deployed source/config hashes and requested
one-H200, 32-CPU, 128-GiB, 12-hour allocation were independently checked. Its
in-job static validator passed the frozen contract before model execution.

The next action is to monitor this immutable job to termination, fetch it with
the repository launcher, run the runner verification and
`scripts/verify_moment_compute_run.py`, audit every registered stratum, and
adopt only a compact hash-verified evidence package. Do not submit a second M0
job or launch another architecture, held sender, receiver, or confirmation
experiment before that process is complete.
