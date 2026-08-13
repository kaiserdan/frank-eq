---
name: spq0-runner
description: Audit, monitor, fetch, and independently verify the consumed development-only Frank-EQ Shared Predictive Quotient census on Olivia without reopening execution authority.
---

# SPQ0 runner

Read `AGENTS.md`, `HANDOFF.md`,
`evidence/real_spq0_olivia/AUDIT.md`,
`docs/28_SPQ0_RESULT_AND_DISPOSITION.md`,
`docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md`, and
`docs/26_SPQ0_OLIVIA_RUNBOOK.md` completely before acting.

## Authority

SPQ0 is completed and consumed. The sole complete outcome is:

```text
job:       frank-eq-spq0-olivia-20260813c
Slurm:     2006680
state:     COMPLETED 0:0
diagnosis: SOURCE_PROBABILITY_PROTOCOL_NOT_QUALIFIED
evidence:  evidence/real_spq0_olivia/
```

There is no resubmission, recovery, or retuning authority and no test role,
held sender, receiver, SPQ1 draft, or claim-bearing authorization. The original
protocol/runbook remain immutable consumed inputs.

## Scientific dimension contract

The system has homogeneous linear PSR rank four and normalization-aware affine
dimension three. The executor appends the known null-test probability `1` at
zero message cost. Verify the same frozen learned packets under both
conventions: normalization-aware ranks one and two are undercomplete, affine
rank three is exactly sufficient, and robust linear rank four must be
noninferior to affine rank three and linear ranks six/eight. Report 12-bit and
16-bit quantization/rate frontiers. Interpret the validation-only system as a
local 10% law perturbation only.

## Non-access contract

Only exact Qwen3-4B and Mistral-7B-v0.3 founders are active. OLMo2 and Granite
are reserved unopened. Never resolve their snapshots, list/open files,
instantiate adapters, load weights, tokenize a task, or run inference. Require
the zero-access receipt and independent verification.

## Preserved preflight

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
python scripts/validate_moment_compute.py
python scripts/validate_spq0.py
git diff --check
git status --short
```

The completed run satisfied a clean commit, deterministic inspected-plan
equality, exact runtime image, exact active revisions/snapshot files, and
unchanged historical evidence before model inference. Keep these checks for
repository validation; they do not reopen model inference.

## Historical launch binding

```text
source archive: 607ef10289b1446bf2c23c92c9c3cdca386d2c93c2df3b0a4a4175f1ec528579
config:         1ad2216396da3a31421366e951476f14472b216a10588dd23760f90863488558
plan:           9274c19646c0261a8aa7ce21ccffbb9d98f4e535533f1c76951cd9a13e816225
image:          a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1
```

These bindings are provenance only. Do not run a dry run as a precursor to
resubmission and do not invoke `submit` for this consumed registration.

## Status, fetch, and verification

```bash
python olivia/cli.py status --job-name frank-eq-spq0-olivia-20260813c --json
python olivia/cli.py fetch  --job-name frank-eq-spq0-olivia-20260813c --json
python olivia/cli.py verify --job-name frank-eq-spq0-olivia-20260813c --json
python scripts/verify_spq0_run.py \
  --config configs/spq0/real_olivia_spq0.yaml \
  --run .agents/state/olivia/frank-eq-spq0-olivia-20260813c/remote/runs
```

The in-job Python 3.10.12 / NumPy 2.2.6 verifier passes exact recomputation with
zero delta. The workstation Python 3.14.6 / NumPy 2.5.2 entrypoints return
nonzero on a strict `2.04e-9` portability difference in Qwen fitted outputs,
while reproducing the same decision and authorization vector. Preserve both
records. Do not reinterpret the workstation refusal as a reason to repair or
rerun SPQ0.

Audit categorical forecasting, exact prefix continuity, exclusive cloned KV,
zero replay, separated roles, validation-only system/renderer/length, all
surface and token controls, frozen target readers, both ordered cross-family
pairs, zero pair mappers, the complete rank/rate census, non-promotional
residual reporting, grouped metrics, and closed protected authorizations.

Never commit generated runs, caches, checkpoints, `.agents/state/`, secrets, or
W&B credentials. The compact hash-verified review is already adopted; preserve
it byte-for-byte.
