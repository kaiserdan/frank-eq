---
name: frank-eq-research
description: Audit and extend Frank-EQ public operational-interface experiments while preserving causal order, rate/compute accounting, and immutable negative evidence.
---

# Frank-EQ research skill

## Use this skill when

- auditing an adopted Frank-EQ result;
- changing a future-operation or response protocol;
- implementing a public operational basis or model-local compiler;
- preparing a development qualification or Stage-A registration;
- adopting a compact evidence package.

## Current authority

Read `AGENTS.md`, `HANDOFF.md`,
`docs/18_RATE_COMPUTE_OPERATIONAL_BASIS.md`, and
`docs/19_STAGE_R_CLUSTER_RUNBOOK.md` before acting.

Stage-A v1/v2 and Stage-Q are preserved negatives. RC0 is an adopted development
pass. Stage-A v3-2 is frozen in:

```text
docs/20_STAGEA_V3_PROTOCOL.md
configs/stagea_v3/real_olivia_v3.yaml
```

The one authorized representation workflow completed on Olivia as job
`1899057`. Its frozen decision is `ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED`; all
receiver and claim authorizations are false. The job deliberately terminated
nonzero when the independent verifier rejected floating-point-exact metric
recomputation. An exact-runtime diagnostic established that config-order
reduction reproduces the stored metrics exactly, while the verifier's
lexicographic bundle order changes 46 numeric fields by at most
`5.551115123125783e-17` and leaves the decision unchanged.

The run is consumed and adopted under `evidence/real_stagea_v3_olivia/`. There
is no authorized next execution: do not retry or tune v3-2, draft a receiver
protocol from it, open receiver worlds, or make a scientific or paper claim.

## Historical RC0 workflow

1. Confirm exact Qwen3-4B and Qwen3-8B revisions.
2. Confirm `chat_turn`, exclusive KV reuse, and no replay fallback.
3. Confirm entity counts `{4,6}`, 96 development worlds each, two renderers,
   and no held/test role.
4. Confirm generated reasoning and fixed-pause budgets are both 32 tokens.
5. Run local compile, lint, tests, `validate_repo.py`, and
   `validate_rate_compute.py`.
6. Dry-run the content-addressed cluster submission.
7. Submit only `audit`.
8. Fetch and run `scripts/verify_rate_compute_run.py`.
9. Inspect aggregate and model/complexity/family strata.
10. Adopt only a compact hash-verified evidence package.

## Scientific invariants

- Form the state before any operation or candidate answer is revealed.
- Treat immediate answer-token probability as one compute/readout protocol, not
  full model competence.
- Separate model-local calibration from state information.
- Compare generated reasoning against an equal-token pause control.
- Use training worlds for calibration/protocol selection and validation worlds
  for frozen scoring.
- Bootstrap worlds, not response rows.
- Public coordinates must have external semantics; do not introduce another
  shared private gauge.
- Runtime basis probing is an upper-bound diagnostic, not a latent interface.
- Behavioral self-future and oracle semantic state are separate objects.
- Report message/rate and downstream compute together.
- Density/reciprocity global tags are controls and cannot promote the method.
- Machine decisions and hashes outrank prose and W&B.

## Historical RC0 promotion logic

Protocol drafting requires:

```text
every model x complexity basis Brier lower95 >= 0
every basis balanced accuracy >= 0.60
compiled hard-operation gain over prior lower95 >= 0
compiled hard-operation gain over train-selected direct lower95 > 0
```

Answer-channel and reasoning-over-pause effects are diagnostics only.

## Historical post-RC0 branches

### Failure

Do not train a larger private latent. Follow the machine diagnosis:

- basis failure: stop or redesign the source/task contract;
- compiler-prior failure: inspect structured calibration/dependence only;
- no direct advantage: preserve as diagnostic, not constructive paper evidence.

### Pass

Draft one Stage-A v3 protocol with fresh worlds, a new unopened held sender,
model-local token/slot compilers into the typed basis, separate behavioral and
semantic channels, and strong text/token/direct/continuous/oracle baselines.
Receiver execution remains locked.

## Completed Stage-A v3-2 sequencing

1. Commit the frozen registration.
2. Implement without changing scientific fields; add focused tests.
3. Commit the green implementation.
4. Commit and inspect `configs/stagea_v3/inspected_plan.json` without creating
   any panel or loading any model.
5. Stage the exact unopened held checkpoint without task queries.
6. Dry-run and launch the registered representation stages through
   `olivia/cli.py`.
7. Require founder and held freeze manifests before test-panel creation.
8. Fetch, verify, independently recompute, document, and commit the outcome.
9. Follow the machine decision; never skip directly to receiver execution.

## Prohibited actions

- Another model-scale screen under immediate A/B scoring.
- Any RC0 stage other than `audit`.
- Tuning gates after validation results.
- Calling interactive tomography a one-shot latent interface.
- Reusing RC0 worlds or exposed models for confirmation/held roles.
- Resuming the shared-head oracle quotient.
- Direct target-hidden reconstruction as the primary continuation.
- Mixing replay and KV branches.
- Retrying, recovering, or tuning the consumed Stage-A v3-2 registration.
- Drafting or executing a receiver protocol from the negative v3-2 decision.
- Using exposed v3-2 outcomes to select a successor registration.
- Committing generated caches, checkpoints, `.agents/state/`, secrets, or W&B
  credentials.

## Validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
```

## Evidence standard

Every adopted result must include frozen config/source identity, causal branch
validation, grouped metrics, machine decision, compact verification, SHA-256
manifest, and an explicit authorization boundary.
