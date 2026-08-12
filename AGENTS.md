# Agent operating contract

## Mission

Frank-EQ studies whether an LLM state formed before an operation is revealed can
be compiled into a public operational interface. The project is not an
unrestricted hidden-state architecture search. Every experiment must answer a
frozen question, preserve information boundaries, and fail closed.

## Reading order

1. `README.md`
2. `HANDOFF.md`
3. `STAGE_M_HANDOFF.md`
4. `docs/22_MAIN_RESULTS_AUDIT_AND_STAGE_M.md`
5. `docs/23_STAGE_M_OPERATION_CLOSED_BASIS.md`
6. `docs/24_STAGE_M_OLIVIA_RUNBOOK.md`
7. `.agents/skills/moment-compute-runner/SKILL.md`
8. `evidence/real_stagea_v3_olivia/AUDIT.md`
9. `docs/20_STAGEA_V3_PROTOCOL.md`
10. `docs/18_RATE_COMPUTE_OPERATIONAL_BASIS.md`
11. `docs/19_STAGE_R_CLUSTER_RUNBOOK.md`
12. `evidence/real_stage_r_olivia_rc0/AUDIT.md`
13. `evidence/real_stagea_lumi_v2/REVIEW.md`
14. `docs/16_STAGEA_V2_INTERPRETATION_CORRECTION.md`
15. `docs/17_STAGEQ_EXECUTION_AND_GATE_CONTRACT.md`
16. `docs/03_INFORMATION_ACCESS_CONTRACT.md`
17. `docs/05_GATES_AND_STOP_RULES.md`
18. `docs/09_IMPLEMENTATION_STATUS.md`
19. `docs/10_DECISION_LOG.md`
20. `docs/13_STAGEA_V1_CORRECTION_LOG.md`

For cluster work, also read the matching runner skill under `.agents/skills/`.

## Current authority

Synthetic Stage 0 passes as implementation evidence only. Real Stage-A v1 and
v2 are exact-pipeline negatives. Stage Q and its scale screens are development
negatives. Stage R / RC0 is an adopted development pass:

```text
capture:  frank-eq-rc0-rate-compute-olivia-20260811c  Slurm 1874736
recovery: frank-eq-rc0-rate-compute-olivia-20260811d-recovery  Slurm 1891471
result:   PUBLIC_BASIS_COMPOSITION_SUPPORTED
```

Stage-A v3-2 completed on Olivia as job `1899057`. It is an adopted
exact-pipeline negative:

```text
decision:  ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED
evidence:  evidence/real_stagea_v3_olivia/
```

Every integrity check passes and the test grant was consumed once. Behavioral
basis, public alignment, held-sender retention, quantization, and oracle
execution pass. Semantic basis, unseen-renderer transfer, activation
specificity, and conjunctive composition fail.

The terminal job failure is a fail-closed verifier bundle-order refusal. An
exact-runtime reducer diagnostic reproduces stored metrics bit-for-bit in
workflow/config order and the same machine decision in either order. Preserve
the original failed audit; do not relabel it as a pass.

Stage-A v3-2 cannot be retried, recovered, or tuned. It also must not be
broadened into a claim that every public operational state fails.

The sole newly authorized execution is the development-only Stage M0 audit:

```text
config:  configs/moment_compute/real_olivia_m0.yaml
cluster: Olivia
profile: full
stages:  audit
```

Stage M0 has no held sender, claim-bearing test role, receiver access, or claim
authority. The immediate executable is the content-addressed Olivia dry run.
Submit the identical command without `--dry-run` only after the complete local
contract passes and the source hash, config hash, exact revisions, clean Git
state, remote target absence, and checkpoint availability are inspected.

## Completed Stage-A v3 execution boundary

- Fresh train/validation/test seeds `2026081201/02/1297` are now exposed and
  cannot be reused for confirmation. The v3-1 test seed `2026081203` remains
  permanently void.
- The held `Qwen/Qwen3-14B` revision
  `40c069824f4251a91eefaf281ebe4c544efd3e18` is task-exposed under v3-2 and
  cannot serve as a future unopened sender.
- Test panels must not exist before founder and held freeze manifests.
- The primary compiler consumes all prefix-token residuals at four frozen depths
  and makes zero post-capture source queries.
- Semantic and behavioral compilers have separate model-local parameters,
  losses, checkpoints, and metrics.
- Run every registered token/text/direct/continuous/interactive/oracle control.
- World-group all intervals and report message rate with source and consumer
  compute.
- A valid gate miss is terminal for v3-2.

## Stage M0 execution boundary

- Stage M0 asks a fresh development-only question: whether the public state is
  undercomplete for nonlinear operations because first-order edge marginals do
  not preserve the required joint events.
- The frozen panel has 64 four-entity worlds, 32 operations, two renderers, and
  disjoint calibration, direct-protocol-selection, and validation roles. It has
  no held or test role.
- The deterministic full-grammar registry has 318 typed event coordinates.
  Static validation reports registry SHA-256 `70ce5d31...a6d55`, contract
  SHA-256 `769fbf65...8326`, and zero exact-executor mismatches.
- Use only pinned Qwen3-4B and Qwen3-8B founders, corrected `chat_turn`, literal
  cloned-KV branches, and no replay fallback.
- Export the inspected Olivia image SHA-256 `a3ca46f0...aa3b1` before planning;
  a null or different runtime-image hash invalidates the plan.
- Fit event calibration only by model, event kind, and event order. Never fit by
  event ID. Preserve the historical marginal/independence executor as a frozen
  control.
- Exact binary event truth must reproduce the formal operation executor before
  model inference. Bootstrap worlds, not response rows.
- Stage M0 is interactive event tomography, not a one-shot compiler,
  communication interface, held-sender result, or receiver result.
- Only `OPERATION_CLOSED_MOMENT_BASIS_SUPPORTED` may authorize drafting one
  separately frozen successor compiler protocol. It never authorizes launching
  that protocol.

## Scientific invariants

### Preserve exact negatives without broadening them

The completed shared-head, final-token architectures failed. Do not generalize
those results to the full KV/runtime state or to all possible operational
interfaces.

### A future signature includes an explicit compute contract

Use:

```text
Sigma_M(h; k, c) = p_M(y | h, k, c)
```

where `c` records the answer/readout protocol and any post-reveal token budget.
Do not call immediate next-token A/B probability the model's full future
computational competence.

### Separate calibration, computation, and information

RC0 distinguishes:

- historical A/B answer-token logits;
- semantic candidate-sequence likelihood;
- generated reasoning tokens;
- matched fixed pause tokens;
- public-basis sufficiency.

A train-only local calibration map may reverse a stable answer-label inversion.
That is a readout correction, not evidence that the state was absent.

### State formation precedes operation reveal

No operation, query, target, candidate answer, or future label may influence the
captured prefix state. `chat_turn` must verify exact token-prefix continuity.

### Use exclusive KV branching

KV reuse and replay differed materially on the LUMI stack. RC0 requires cloned
KV branches and forbids replay fallback or mixed caches.

### Public coordinates must be identifiable

A shared private vector is not a public interface. RC0's directed-edge basis is
gauge fixed by semantics: coordinate `(i,j)` always means the same fact.

A future Stage-A compiler may be completely model local. Only typed coordinate
meaning, packet schema, and executor are shared.

### A separating basis is an upper-bound diagnostic, not yet a latent interface

RC0 queries the source model for every basis coordinate after capture. This
interactive tomography tests information and composition. It is not a one-shot
hidden-state compiler and cannot support a communication claim.

Stage-A v3, only after RC0 passes, must learn a source-local token/slot compiler
that emits the same typed coordinates without runtime basis interrogation.

### Behavioral and semantic channels are distinct

`model_signatures` describe what the frozen source will do. Oracle facts describe
external correctness. Do not merge them into one unnamed loss or claim.

### Consumer compute is part of the interface

A message can be useful only relative to an executor and compute budget. Report
both message/rate and downstream computation; do not compare a reusable basis
against a single direct answer without the amortization boundary.

### World is the paired independent unit

All model, renderer, operation, basis, and protocol rows for a world remain in
one development split. Calibration and protocol selection use training worlds;
validation worlds are scored only after those choices are frozen.

### Declared controls are not discoveries

Density and reciprocity labels are printed in the historical graph prefixes.
They remain controls and cannot promote RC0.

### Machine artifacts outrank prose

Use evidence in this order:

1. frozen config and source identity;
2. causal branch and cache validation;
3. machine decision;
4. world-grouped metrics;
5. response artifacts and calibration state;
6. W&B telemetry;
7. prose.

## Completed RC0 execution record

1. Local compile, lint, tests, shell checks, and both repository validators passed.
2. The content-addressed Olivia plan ran only the frozen audit stage.
3. A post-capture aggregation failure was recovered artifact-only under a fresh
   source and job, with no model inference and no prior outcome artifacts.
4. Repository and RC0-specific verification passed.
5. An independent audit reproduced the metrics, strata, and decision.
6. The compact hash-verified package is under
   `evidence/real_stage_r_olivia_rc0/`.
7. Do not launch Stage-A v3 merely because RC0 passes.

## Completed Stage-A v3-2 execution record

1. The frozen registration, implementation, numeric storage repair, and
   content-addressed plan passed the full local contract.
2. The immutable Olivia outcome job preserved founder freeze, held onboarding,
   held freeze, one-time test access, and evaluation order.
3. All six test captures and all 12 prediction files completed with no capture
   error; all eleven integrity checks pass.
4. The machine decision is `ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED`.
5. The original independent audit fails only byte-exact metric recomputation;
   decision and rate/compute recompute exactly.
6. Exact-runtime diagnostic job `1953471` localizes the refusal to bundle-order
   roundoff no larger than `5.55e-17`, with no gate or decision change.
7. The complete fetched tree matches Olivia byte-for-byte, and the compact
   package is adopted under `evidence/real_stagea_v3_olivia/`.
8. Stop this compiler/task contract. Do not launch receiver work or another
   v3-2 run.

## RC0 decision tree

### `BASIS_READOUT_NOT_QUALIFIED`

Stop the current source/task contract. Do not enlarge the latent architecture.

### `PUBLIC_BASIS_NOT_SUFFICIENT`

Inspect only structured calibration or executor assumptions on development data.

### `NO_COMPOSITION_ADVANTAGE_OVER_TRAIN_SELECTED_DIRECT_BASELINE`

The basis is a diagnostic, not the constructive paper result.

### `PUBLIC_BASIS_COMPOSITION_SUPPORTED`

Draft exactly one fresh Stage-A v3 protocol using:

- new claim-bearing worlds;
- a new unopened held sender;
- complete model-local token/slot compilers;
- separate behavioral and oracle-semantic channels;
- token/text/direct/continuous/oracle baselines;
- receiver work still locked.

## Prohibited shortcuts

- Another scale-only Stage-Q screen under immediate A/B readout.
- Running RC0 with anything other than `audit`.
- Tuning RC0 thresholds after validation outcomes are read.
- Treating runtime basis probing as latent communication.
- Reusing RC0 worlds or exposed models as future held/confirmation roles.
- Resuming the shared-head oracle quotient.
- Mapping directly into a receiver hidden state as the primary method.
- Mixing replay and KV branches.
- Jointly training sender and receiver in the primary condition.
- Committing generated caches, checkpoints, `.agents/state/`, API keys, or W&B
  credentials.
- Creating or opening Stage-A v3 test panels before both compiler freeze steps.
- Selecting a compiler seed, baseline, calibrator, threshold, renderer, or
  checkpoint after v3 test access.
- Retrying or recovering Stage-A v3-2 after its consumed test access.
- Using v3-2 test metrics to tune or select a successor registration.
- Drafting a receiver protocol from the negative v3-2 decision.
- Running Stage M0 with any stage other than exactly `audit`.
- Changing the Stage M event registry, panel, roles, calibrators, baselines,
  thresholds, or gates after validation outcomes are read.
- Calling Stage M interactive event tomography a one-shot interface or a
  communication result.
- Launching a successor compiler, held-sender, confirmation, or receiver run
  merely because Stage M0 passes.

## Development validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
python scripts/validate_moment_compute.py
```

For any behavioral change, add a focused test and update the frozen protocol
before an experiment is launched.
