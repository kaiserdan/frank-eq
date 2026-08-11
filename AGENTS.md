# Agent operating contract

## Mission

Frank-EQ studies whether an LLM state formed before an operation is revealed can
be compiled into a public operational interface. The project is not an
unrestricted hidden-state architecture search. Every experiment must answer a
frozen question, preserve information boundaries, and fail closed.

## Reading order

1. `README.md`
2. `HANDOFF.md`
3. `docs/18_RATE_COMPUTE_OPERATIONAL_BASIS.md`
4. `docs/19_STAGE_R_CLUSTER_RUNBOOK.md`
5. `evidence/real_stagea_lumi_v2/REVIEW.md`
6. `docs/16_STAGEA_V2_INTERPRETATION_CORRECTION.md`
7. `docs/17_STAGEQ_EXECUTION_AND_GATE_CONTRACT.md`
8. `docs/03_INFORMATION_ACCESS_CONTRACT.md`
9. `docs/05_GATES_AND_STOP_RULES.md`
10. `docs/09_IMPLEMENTATION_STATUS.md`
11. `docs/10_DECISION_LOG.md`
12. `docs/13_STAGEA_V1_CORRECTION_LOG.md`

For cluster work, also read the matching runner skill under `.agents/skills/`.

## Current authority

Synthetic Stage 0 passes as implementation evidence only. Real Stage-A v1 and
v2 are exact-pipeline negatives. Stage Q and its scale screens are development
negatives. None authorizes another latent run, receiver execution, or a claim.

The only authorized next execution is **Stage R / RC0**:

```text
configs/rate_compute/real_lumi_rc0.yaml
configs/rate_compute/real_olivia_rc0.yaml
```

Run exactly:

```text
--stages audit
```

RC0 is development-only. A passing machine decision permits drafting one
Stage-A v3 protocol; it never permits launching that protocol.

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

## RC0 execution procedure

1. Run local compile, lint, tests, and both repository validators.
2. Inspect the content-addressed dry-run plan.
3. Launch one frozen config with `--stages audit`.
4. Fetch and verify the run.
5. Run `scripts/verify_rate_compute_run.py`.
6. Audit the basis, compiled, answer-channel, and reasoning/pause strata.
7. Adopt only a compact hash-verified evidence package.
8. Do not launch Stage-A v3 merely because RC0 passes.

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

## Development validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
```

For any behavioral change, add a focused test and update the frozen protocol
before an experiment is launched.
