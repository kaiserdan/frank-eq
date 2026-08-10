# Agent operating contract

## Mission

Frank-EQ studies whether LLM hidden states admit a compact, future-defined operational equivalence quotient that can be populated by independently trained senders and executed by frozen receivers.

The project is not an open-ended hidden-state architecture search. Every experiment must answer a frozen scientific question and must fail closed.

## Reading order

1. `README.md`
2. `docs/00_PROJECT_CHARTER.md`
3. `docs/01_SCIENTIFIC_HYPOTHESIS.md`
4. `docs/02_ARCHITECTURE.md`
5. `docs/03_INFORMATION_ACCESS_CONTRACT.md`
6. `docs/04_STAGE0_PROTOCOL.md`
7. `docs/05_GATES_AND_STOP_RULES.md`
8. `docs/09_IMPLEMENTATION_STATUS.md`
9. `HANDOFF.md`
10. `docs/10_DECISION_LOG.md`

## Non-negotiable scientific invariants

### State formation precedes operation reveal

A claim-bearing state must be cached before the future operation, query, intervention, verification request, or target is revealed. The operation may enter the decoder or query-conditioned selector only after state formation.

`src/frank_eq/contracts.py` enforces this boundary for real caches.

### Split by world, never by row

All renderers, model views, operations, branches, and seeds belonging to one world remain in one split. A duplicated world under another surface form is the same grouping unit.

### Local charts, public semantics

Model-local charts may be arbitrarily expressive. The shared quotient must have externally defined operational semantics. Do not force private chart coordinates to be identical and do not credit private chart geometry as interoperability.

### Held-sender onboarding freezes the shared executor

The held sender may train its own source-local chart using source-side labels. The public decoder, packet schema, receiver, operation registry, bit allocation, and gates remain frozen.

### No target-private rescue

The strict path may not consume target hidden states, target logits, target labels, pair IDs, or receiver gradients at sender runtime. Assisted upper bounds must use a separate metric namespace.

### Facts-only is mandatory

Every operational residual claim must beat a grounded-facts-only implementation under the same operation bank and rate. If the residual does not add a positive held-out margin, remove it.

### Machine decisions are authoritative

README prose, W&B summaries, and training loss do not authorize continuation. Only a complete decision artifact with all required hashes and a passing parent gate can unlock the next phase.

## Prohibited shortcuts

- Selecting layers, operation families, rates, or thresholds after test outcomes.
- Treating hidden-state R2, CKA, cosine, or reconstruction MSE as execution evidence.
- Revealing the operation before caching the state.
- Splitting renderer variants or branches of one world across roles.
- Jointly training a sender and receiver in the primary establishment condition.
- Replacing a failed compiled packet with oracle fields at evaluation time.
- Reporting synthetic Stage 0 as evidence about real LLMs.
- Committing model weights, private datasets, API keys, or W&B credentials.

## Development workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m compileall -q src scripts
pytest -q
python scripts/validate_repo.py
```

For a behavioral change:

1. add or update a focused test;
2. update the relevant protocol document before any outcome-bearing run;
3. run the synthetic smoke;
4. run the full Stage-0 reference if metrics or gates change;
5. append a decision-log entry;
6. update `HANDOFF.md` and `docs/09_IMPLEMENTATION_STATUS.md`.

## File map

```text
src/frank_eq/config.py              Frozen configuration schema
src/frank_eq/contracts.py           Real-cache information-boundary contracts
src/frank_eq/data/synthetic.py      Controlled Stage-0 generator
src/frank_eq/models/encoder.py      Local charts and gauge-fixed quotient
src/frank_eq/models/layers.py       Frozen public decoder and neural primitives
src/frank_eq/training/              Objectives and held-sender training
src/frank_eq/evaluation/            Metrics, bootstrap, and reducer
src/frank_eq/packet/                Typed query-conditioned wire protocol
configs/stage0/                     Smoke and full frozen configurations
evidence/reference_stage0/          Small reproducible reference evidence
docs/                               Scientific and operational contracts
```

## Artifact hierarchy

Use evidence in this order:

1. frozen protocol/config and source hash;
2. machine decision and reducer;
3. held-out metric artifact;
4. prediction artifact;
5. training summary/history;
6. W&B as secondary telemetry;
7. prose documentation.

## Current authority

Synthetic Stage 0 is implemented and the full reference configuration passes. No real-model cache or receiver execution result exists. The only authorized next scientific action is the real-model Stage-A cache and operation-bank canary described in `docs/06_REAL_MODEL_PLAN.md`.
