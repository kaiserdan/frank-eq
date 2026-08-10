# Frank-EQ handoff

Snapshot: 2026-08-10

## Current state

The repository now contains a complete synthetic Stage-0 implementation of future-defined operational equivalence quotients.

Implemented:

- deterministic heterogeneous multi-model benchmark;
- operation-before/after causal-boundary contracts;
- model-local private charts;
- gauge-fixed public state coordinates;
- parameter-free public future-operation execution;
- held-out operation instances;
- renderer invariance and wrong-world collision tests;
- founder training and frozen-decoder held-sender onboarding;
- grounded-facts-only comparison and operational residual test;
- model-ID leakage probe;
- quantized typed packet with deterministic checksum;
- grouped bootstrap intervals;
- fail-closed gate reducer and artifact manifest;
- tests, CI, cluster wrapper, and agent documentation.

## Verified reference result

`configs/stage0/synthetic_full.yaml` completed locally on CPU. The adopted small evidence copy is under `evidence/reference_stage0/`.

Selected results:

```text
held-out signature Brier:          0.0370
fact accuracy:                     0.9515
renderer cosine:                   0.9598
cross-model retrieval top-1:       0.9452
wrong-world margin:                0.2312
operational residual Brier gain:   0.0338
held-sender retention:             0.9553
model-ID leakage over chance:     -0.0164
8-bit quantization retention:      1.0000
```

The decision is `PROMOTE_REAL_MODEL_CANARY`. Its scope is synthetic implementation validation only; `authorizes_scientific_claim` is false.

## Next scientific action

Implement the real-model Stage A cache without changing the public quotient or gates:

1. choose two founder checkpoints and one unopened held sender;
2. freeze 8–16 future operations and their canonical outcome spaces;
3. cache hidden states before operation reveal;
4. branch each cached state through every operation;
5. verify world-grouped splits and renderer swaps;
6. compare raw hidden, prior canonical, Bary-style predictable, facts-only, and Frank-EQ quotient candidates;
7. stop before receiver execution unless the real representation gate passes.

See `docs/06_REAL_MODEL_PLAN.md`.

## Known limitations

- The reference generator exposes exact fact and residual targets; real LLMs will require source-side probes or formal task labels.
- The frozen public decoder is exact for the synthetic operation algebra. A real operation registry must be frozen before outcomes and may require a public solver/interrogator.
- No natural-language receiver is integrated.
- No W&B integration is required for authority; if added, it remains telemetry only.
- The learned workspace gate is present but the reference configuration does not yet produce sparse gates. It is not part of the passing claim.

## Do not do next

Do not add another target hidden-state decoder, CSSM-like dynamics stack, pair-specific translator, or learned receiver rescue before the real representation gate. These would reopen already falsified branches without testing the new object.
