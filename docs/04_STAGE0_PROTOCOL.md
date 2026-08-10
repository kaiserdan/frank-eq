# Synthetic Stage-0 protocol

## Purpose

Stage 0 validates the scientific contracts and tests whether the proposed quotient can be learned under known ground truth before real-model compute is spent.

It is not evidence about language models.

## Generator

Each synthetic world contains:

- binary grounded facts;
- continuous operational residual variables;
- a public future-operation bank;
- multiple renderer nuisance views;
- multiple model-specific hidden charts with heterogeneous widths;
- model-private variation;
- a known future signature for every operation.

All views of one world share the same public causal state.

## Operation families

The reference bank includes:

```text
lookup
xor
and
implication
residual
hybrid
```

Operations are represented by public coefficient descriptors. A family-stratified subset of operation instances is hidden during training and used only for evaluation.

## Training phases

### Founder phase

Train all founder charts and shared public heads using only train worlds and train operation instances. Select the checkpoint on validation task/invariance losses, excluding model-adversary loss from checkpoint selection.

### Held-sender phase

Freeze the public decoder, public heads, founder charts, and all non-held parameters. Train only the held sender chart using source-side train-world labels.

## Evaluation

On untouched test worlds, report:

- seen and held-out future-signature error;
- grounded-fact accuracy;
- residual prediction and facts-only residual margin;
- renderer invariance;
- cross-model same-world retrieval;
- hardest wrong-world margin;
- sender/model-ID leakage;
- held-sender retention;
- 8-bit quantization retention;
- raw hidden pairwise ridge R2 as a non-gating geometry baseline;
- packet round-trip integrity and serialized size.

Bootstrap units are worlds. Views and directed model pairs are averaged within world before resampling.

## Reference configurations

`synthetic_smoke.yaml` checks workflow and artifact integrity with permissive gates and a small sample.

`synthetic_full.yaml` is the authoritative synthetic implementation gate. Its thresholds are frozen in the file and reproduced in `docs/05_GATES_AND_STOP_RULES.md`.
