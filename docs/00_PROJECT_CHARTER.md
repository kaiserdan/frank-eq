# Project charter

## Scientific question

Do heterogeneous language-model hidden states admit a compact, operation-agnostic state whose coordinates are defined by the future computations they support, rather than by another model's native representation?

## Proposed object

For a state `h` cached before an operation is revealed and a frozen operation family `K`, define

```text
Sigma_K(h) = {p(y | h, k) : k in K}.
```

Two states are operationally equivalent when their future signatures agree within the frozen task metric. Frank-EQ seeks a compact quotient state that preserves this equivalence, remains renderer invariant, supports held-out operations, and can be populated by a new sender without changing the shared executor.

## Primary hypotheses

**H1 — Future-defined quotient.** A single operation-agnostic state predicts multiple future branches better, per bit and per parameter, than operation-local or coordinate-reconstruction baselines.

**H2 — Cross-model stability.** Gauge-fixed operational coordinates are more stable across model families and renderers than raw hidden coordinates.

**H3 — Residual beyond explicit facts.** Some future behavior is not determined by the grounded fact ontology alone and is captured by a compact operational residual. If this is false, the residual must be removed.

**H4 — Establishment.** A held sender can learn only a model-local chart into the frozen public quotient while preserving most founder performance.

**H5 — Receiver relevance.** If the representation gate passes, a frozen receiver can use a query-conditioned packet derived from the quotient more reliably than continuous hidden-state reconstruction and rate-matched controls.

H5 is not tested by the current synthetic Stage 0.

## Scope

The project includes:

- controlled synthetic validation;
- real-model future-branch caches;
- model-local source charts;
- public operation registries and interrogators;
- query-conditioned typed packets;
- held-sender establishment;
- receiver-native execution after representation qualification.

It excludes:

- a claim of coordinate identity across models;
- pair-specific primary translators;
- target-hidden access in the strict path;
- joint sender-receiver tuning in establishment experiments;
- unrestricted architecture search after a frozen gate fails.

## Success condition

A strong paper requires a prospectively frozen common matrix showing:

1. operation-agnostic state sufficiency;
2. held-out operation and renderer generalization;
3. low sender-ID leakage;
4. positive operational residual or its principled removal;
5. held-sender onboarding with a frozen executor;
6. receiver-native utility and source specificity;
7. margins over rate-matched text and continuous baselines;
8. uncertainty and harm-tail reporting.

Synthetic success alone is an implementation prerequisite, not a paper result.
