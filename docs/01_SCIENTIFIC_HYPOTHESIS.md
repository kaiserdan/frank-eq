# Scientific hypothesis: operational equivalence quotients

## Why hidden-state alignment is underidentified

A hidden state contains many directions irrelevant to a particular receiver behavior, while small directions may dominate a receiver's decision boundary. Euclidean reconstruction, CKA, cosine, and pooled R2 therefore do not identify operational equivalence.

Furthermore, an encoder-decoder interface has an invertible gauge ambiguity. For any invertible `G`, the pair

```text
E' = G o E
D' = D o G^-1
```

has the same composed objective. Independently trained senders have no reason to choose the same private gauge.

Frank-EQ fixes the gauge by defining public coordinates through externally registered future operations.

## Operation-agnostic state formation

The state is formed from the source prefix before the operation is known:

```text
z = E_model(h_prefix).
```

Only afterward does a frozen operation descriptor enter:

```text
p(y | z, k) = I(z, descriptor(k)).
```

This distinguishes a reusable causal state from a query-conditioned answer encoder.

## Explicit state and operational residual

The public code has two parts:

```text
z_public = [grounded declarative coordinates, operational residual].
```

The facts-only path predicts future operations from grounded coordinates while marginalizing the residual. The residual is retained only if it improves held-out future-signature prediction under paired uncertainty.

## Quotient interpretation

The public code is not asserted to reconstruct the complete native state. It identifies the equivalence class relevant to the frozen operation family. Different native states may map to the same public state, and the same public state may admit multiple receiver-native realizations.

## Falsifiers

The hypothesis is weakened or rejected if:

- operation-local models dominate the shared state under matched rate;
- held-out operations fail despite good seen-operation fit;
- renderer or sender identity remains strongly decodable;
- facts-only matches the full state;
- held-sender onboarding requires changing the shared decoder;
- receiver utility does not follow representation qualification;
- query-conditioned text equals or exceeds activation-derived packets at matched rate.
