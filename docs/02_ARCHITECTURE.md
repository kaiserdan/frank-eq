# Architecture

## Overview

```text
hidden trajectory H_m
    |
    v
private chart C_m                    model-local, unconstrained gauge
    |
    v
private bottleneck u_m
    |
    +--> grounded fact head -------- externally named coordinates
    |
    +--> operational residual head - bounded public residual
    |
    v
public quotient z = [facts, residual]
    |
    +--> frozen public interrogator - future signature
    |
    +--> query selector ------------ rate-limited packet
    |
    v
future receiver-native executor
```

## Private chart

Each model has its own chart over a fixed hidden-state capture. The current synthetic implementation consumes a multi-layer trajectory with heterogeneous model widths. The chart may use a learned workspace gate and an MLP. Its private coordinates are never compared across models and never serialized.

## Gauge-fixed quotient

The public code consists of signed fact probabilities and bounded residual coordinates. These coordinates are shared because their semantics are fixed by the task registry, not because private model representations are forced to match.

In the synthetic implementation:

```text
z_fact = 2 * sigmoid(fact_logits) - 1
z_residual = clip(residual / scale, -1, 1)
```

## Frozen public decoder

Operation descriptors contain a frozen coefficient block over the public state basis. The decoder has no learned parameters. This ensures that held-out operation instances are executable without updating the model and prevents a learned shared decoder from absorbing model-specific gauge.

The real-model version may use a frozen solver, canonical evaluator, registered probe bank, or deterministic interpreter. It must be frozen before claim-bearing outcomes.

## Invariance and separation

Training includes:

- renderer-group dispersion loss;
- cross-model same-world dispersion loss;
- supervised world contrastive loss;
- model-ID gradient reversal;
- fact and residual supervision;
- operation-signature supervision;
- code variance floor;
- optional sparse workspace regularization.

Invariance is never sufficient by itself. Wrong-world separation and held-out operation fidelity are required jointly.

## Held-sender onboarding

Founder charts and the shared public heads are trained first. For the held sender:

- all shared heads and the public decoder are frozen;
- all founder charts are frozen;
- only the held sender's private chart is trainable;
- source-side public labels may be used;
- receiver outcomes are unavailable.

This is the Stage-0 analogue of protocol establishment.

## Packet

`FRANK-EQ/OPERATIONAL-PACKET/1` contains:

- task family and query operation ID;
- quantized grounded facts;
- selected public future probes;
- quantized uncertainty;
- deterministic canonical serialization;
- SHA-256 checksum.

State formation remains query blind. Query conditioning enters only during probe selection and compression.

## Why no target hidden decoder

A deterministic hidden-state decoder can average multiple valid receiver realizations into an invalid off-manifold state. Frank-EQ instead transmits public operational constraints and reserves receiver-native realization for the receiver stage.
