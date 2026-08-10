# Real-model implementation plan

## Objective

Test whether future-defined operational quotients exist in real frozen LLMs before attempting cross-model receiver execution.

## Stage A: operation-bank and cache canary

Use a controlled formal task with exact world state and branching operations. The existing CTSI causal worlds or a solver-backed proof task are suitable.

Freeze 8–16 operations such as:

- relation lookup;
- comparison;
- composition;
- verification;
- contradiction detection;
- counterfactual update;
- alternate proof;
- next inference;
- confidence/calibration.

For every model/world/renderer:

1. present only the world prefix;
2. capture selected hidden layers before any operation appears;
3. terminate or freeze the source state;
4. branch the cached state into every registered operation;
5. store canonical outcome distributions and exact operation descriptors;
6. audit `t_reveal > t_capture`;
7. group all branches by world.

## Candidate representations

Evaluate under one common harness:

1. raw hidden state;
2. pairwise affine/Ridge alignment;
3. strongest prior canonical latent;
4. Bary-style predictable component;
5. ExplainFrank semantic/predictive factors;
6. grounded facts only;
7. Frank-EQ facts plus operational residual;
8. operation-local encoders;
9. oracle world state.

## Model roster

Minimum:

- two founder sender families;
- one held sender family unopened during public-decoder development;
- two renderers per world;
- three chart seeds.

Use small checkpoints for Stage A. Receiver-scale experiments are premature until the quotient gate passes.

## Source capture

Start with a small census-informed capture:

- residual stream at normalized depths around 0.35, 0.60, and 0.85;
- optional selected K/V at one or two depths;
- exact byte/span metadata;
- no all-layer/all-KV default.

Compare text-only and activation-input charts under matched parameter count and serialized rate.

## Public decoder

The operation decoder must be frozen and external to model coordinates. Depending on the task, use:

- a deterministic formal solver;
- canonical operation coefficients;
- a frozen public probe/interrogator;
- exact proof/outcome labels.

Do not use a learned target hidden-state decoder as the primary executor.

## Stage B: receiver-native execution

Only after Stage A passes:

```text
source quotient
→ query-conditioned packet
→ deterministic public rendering/tool schema
→ frozen receiver-native reasoning
```

The receiver path must use no source activation and no pair-specific learned weights. A learned receiver port may be included only as a separate assisted baseline.

## Recommended immediate implementation task

Add a backend adapter that emits `FutureSignatureRecord` objects from two small Hugging Face checkpoints and the frozen CTSI operation panel. Preserve `src/frank_eq/contracts.py` unchanged and write a standalone cache validator before any training code.
