# Independent audit of the first real Stage-A outcome

Snapshot: 2026-08-10  
Run: `frank-eq-stagea-devg-v2`  
LUMI job: `20942127`  
Frozen source archive: `bc2bff426e1c7f5a438c4004abf3e51999e3fbf2e3f13c25c6b11821d65a4751`

## Bottom line

The run is an **engineering-valid, scientifically negative result for the exact
Stage-A v1 pipeline**:

```text
three last-token residual captures
→ model-local chart
→ shared fact/residual heads
→ frozen graph interrogator
```

The decision `STOP_OR_REVISE_STAGE0` is correct and remains immutable. The run
does not establish that future-defined operational quotients are absent from
LLMs, and it does not yet localize the failure uniquely to hidden-state capture.

## What the outcome establishes

The frozen pipeline fails six conjunctive gates:

- held-out signature Brier upper 95% bound `0.1978 > 0.18`;
- fact accuracy lower 95% bound `0.4954 < 0.70`;
- cross-model retrieval lower 95% bound `0.0972 < 0.30`;
- correct-minus-hardest-wrong margin lower 95% bound `-0.0791 < 0.03`;
- held-sender retention `0.4717 < 0.70`;
- model-ID leakage over chance `0.6528 > 0.25`.

The public fact coordinates carry almost no useful edge information. Rebuilding
the frozen panel and split from the committed generator/config gives a
global-majority test fact accuracy of `0.5278`; the trained quotient reaches
`0.5296`, a gain of only `0.0019`.

The quotient does improve oracle-signature Brier over an operation-wise
training prior (`0.1729` versus reconstructed prior `0.2080`), but the source
models' own post-reveal behavior is weak: branch accuracy to the oracle is
`0.4392`, below the operation-prior classification baseline `0.6354`.

## What the passing checks do not establish

- Renderer cosine `0.9925` is not a non-collapse certificate. The simultaneously
  negative wrong-world margin and near-perfect model-ID probe show that a code
  can be renderer-stable while remaining world-inspecific and model-specific.
- Quantization retention `0.9962` only shows that quantization preserves the
  already-failing code.
- The reported operational-residual gain is not evidence for an irreducible
  hidden operational state. The prefix explicitly states the density and
  reciprocity labels, and these same two declared labels are the residual
  targets. The comparison therefore largely tests extraction of explicit
  global tags while zeroing them in the facts-only arm.

## Why the current localization is not identified

The adopted prose says that per-edge facts are not linearly readable from the
captured state. The executed model did not test that proposition:

1. the fact head is a nonlinear MLP;
2. it is optimized jointly with signature, residual, invariance, contrastive,
   adversarial, variance, workspace, and quantization objectives;
3. there is no independent model-by-model, layer-by-layer linear or nonlinear
   readability upper bound;
4. the capture contains only the final-token residual at three depths, while
   the actual branched causal state is the complete prefix KV cache;
5. the fact and residual heads are shared across model families, so held-sender
   onboarding must enter a founder-induced private bottleneck gauge before it
   reaches public coordinates.

The result is therefore compatible with capture insufficiency, pooling
insufficiency, shared-head gauge, sample/parameter mismatch, objective
interference, or insufficient native task competence.

There is also a target-definition mismatch. The project defines an operational
signature as the model's own future response distribution, but primary training
uses formal oracle outcomes. The cache contains both quantities; the next
diagnostic must separate self-future readability from external semantic
grounding.

## Correct next action

Do not change the v1 gates and do not run another outcome-bearing test variant.
Use only training and validation worlds from the existing cache to measure:

1. direct fact readability by model and capture layer;
2. direct readability of each model's own future branch signature;
3. direct oracle-signature readability;
4. renderer-transfer readability;
5. native branch competence against an operation-wise prior.

The repository command is:

```bash
frank-eq diagnose-real-cache \
  --cache <fetched-run>/runs/cache \
  --out <fetched-run>/runs/diagnostics
```

This diagnostic is explicitly non-promotional. Its outcome determines the
versioned Stage-A v2 hypothesis:

- poor native competence → repair the prompt/task competence contract first;
- native competence but poor self-signature readability → expand capture to
  token-sequence or selected-KV state;
- self-signature readable but facts unreadable → separate operational state
  from semantic grounding;
- raw targets readable but the quotient fails → replace the shared public heads
  with complete model-local compilers and revise the joint objective.

A new outcome-bearing Stage-A v2 run requires a fresh world seed/test role,
pinned checkpoint revisions, a native-competence prerequisite, and a separately
frozen protocol.
