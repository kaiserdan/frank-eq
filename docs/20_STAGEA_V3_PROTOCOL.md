# Stage-A v3 protocol registration: one-shot typed-basis compiler

Status: **frozen registration; implementation and one representation run
authorized after this registration is committed and validated**.

Date: 2026-08-12
Protocol: `stagea-v3-2`
Config: `configs/stagea_v3/real_olivia_v3.yaml`

## 1. Authority and scope

The adopted RC0 diagnosis is `PUBLIC_BASIS_COMPOSITION_SUPPORTED`. RC0 showed
that interactive, post-capture queries can recover a typed directed-edge basis
whose deterministic composition beats the source models' training-selected
direct protocols. RC0 did not implement a one-shot compiler.

The user explicitly authorized the sequential next steps on 2026-08-12, with
results documented and committed along the way. This registration therefore
authorizes implementation and exactly one outcome-bearing Stage-A v3
**representation** run after the protocol and implementation are separately
committed and all validators pass.

This registration does not pre-authorize receiver execution. A passing machine
decision may authorize drafting a receiver protocol. That protocol must be
frozen and committed before a receiver run.

## 2. Frozen question

> Can a model-local compiler map the query-blind, all-token residual trajectory
> from one source-prefix forward pass into the same typed public directed-edge
> basis used by RC0, without runtime basis interrogation, while preserving
> semantic facts, the source model's behavioral edge signature, hard-operation
> composition, renderer transfer, and onboarding of one unopened sender?

The primary object is a public packet relative to a consumer-compute contract:

```text
query-blind prefix state
    -> model-local token/slot compiler
    -> typed edge probabilities (i,j)
    -> quantized public packet
    -> frozen deterministic graph executor
```

The compiler receives no operation, target, candidate answer, receiver state,
or test label. The operation is supplied only to the frozen executor after the
packet exists.

## 3. Claims this experiment cannot establish

The controlled graph is written explicitly in the source prefix. A deterministic
renderer-aware text parser can recover the oracle graph and is registered as an
oracle-like control. Consequently, even a full pass does not establish that
hidden activations contain more information than text, or that activation
communication beats an optimal text compiler.

A pass can establish only:

- one-shot compilation from the registered residual trajectory into identifiable
  public coordinates;
- separation of semantic and behavioral state channels;
- deterministic composition under explicit rate and compute accounting;
- model-local onboarding of one unopened same-family sender;
- qualification to draft a receiver protocol on fresh worlds.

It cannot establish a cross-family interface, natural-task generality, receiver
utility, a safety claim, or a paper-level communication advantage.

## 4. Models and roles

Founders are the two RC0 sources. They are development-exposed and may be used
to implement and validate the compiler:

```text
Qwen/Qwen3-4B  1cfa9a7208912126459214e8b04321603b3df60c
Qwen/Qwen3-8B  b968826d9c46dd6066d109eabc6255188de91218
```

The held sender is:

```text
Qwen/Qwen3-14B  40c069824f4251a91eefaf281ebe4c544efd3e18
```

The held revision was selected from the official Hugging Face repository on
2026-08-12 for checkpoint availability, shared exact-prefix chat semantics, and
GH200 feasibility. It has not been queried on Frank-EQ worlds or operations.
Checkpoint download, tokenizer loading, and a dummy prefix-continuity smoke do
not expose task outcomes. The held model may first see registered train worlds
only after founder checkpoints and the public executor are frozen.

All three models are Qwen3 dense checkpoints. Same-family establishment is an
explicit limitation; no cross-family claim is permitted.

## 5. Fresh panels and access order

RC0 worlds and all earlier Stage-A/Stage-Q worlds are forbidden. V3 uses
independent deterministic panels for each entity count in `{4, 6}`:

| Role | Worlds per complexity | Seed | Permitted use |
|---|---:|---:|---|
| train | 80 | 2026081201 | compiler fitting, train-only calibration and direct-protocol selection |
| validation | 24 | 2026081202 | early stopping and pre-test checkpoint selection only |
| test | 32 | 2026081297 | one frozen representation evaluation |

Every model, renderer, coordinate, operation, protocol, compiler seed, and
baseline row for a world inherits that world's role. No row may cross roles.

The implementation must enforce this order:

1. generate/capture founder train and validation panels;
2. fit founder compilers and baselines;
3. write a hash-bound `freeze_manifest.json` containing code, config, panel,
   checkpoint, calibrator, protocol-selection, and executor hashes;
4. generate/capture held-sender train and validation rows and fit only its local
   compilers and local baselines;
5. extend the freeze manifest with held checkpoint hashes;
6. only then generate the test panels and evaluate once.

Test panel files and test labels must not exist before both freeze steps. A test
access ledger records creation time, process stage, and every opened test file.

### v3-2 access repair

The original v3-1 seed `2026081203` was deterministically instantiated by a
unit test while the implementation was still under construction. No checkpoint
or model was loaded and no model outcome existed, but the world/label role was
no longer unopened. V3-1 is therefore void before execution. V3-2 changes only
the test seed to fresh value `2026081297` and requires a consumed, hash-matching
access-ledger grant at the panel generator itself. Unit and repository tests may
exercise train/validation worlds or synthetic role wrappers only; they cannot
instantiate the registered test panel.

## 6. Renderers and operations

Train and validation use two renderer grammars:

- natural closed-world relational sentences;
- an adjacency-table rendering.

Test additionally includes a canonical edge-list grammar that is never captured
or scored during fitting or early stopping. The grammar is frozen in source
before execution. Seen-renderer and unseen-renderer results are separate gates.

Every complexity contains 32 frozen operation instances, four per historical
family:

```text
lookup, inverse, mutual, compose, compare_outdegree,
counterfactual_add, density, reciprocity
```

Density and reciprocity remain printed controls and cannot promote the method.
The hard composition families are `mutual`, `compose`, `compare_outdegree`, and
`counterfactual_add`.

Operation sampling uses the independent frozen seed `2026081213` plus entity
count and is identical across train, validation, and test roles. Role seeds vary
worlds only. This binding was added in a pre-implementation completeness
amendment before any panel or model outcome existed.

## 7. Causal capture contract

- `chat_turn` state formation with `enable_thinking: false`;
- exact token-prefix continuity for every branch;
- no operation, query, target, candidate answer, or label before capture;
- residual stream for every unpadded prefix token at normalized depths
  `0.25`, `0.50`, `0.75`, and `1.00`;
- exact token IDs, attention mask, token spans, prefix bytes, selected layer
  indices, observed hidden width, and SHA-256 recorded per capture;
- bfloat16 model execution and float32 serialized compiler input;
- exclusive cloned-KV branches for behavioral teachers and direct baselines;
- no replay fallback or mixed cache modes;
- branch batches have query-exclusive cache slots;
- one prefix forward per model/world/renderer cell.

A negative result applies only to this all-token residual capture. It must not be
generalized to the complete KV/runtime state.

## 8. Separate model-local compilers

Each source has two independent compiler modules with no shared trainable
parameters:

1. `semantic`: predicts oracle edge facts;
2. `behavioral`: predicts the frozen source model's semantic-sequence
   probability for the corresponding post-reveal edge query.

Each module consumes all registered token-layer vectors. A model-local linear
projection maps each layer to width 192. Learned depth and normalized-position
embeddings are added. Fixed coordinate order supplies one learned local query
slot for each typed `(source,target)` edge. Two cross-attention blocks with six
heads and a 384-wide feed-forward layer resample the token trajectory into edge
slots. A slot-local head emits one logit per coordinate.

Coordinate order, packet schema, quantizer, and graph executor are public and
shared. Input projections, coordinate queries, attention blocks, and heads are
model local. Entity count uses an explicit mask; a six-entity compiler contains
all 30 slots and the four-entity condition activates its canonical 12-slot
subset.

The channels have separate parameters, optimizers, checkpoints, losses, and
metric namespaces. No unnamed joint loss may merge behavioral and semantic
targets.

## 9. Training and held onboarding

Three fixed compiler seeds (`211`, `223`, `227`) are trained. The primary
prediction is the equal-weight mean logit across all three seeds; no best-seed
selection is permitted. Individual seeds are reported.

Semantic modules use coordinate BCE plus a `0.10` same-world renderer-consistency
penalty. Behavioral modules use soft-label Brier loss plus the same consistency
weight. Training is world-balanced, uses AdamW, learning rate `3e-4`, weight
decay `1e-4`, gradient norm `1.0`, at most 120 epochs, and patience 20. The
frozen validation criterion is channel Brier plus `0.10` renderer variance.

No post-hoc calibration is applied to the primary compiler. Direct source
protocols use train-only model/complexity/family calibration and train-only
protocol selection. Validation selects checkpoints only; test never selects or
fits anything.

Held onboarding starts from random local compiler weights and uses held train
and validation worlds. Founder compilers, public coordinate semantics,
quantizer, executor, gates, and baseline definitions remain frozen.

## 10. Registered baselines

Every test world must include:

1. train-world edge prior;
2. parameter-matched token-ID resampler with no model activations;
3. historical final-token public MLP at the same four depths;
4. historical continuous private quotient with a learned operation head;
5. direct source prediction, train-selected from sequence, 32-token reason, and
   32-token pause protocols;
6. RC0-style interactive semantic edge queries;
7. deterministic renderer-aware text parser;
8. rate-matched canonical natural-language edge packet;
9. oracle binary edge basis;
10. shuffled-world, wrong-world, and zero packets.

The token-ID and final-token baselines use the same train/validation roles,
seeds, parameter budget within 5%, early-stopping rule, and quantizer as the
primary compiler. Assisted use of target hidden states or receiver gradients is
forbidden.

## 11. Rate and compute accounting

The primary packet uses four bits per edge probability: 48 payload bits for
four entities and 120 for six. Entity count, schema/version, quantizer scale,
checksum, and framing overhead are reported separately. One-, two-, four-, and
eight-bit frontiers are all retained.

For every condition report:

- prefix forwards;
- post-capture source queries;
- generated and pause tokens;
- compiler parameters, FLOPs estimate, and wall time;
- serialized payload and framing bits;
- deterministic executor operations and wall time;
- amortized cost across 1, 4, 16, and 32 target operations.

The primary compiler uses zero post-capture source queries. RC0 interactive
basis uses 12 or 30. Direct prediction uses one query per target. The text parser
uses prefix bytes plus parser compute. These are not interchangeable rates.

## 12. World-grouped evaluation

All confidence intervals use 2,000 bootstrap replicates over worlds. Renderer,
coordinate, operation, compiler-seed, and protocol rows are averaged within
world before resampling. Report aggregate, model-by-complexity,
model-by-complexity-by-renderer-role, and hard-family strata.

Primary metrics are Brier score, Brier gain over the training-world prior,
balanced accuracy, ECE, correct-minus-hardest-wrong-world margin, cross-model
same-world retrieval, sender identity leakage, quantization retention, compiled
hard-operation gain over prior/direct baselines, and held-sender retention.

Behavioral metrics compare the behavioral compiler to the frozen source's edge
query probabilities. Semantic metrics compare the semantic compiler and its
composed operations to external oracle labels. They are never pooled.

## 13. Frozen representation gate

All integrity checks and all conditions below are conjunctive:

1. every model × complexity semantic basis group has Brier-gain lower 95% bound
   strictly above zero and Brier at most `0.10`;
2. every model × complexity semantic balanced-accuracy lower bound is at least
   `0.75`;
3. the unseen-renderer semantic Brier-gain lower bound is strictly above zero
   for every model × complexity group;
4. every model × complexity behavioral Brier-gain lower bound over the
   train-world coordinate prior is strictly above zero;
5. compiled hard operations beat the prior and training-selected direct
   protocol with strictly positive lower bounds, both aggregate and for every
   model × complexity group;
6. every hard family has strictly positive compiled gain over both baselines;
7. the all-token activation compiler beats the parameter-matched token-ID and
   final-token public baselines with strictly positive aggregate lower bounds,
   and is non-inferior to the continuous quotient (`lower95 >= 0`);
8. cross-model same-world retrieval lower bound is at least `0.75`,
   correct-minus-hardest-wrong-world margin lower bound is at least `0.05`, and
   sender-ID accuracy over one-third chance is at most `0.20`;
9. held-sender semantic and behavioral gains retain at least `0.70` of the mean
   founder gain at each complexity;
10. four-bit quantization retains at least `0.95` of float compiled gain and
    still clears every prior/direct composition gate;
11. exact binary oracle basis has zero hard-operation mismatches;
12. all baselines, rate strata, compute strata, access ledgers, hashes, and
    protected authorization fields are complete.

The deterministic text parser and oracle basis are ceilings, not promotion
comparators. Their presence prevents a hidden-over-text claim.

## 14. Machine decision and stop tree

The reducer emits exactly one diagnosis:

```text
INVALID_STAGEA_V3_RUN
ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED
BEHAVIORAL_STATE_NOT_QUALIFIED
NO_ACTIVATION_SPECIFIC_ADVANTAGE
NO_ONE_SHOT_COMPOSITION_ADVANTAGE
HELD_SENDER_NOT_ESTABLISHED
STAGEA_V3_REPRESENTATION_QUALIFIED
```

Only `STAGEA_V3_REPRESENTATION_QUALIFIED` may set
`receiver_protocol_draft_authorized=true`. It must keep receiver execution,
new receiver-world access, scientific claim, and paper-claim authorization
false.

A valid miss is terminal for `stagea-v3-2`. Do not tune gates, layers, seeds,
models, renderers, compiler width, losses, or baselines after test access. An
engineering failure before test creation may be repaired under a fresh source
and job only if the failed run remains immutable and a hash-bound provenance
record proves no test outcome existed. Any failure after test creation is a
consumed outcome and cannot be retried as the same registration.

## 15. Required artifacts

```text
config.yaml
run_manifest.json
workflow_status.json
access_ledger.json
train_panel_manifest.json
validation_panel_manifest.json
freeze_manifest.json
held_onboarding_manifest.json
test_panel_manifest.json
models.json
capture_validation.json
compiler_checkpoints_manifest.json
training_summary.json
baseline_manifest.json
predictions_manifest.json
rate_compute.json
metrics.json
decision.json
artifact_manifest.json
independent_audit.json
```

Generated captures, response rows, checkpoints, and operational state remain
outside Git. Adopt only a compact hash-verified evidence package after fetching,
project verification, independent recomputation, and strata review.
