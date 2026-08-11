# Stage R / RC0: rate--compute operational-basis audit

Status: **completed development pass; adopted evidence**.
Authority: drafting one fresh Stage-A v3 registration only. No RC0 rerun,
claim-bearing test access, v3 launch, latent compiler training, receiver execution,
or scientific claim is authorized.

## 0. Outcome

The frozen Olivia audit returned `PUBLIC_BASIS_COMPOSITION_SUPPORTED`. Across
3,712 hard-family validation predictions, compiled Brier is `0.0408`, versus
`0.2035` for the training-selected direct protocol and `0.2181` for the prior;
the lower-95 gains are `0.1542` and `0.1661`. Every model/complexity basis and
composition group passes, every hard family is positive over both baselines, and
the independent executor audit has zero hard-oracle mismatches.

The original capture completed all response rows and failed before outcomes due
to a mechanical aggregation API regression. A fresh, hash-bound artifact-only
recovery executed no model inference. Both verifiers and an independent
recomputation pass. See `evidence/real_stage_r_olivia_rc0/AUDIT.md`.

Semantic sequence likelihood improves over immediate answer-token scoring, but
generated reasoning is worse than the matched 32-token pause control. Runtime
basis probing therefore remains an information-and-composition upper bound, not
a one-shot compiler or communication result.

## 1. Why the completed screening does not localize the scientific wall

The Stage-Q screens evaluated the probability of the registered true answer by
reading a single false/true token pair immediately after the operation query.
The state was formed before the operation, but the branch allowed no additional
autoregressive scratchpad tokens. The reported source-competence quantity therefore
mixed three different objects:

1. **answer-channel calibration** -- whether the model maps its internal evidence to
   the selected A/B token pair with a useful scale and orientation;
2. **post-reveal computation** -- whether the operation requires additional causal
   token positions after it is revealed;
3. **state sufficiency** -- whether the query-blind state contains enough information
   to support the operation at all.

The observed family pattern is diagnostic but not conclusive. At 8B, inverse and
reciprocity pass while mutual, composition, and out-degree comparison remain
strongly negative. That is compatible with an insufficient public state, but it is
also exactly the pattern expected when a one-step readout is asked to perform
increasingly compositional Boolean computation.

RC0 changed no latent architecture. It localized these three factors before another
shared-state protocol could be drafted.

## 2. Revised mathematical object

Let `h` be a query-blind source state, `k` a future operation, and `c` an explicit
post-reveal compute protocol. The relevant signature is compute-indexed:

```text
Sigma_M(h; k, c) = p_M(y | h, k, c).
```

The historical branch is one point on this surface: `c = immediate answer-token
readout`. RC0 also evaluates semantic sequence likelihood, bounded generated
reasoning, and a matched fixed-pause control.

### 2.1 Operational equivalence

For a public operation bank `K`, two external states are operationally equivalent
when every registered operation agrees:

```text
z ~_K z'  iff  f_k(z) = f_k(z') for all k in K.
```

A set of public probes `B` is a **separating operational basis** when

```text
T_B(z) = T_B(z')  implies  z ~_K z'.
```

### 2.2 Factorization proposition

If `B` separates the operational equivalence classes, every target operation
factors through the basis:

```text
f_k = g_k o T_B.
```

Proof: `T_B` is injective on the quotient `Z / ~_K`; define `g_k` on the image of
`T_B` by choosing any representative. Separation makes the definition
representative-independent.

For an `n`-entity closed directed graph, the `n(n-1)` non-diagonal edge indicators
form an exact separating basis for all graph operations used here. RC0 uses 12
public coordinates for four entities and 30 for six entities. The deterministic
executor is parameter-free and is tested against the formal oracle for every
registered structural operation.

This basis is a public ABI rather than another private latent gauge: coordinate
`(i,j)` always means the same directed edge for every model.

## 3. Frozen hypotheses

RC0 tests three hypotheses separately.

### H1: answer-channel distortion

A semantic sequence-likelihood response, followed by train-only affine log-odds
calibration, is more useful than the historical immediate A/B token probability.
The calibration slope is allowed to be negative because a stable answer-label
inversion is a correctable local interface defect, not absence of state information.

### H2: contentful post-reveal computation

Generated reasoning tokens improve hard-operation Brier score more than the same
number of fixed pause tokens. Both conditions use the identical query, final cue,
and token budget. A positive paired interval identifies contentful sequential
computation; a pause gain without a reasoning-over-pause gain identifies extra
workspace rather than reasoning content.

### H3: separating-basis sufficiency

Calibrated direct-edge probes recover a public graph basis from which a frozen
executor predicts the hard structural operations. The primary comparison is against
both the operation prior and a direct baseline selected on training worlds from the
four registered source protocols.

## 4. Frozen design

### Models

```text
Qwen/Qwen3-4B  revision 1cfa9a7208912126459214e8b04321603b3df60c
Qwen/Qwen3-8B  revision b968826d9c46dd6066d109eabc6255188de91218
```

Both are founders in this development audit. There is no held sender.

### Panels

- entity counts: `{4, 6}`;
- worlds per complexity: `96`;
- renderer views: `2`;
- target operations: `32`, balanced within the eight historical families;
- public basis: every non-diagonal directed edge;
- development split: `70%` calibration/selection worlds, remainder validation;
- seed: `20260820`;
- no claim-bearing test role.

Density and reciprocity remain descriptive controls. The promotion decision is based
on the structural basis and hard composition families, not on the explicitly
rendered global tags.

### Causal branch contract

- corrected `chat_turn` prefix;
- exact token-prefix continuity;
- exclusive cloned-KV branching;
- query-exclusive cache slots grouped only by protocol and token length in the
  frozen batches of eight; batching changes scheduling, not causal access;
- no exact-replay fallback;
- operation revealed only after the prefix state has formed.

### Response protocols

1. `answer_token`: historical immediate A/B readout;
2. `sequence`: normalized false/true candidate-sequence likelihood;
3. `reason`: 32 greedy generated scratchpad tokens, then the same final cue;
4. `pause`: 32 fixed pause tokens, then the same final cue.

The basis probes use semantic sequence likelihood. Target operations evaluate all
four protocols. Calibration and direct-protocol selection use training worlds only.
Validation worlds are untouched until the frozen calibrators and protocol choices
are fixed.

Basis calibration is model-local and coordinate-specific: each
`model x complexity x directed-edge slot` receives one affine log-odds map fitted
across training worlds and both renderers. Direct target-response calibration is
pooled by `model x complexity x operation family x response protocol`. This keeps
the typed public coordinates identifiable while preventing validation outcomes or
renderer identity from selecting a readout map.

## 5. Public executor

Given calibrated edge probabilities `p_ij`, the frozen executor uses:

```text
lookup(i,j)       = p_ij
inverse(i,j)      = p_ji
mutual(i,j)       = p_ij p_ji
compose(i,j)      = 1 - product_m (1 - p_im p_mj)
outdegree(i) > outdegree(j)
                   = exact Poisson-binomial difference probability
counterfactual    = set the registered added edge to one, then execute compose
```

Negative-polarity operations return `1-p`. The independence interpretation is an
explicit probabilistic compiler assumption; exact binary basis inputs reproduce the
formal oracle by construction.

## 6. Metrics and paired units

The independent unit is a world. Renderer, operation, and protocol rows are averaged
within world before a 2,000-replicate bootstrap.

Reported quantities include:

- raw and calibrated Brier score;
- Brier gain over the train-world operation prior;
- balanced accuracy and ECE;
- sequence-minus-answer-token gain;
- reasoning-minus-pause gain;
- basis readout by model and complexity;
- compiled-minus-prior and compiled-minus-direct gains;
- exact-oracle executor check;
- one-, two-, four-, and eight-bit basis rate frontiers;
- generated token and source-query accounting.

The rate figures are descriptive in RC0. Interactive basis probing is not yet a
one-shot latent compiler and must not be presented as communication efficiency.

## 7. Frozen promotion gate

RC0 supports drafting one Stage-A v3 protocol only when all of the following pass:

1. every model x complexity basis group has lower-95 Brier gain at least zero;
2. every group has balanced accuracy at least `0.60`;
3. compiled hard-family predictions have lower-95 gain over the prior at least zero;
4. compiled hard-family predictions have strictly positive lower-95 gain over the
   training-selected direct protocol, for the aggregate and every model x complexity
   group.

Answer-channel and reasoning-over-pause effects are diagnostic. They identify the
failure mechanism but do not rescue a failed public basis.

A pass authorizes **protocol drafting only**. It does not authorize a fresh Stage-A
run, a hidden-state claim, or receiver execution.

## 8. Decision tree

### Basis readout fails

The current source/task pair does not expose an informationally complete low-complexity
basis under the tested branch protocols. Stop the graph line or redesign the task on
development data. Do not train another cross-model latent.

### Basis passes, composition fails against the prior

The edge probabilities are insufficiently calibrated or the independent-edge public
executor is misspecified. Inspect joint uncertainty and structured calibration; do
not add a larger private code.

### Composition beats the prior but not direct computation

The public basis is valid but offers no functional advantage in the current regime.
It remains useful as a diagnostic, not as the paper's constructive method.

### Basis composition beats direct computation

Draft Stage-A v3 with fresh worlds and a new unopened held sender. The architecture
must use model-local token/slot compilers into the typed edge basis, not runtime
interactive probing. Behavioral self-future prediction and oracle semantic grounding
remain separate heads. Required baselines include token-only/text extraction, direct
complex-operation prediction, the historical continuous quotient, oracle basis, and
rate-matched natural-language communication.

## 9. Paper-level interpretation

The intended positive paper is not "another shared latent space." Its candidate thesis
is:

> Cross-model interoperability requires a public separating operational basis and an
> explicit consumer-compute budget. Hidden-coordinate alignment and immediate readout
> confound information, calibration, and computation.

The accumulated Frank results motivate this thesis: high representation prediction
has repeatedly failed execution, while receiver-native structured computation is the
strongest positive endpoint. RC0 was the first prospective experiment to isolate
that synthesis rather than adding another translator.

The nearest current work, *Hidden APIs in Language Models* (arXiv:2607.27617), tests
whether reusable causal interfaces exist within individual models using forked
futures and architecture competition. Frank-EQ's intended novelty is different:
identifiable public coordinates, cross-model sender establishment, and the
rate--compute frontier. Work on pause tokens (arXiv:2505.21024) makes the matched
pause control scientifically necessary because extra token positions can themselves
increase transformer expressivity.
