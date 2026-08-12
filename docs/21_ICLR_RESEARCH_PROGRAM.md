# Frank-EQ ICLR research program

Status: research synthesis and prospective decision map. RC0 remains the only
currently authorized execution.

## 1. Scientific diagnosis from the complete Frank lineage

The accumulated projects do not support a universal continuous coordinate
system as the primary object. They support four sharper separations:

```text
shared/predictable geometry  != receiver-native utility
receiver-native utility      != semantic specificity
oracle packet sufficiency    != sender compilation
single-step readout           != future computational sufficiency
```

The strongest recurring facts are:

1. hidden states often share descriptive and predictive structure;
2. target-state reconstruction and legal vocabulary realization remain brittle;
3. source-local calibration is repeatedly load bearing;
4. structured receiver-native computation is the strongest causal endpoint;
5. independently learned senders fail when asked to enter a founder-private
   gauge or reproduce a learned renderer;
6. the recent Stage-Q screens confound answer calibration, post-query compute,
   and state information.

The correct constructive question is therefore not:

> Can one model's hidden vector be mapped into another model's hidden vector?

It is:

> Can independently trained source-local compilers populate the same externally
> identifiable operational state, at a useful rate, for a frozen executor with
> a declared compute budget?

## 2. Public operational quotients

Let `Z` be the external state space and `K` a family of future operations. Define

```text
z ~_K z'  iff  f_k(z) = f_k(z') for every k in K.
```

The quotient `Q_K = Z / ~_K` is the minimal semantic object needed to support the
registered future operations. It does not preserve distinctions that no future
operation can observe.

### Proposition 1: message lower bound

Any zero-error one-shot public message supporting all operations in `K` must
distinguish every class in `Q_K`. Therefore its alphabet satisfies

```text
|M| >= |Q_K|,
```

and its worst-case rate is at least

```text
R >= ceil(log2 |Q_K|).
```

This is the relevant lower bound for interoperability. Hidden dimension, CKA,
and reconstruction R2 do not determine it.

### Proposition 2: separating-basis factorization

A public test bank `B` is separating when equality under all basis tests implies
operational equivalence under `K`. Then every target operation factors through
the public basis:

```text
f_k = g_k o T_B.
```

The proof follows because `T_B` is injective on `Q_K`; `g_k` is well defined on
its image.

### Proposition 3: query-conditioned rate

For one known future operation `k`, the relevant quotient is

```text
Q_k = Z / ~_k,
```

so a query-conditioned packet needs only

```text
R_k >= ceil(log2 |Q_k|).
```

A universal reusable packet and a query-conditioned packet therefore occupy
different rate points. Their utility must not be compared without amortizing the
number of future queries served by the reusable state.

### Approximate interfaces

For a probabilistic basis estimate `b_hat`, an operation executor `g_k`, and
metric `d`, local error obeys

```text
d(g_k(b), g_k(b_hat)) <= L_k d(b, b_hat)
```

when `g_k` is `L_k`-Lipschitz on the relevant region. Composition operations can
have much larger sensitivity than direct lookup. This predicts a graded
rate/calibration requirement even when the same basis is semantically complete.

## 3. The full rate--compute object

The useful interface frontier has three resources:

```text
U(C_source, R_message, C_consumer).
```

- `C_source`: query-blind producer computation used to form a reusable state;
- `R_message`: communicated public state or selected packet;
- `C_consumer`: post-reveal computation used to execute a future operation.

Previous Frank projects usually changed all three implicitly or held them at
unexamined values. Stage-Q fixed `C_source` near zero, used no explicit message,
and evaluated an immediate one-token consumer. RC0 varies the response/consumer
contract and public rate while holding producer preparation fixed.

A later producer-preparation audit is warranted only if RC0 shows that public
basis information is inaccessible without query-time computation. It must
compare query-blind generated preparation against an equal-token pre-pause
control before capture.

## 4. Why the old graph conclusion was not identified

The earlier 16-operation panels contained two instances per family and held out
one. Each reported family result therefore depended on a single operation
instance, argument tuple, and polarity. The claim that multi-edge depth was the
binding wall was stronger than the design supported: lookup and inverse have the
same one-edge complexity yet moved in opposite directions in the 8B screen.

RC0 uses four registered instances per family and scores the same frozen
instances across independent validation worlds. It still remains a development
localization study, not a broad benchmark claim.

## 5. RC0 and the next architecture

RC0 tests an exact semantic basis: every directed edge. The basis is deliberately
simple because it provides a theorem-backed separating set and a deterministic
executor. Its interactive source probes are an upper bound on what a future
one-shot compiler could recover.

A positive RC0 result supports the following Stage-A v3 architecture:

```text
source token/layer/KV state before query
        |
        v
model-local typed slot extractor E_m
        |
        v
public basis b_hat in [0,1]^d
        |
        +--> frozen deterministic executor g_k
        |
        +--> optional receiver-native rendering
```

### Model-local compiler

Each model receives an independent compiler. The recommended extractor is a
small set of typed slot queries attending over source token positions and
selected layers, rather than an MLP over the final token:

```text
slot(i,j) query -> cross-attention over token/layer memory -> edge probability
```

Entity anchors are public or pointer grounded. No hidden chart, fact head,
router, or residual decoder is shared between model families.

### Two output namespaces

```text
semantic basis:
  externally correct public facts/core tests

behavioral basis:
  frozen source model's own future response distribution
```

The semantic channel enables correct external execution. The behavioral channel
tests whether the compiler preserves the model's own reusable causal state even
when that state is wrong. They require separate losses and claims.

### Required controls

A latent-space contribution is authorized only if the hidden-state compiler
beats:

- deterministic/raw-text parsing where available;
- token-ID and embedding-only slot extractors;
- final-token residual probes;
- direct complex-operation prediction;
- historical continuous quotient transport;
- rate-matched natural-language summaries;
- oracle public basis.

On the current explicit graph text, a parser is an intentionally unbeatable
semantic baseline. Stage-A v3 is therefore an architecture/mechanism canary, not
the final application result.

## 6. The natural task required for the final paper

A competitive paper needs a domain where the useful operational state is not
merely a lossless reformatting of visible input. The final task should satisfy:

1. observations arrive sequentially or noisily;
2. the sufficient state requires aggregation, filtering, or closure;
3. multiple future operations are unknown when the state is formed;
4. an external solver or simulator defines public core tests;
5. token/text baselines are rate matched;
6. a new sender can be onboarded without retraining the executor.

The strongest candidate is a controlled latent-state filtering task:

```text
observation history -> belief/predictive state -> unknown future tests
```

Examples include finite POMDP belief tracking, hidden automata, noisy relational
worlds, and solver-backed logical closure. Predictive-state representation theory
provides the relevant object: a finite set of core tests whose predictions form
a sufficient state without requiring recovery of an arbitrary latent variable.

A natural-language proof domain can follow as Stage 1 once the controlled
mechanism passes. ProofWriter-style private facts or solver-backed logic grids
allow exact public facts, proof certificates, matched wrong worlds, and native
receiver execution.

## 7. Prospective paper structure

### Thesis

> Public operational bases, not aligned hidden coordinates, determine whether
> independently trained language models can interoperate. Utility lies on a
> producer-compute/message-rate/consumer-compute frontier.

### Main contributions

1. formal operational quotient, message lower bound, and separating-basis
   factorization;
2. a prospective ladder separating geometry, basis recovery, composition,
   sender establishment, and receiver execution;
3. model-local typed compilers into an externally identifiable predictive state;
4. held-sender establishment with a frozen executor;
5. rate--compute curves and matched text/token/direct/continuous baselines;
6. a compact retrospective map of prior Frank negatives as motivating evidence,
   not pooled confirmatory data.

### Positive evidence required

The paper should not be submitted as a constructive ICLR paper unless a frozen
confirmation shows:

- public-basis recovery above all token/text controls where hidden state is
  claimed to help;
- composition gain over direct prediction;
- positive specificity against matched wrong states;
- held-sender retention at least 0.80;
- no receiver retraining;
- stable worst-model and worst-complexity results;
- a useful rate advantage or amortized multi-query advantage;
- receiver-native execution on at least one formal natural-language domain.

## 8. Kill rules

Stop the constructive line when any of the following remains true after the
registered stage designed to address it:

- the source cannot expose a separating basis under calibrated semantic probes;
- a token/text extractor matches the hidden-state compiler everywhere;
- public composition does not beat direct operation prediction;
- a held sender requires founder or receiver retraining;
- gains disappear under matched wrong-state controls;
- only average utility is positive while worst-tail harm worsens;
- the final method reduces to pair-specific target-state translation.

A negative RC0 does not justify another broad architecture sweep. A positive RC0
does not by itself justify a latent-interface claim. It determines whether the
one-shot typed-compiler experiment is scientifically warranted.

## 9. Current next action

RC0 is complete and adopted; Stage-A v3-2 is a consumed negative. Stage M0 also
completed and is adopted with diagnosis
`OPERATION_CLOSED_EVENTS_NOT_READABLE`. Sparse joint public events beat the
cross-fitted direct protocol but not the historical marginal/independence
executor, and required high-order event groups fail readout.

The registered stop rule closes the current graph/source line. No successor
compiler, held sender, receiver work, claim-bearing role, or cluster executable
is authorized. The next research action, if any, is to formulate a new
task-level scientific question where useful state must be computed by filtering
or closure rather than copied from explicit graph text. That question requires
fresh roles and a separately frozen protocol before any implementation or run.
