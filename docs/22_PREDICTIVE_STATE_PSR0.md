# PSR0: public predictive state under grammar and history-length shift

Status: **frozen development-only protocol; not yet executed**.  
Authority: one Olivia audit may run. No held sender, claim-bearing test split,
receiver execution, scientific claim, or paper claim is authorized.

## 1. Why Stage-A v3 is terminal but the research question is not

Stage-A v3-2 tested a one-shot model-local compiler from query-blind LLM states
into a public graph-edge basis. The run is an engineering-valid negative. The
important pattern is not simply that the aggregate gate failed:

- semantic edge decoding was strong on the two fit renderers;
- the behavioral self-future basis passed;
- public alignment, held-sender retention, quantization, and exact execution
  passed;
- every model failed on the unseen canonical edge-list grammar;
- activations did not beat the matched token-ID compiler;
- aggregate composition improved, but held-model and family strata failed.

This localizes two scientific problems.

First, the graph basis is almost a lossless reformatting of facts printed in the
prefix. A text parser is therefore an unbeatable semantic solution. Hidden-state
advantage is neither necessary nor scientifically natural on this task.

Second, the v3 compiler learned from surface realizations. Its learned slot
queries had no explicit entity-span or permutation-equivariant grounding. The
unseen-renderer reversal is therefore evidence of grammar binding, not evidence
that no reusable operational state exists in the frozen model.

The successor must not tune the consumed v3 registration. PSR0 changes the task
and the mathematical object. It uses only fresh development histories and no
claim-bearing test role. The v3 outcome is retrospective motivation, not a
selection set for PSR0 thresholds or outcomes.

## 2. Revised object: predictive state rather than visible facts

Let a controlled stochastic process have latent state `x_t`, action `a_t`, and
observation `o_t`. A history is

```text
h_t = (a_1, o_1, ..., a_t, o_t).
```

Its exact Bayesian belief is

```text
b_t(x) = P(x_t = x | h_t).
```

For a future test

```text
tau = (a_{t+1}, ..., a_{t+k}; o_terminal),
```

define

```text
p_tau(h_t) = P(o_terminal | h_t, a_{t+1:t+k}).
```

These probabilities are observable predictions of future events. They do not
require assigning a public meaning to an arbitrary latent-state label.

### 2.1 Public core tests

For every test `tau`, let `q_tau` be its state-conditional probability vector:

```text
q_tau[x] = P(o_terminal | x_t=x, a_{t+1:t+k}).
```

Then

```text
p_tau(h) = b(h)^T q_tau.
```

Choose a core test bank `B = {tau_1, ..., tau_r}` and define

```text
Q_B = [q_tau_1 ... q_tau_r].
```

When `Q_B` has full predictive rank `r`, the public core vector

```text
s_B(h) = b(h)^T Q_B
```

is a separating predictive-state basis. For every target test `tau`,

```text
p_tau(h) = s_B(h) W_tau,
W_tau = Q_B^{-1} q_tau.
```

Thus all registered future tests factor through a fixed, externally identifiable
public state. The executor is exact and parameter free.

### 2.2 Why this is the right successor

The sufficient state is no longer explicitly printed. It must be obtained by
filtering a sequence of noisy observations through known stochastic dynamics.
A raw transcript preserves all evidence, but it is not itself the compact
predictive quotient. The experiment can therefore ask whether LLM activations
contain an amortized belief/predictive state that is easier to read than the same
information from matched token-only controls.

## 3. Frozen PSR0 environment

PSR0 uses one four-state controlled hidden Markov model with:

```text
latent states:  S0, S1, S2, S3
actions:        orbit, fold, shift
observations:   amber, blue, coral
initial belief: uniform
```

The transition and emission matrices are fixed in
`configs/predictive_state/real_olivia_psr0.yaml`. Every probability has full
support, avoiding impossible histories and endpoint calibration failures.

Candidate future tests have action horizons one, two, or three. The public core
bank is selected before model execution by a deterministic rank/conditioning
procedure. The frozen result has:

```text
predictive rank:          4
core tests:               4
target tests:             18
core condition number:    1.8311192670
maximum executor L1:      2.4214182125
```

The target bank is stratified across horizons and terminal observations. Exact
oracle core vectors reproduce every target probability to numerical tolerance.

## 4. Fresh development panels

```text
train histories:
  lengths 8 and 16
  128 histories per length
  seed 2026083201

validation histories:
  lengths 8, 16, and 32
  64 histories per length
  seed 2026083202
```

Histories are retained only when posterior entropy and public-core variance fall
inside frozen non-degeneracy bounds. Train and validation IDs are disjoint.
There is no test split.

### 4.1 Renderer contract

Two grammars are available during probe fitting:

```text
narrative
probability-table
```

Validation additionally contains an unseen grammar:

```text
symbolic compact notation
```

All renderer views describe exactly the same automaton and history. They are
paired by history. Length 32 is also validation-only. This gives four mandatory
conditions:

```text
seen
unseen_renderer
length_transfer
joint_ood = unseen renderer at length 32
```

## 5. Causal capture contract

Models:

```text
Qwen/Qwen3-4B
revision 1cfa9a7208912126459214e8b04321603b3df60c

Qwen/Qwen3-8B
revision b968826d9c46dd6066d109eabc6255188de91218
```

Both are development founders. No held model is loaded.

For every rendered history:

1. render the automaton and complete observed history with no future test;
2. form the corrected `chat_turn` prefix;
3. capture final-token residuals at normalized depths `0.25/0.50/0.75/1.00`;
4. preserve the literal frozen KV cache;
5. reveal each core or target future test as a new user turn;
6. verify exact token-prefix continuity;
7. score semantic `false` and `true` candidate sequences.

No operation, candidate answer, target probability, or future-test identity is
available when the captured state is formed. Replay fallback is forbidden.

Runtime future-test queries are development tomography. They are an upper-bound
teacher and direct baseline, not a one-shot communication method.

## 6. Train-only probes and controls

### 6.1 Primary semantic activation probe

For each model, one contextual depth and one ridge penalty are selected using a
history-grouped split of training histories only. A linear probe maps the
query-blind final-token residual to the four public core-test probabilities.

A linear probe is deliberate. PSR0 asks whether a public predictive state is
already geometrically readable. It is not another flexible compiler search.

### 6.2 Token-only control

The complete token sequence is mapped to a deterministic, order-sensitive hash
feature vector with the same width as the activation vector. The same train-only
ridge procedure predicts the public core vector. This control contains the
entire visible transcript but no model activations.

High-dimensional ridge fitting uses the dual system when feature width exceeds
history count, avoiding a hidden-width-squared allocation.

### 6.3 Embedding control

The mean input-embedding representation is probed with the same training and
selection contract. This tests whether contextual processing adds information
beyond the model's lexical embedding table.

### 6.4 Interactive teacher and behavioral state

Semantic future-test scores from the frozen model are calibrated on training
histories only. They provide:

- an interactive core teacher;
- a direct target-test baseline.

A separate behavioral probe predicts the frozen model's own core-test response
probabilities. It is reported separately from oracle-semantic predictive state.

## 7. Exact public executor and rate

The activation, token, embedding, and interactive core predictions are passed
through the same frozen linear executor:

```text
predicted target tests = predicted core state @ W.
```

The executor consumes no model weights and no target-specific training.

The reusable public packet has four probability coordinates. Quantized packets
at two, four, and eight bits per coordinate are reported. Message rates are:

```text
2-bit:   8 payload bits
4-bit:  16 payload bits
8-bit:  32 payload bits
```

Framing and registry costs remain separate from payload rate.

## 8. Paired metrics

The independent unit is the underlying history. Renderer rows for one history
remain paired and are averaged before the 2,000-replicate bootstrap.

For each model and condition, report:

- public-core Brier score;
- core gain over the train-history prior;
- activation gain over token-only and embedding controls;
- interactive-teacher Brier;
- compiled-target Brier;
- compiled gain over target prior and direct interactive prediction;
- behavioral-core Brier;
- fit-to-unseen renderer Brier gap;
- selected layer and ridge;
- two/four/eight-bit oracle execution error.

## 9. Frozen gate

PSR0 supports drafting one fresh, claim-bearing PSR Stage 1 protocol only if all
conditions pass conjunctively.

### Oracle basis

```text
rank = 4
condition number <= 10
maximum oracle executor error <= 1e-10
```

### Semantic readability

For every model and each of
`seen/unseen_renderer/length_transfer/joint_ood`:

```text
activation core gain over prior lower95 >= 0
```

### Activation specificity

For every model on joint OOD:

```text
activation gain over token-only lower95 > 0
wrong-history specificity margin lower95 > 0
```

### Renderer transfer

For every model:

```text
unseen-renderer activation gain over prior lower95 >= 0
absolute seen/unseen core-Brier gap <= 0.02
```

### Length transfer

For every model at validation-only length 32:

```text
activation gain over prior lower95 >= 0
```

### Compositional utility

For every model on joint OOD, both in aggregate and separately for every
registered target-test horizon:

```text
compiled activation gain over target prior lower95 > 0
compiled activation gain over direct teacher lower95 > 0
```

Target-observation strata are also reported as mandatory diagnostics.

A pass authorizes protocol drafting only. PSR Stage 1 execution, held-sender
onboarding, receiver work, test access, and claims remain false.

## 10. Decision tree

### `PREDICTIVE_BASIS_OR_EXECUTOR_INVALID`

The mathematical environment or implementation is invalid. Stop and repair only
engineering defects before any model interpretation.

### `ACTIVATION_PREDICTIVE_STATE_NOT_READABLE`

The frozen LLM states do not expose the registered predictive quotient under a
linear readout. Stop this model/task contract rather than adding another large
compiler.

### `NO_ACTIVATION_SPECIFIC_PREDICTIVE_STATE_ADVANTAGE`

The transcript controls contain everything the activations provide. The result
is scientifically useful but does not support a hidden-state interface paper.

### `PREDICTIVE_STATE_NOT_RENDERER_INVARIANT`

The probe is bound to grammar. A future direction would require explicit
structure grounding; PSR0 itself remains negative and cannot be tuned on its
validation histories.

### `PREDICTIVE_STATE_NOT_LENGTH_TRANSFERABLE`

The readout memorizes the training-length regime rather than representing a
stable filtering state.

### `PUBLIC_PREDICTIVE_STATE_NOT_COMPOSITIONALLY_USEFUL`

The core state is readable but does not improve target-test execution beyond
one-query direct model prediction.

### `PUBLIC_PREDICTIVE_STATE_CANDIDATE_SUPPORTED`

Draft one new PSR Stage 1 registration with fresh histories, a new unopened
sender, a learned model-local one-shot compiler, rate-matched text summaries,
and a frozen executor. Do not launch it automatically.

## 11. Paper path after a pass

A competitive positive paper would then require a separate prospective stage:

1. freeze fresh train/validation/test histories and a new held model;
2. replace linear diagnosis with model-local typed core-test compilers;
3. preserve the exact public executor;
4. compare hidden activations against token-only, text-summary, recurrent-filter,
   and oracle-belief baselines at matched rate;
5. demonstrate held-sender establishment with no executor retraining;
6. demonstrate receiver-native use on a formal natural-language filtering or
   logical-closure domain;
7. report the source-compute/message-rate/consumer-compute frontier.

PSR0 is the smallest experiment that can justify that program without consuming
another claim-bearing role.
