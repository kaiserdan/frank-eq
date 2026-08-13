# Shared Predictive Quotient census: SPQ0

Status: frozen prospective development protocol; no model run has occurred.

## Question and boundary

SPQ0 asks whether two independently pretrained model families expose a common,
low-rank predictive state for a fresh controlled stochastic process. It is a
new task-level question after the completed graph experiments. It does not
reuse or retune any RC0, Stage-A, Stage-Q, or Stage-M world, renderer,
operation, threshold, response, model role, or evidence artifact.

For controlled system `u`, history `h`, and public future test `tau`, define

```text
q_u(h, tau) = P_u(the terminal observation named by tau
                  | h, interventions named by tau).
```

The exact public core is a typed list of future tests. A coordinate always has
the same action-sequence and observation-event meaning. It is not a shared
private coordinate system and no pair-specific source-to-target mapper is
allowed.

SPQ0 is development-only. It has calibration, architecture-selection, and
validation roles, and no test role. A pass may authorize drafting one fresh
SPQ1 protocol. It never authorizes an SPQ1 run, reserved-checkpoint access, a
held sender, receiver execution, a scientific claim, or a paper claim.

## Registered sources and reserved checkpoints

The two active founders are cross-family and revision pinned:

```text
qwen3-4b
  Qwen/Qwen3-4B
  1cfa9a7208912126459214e8b04321603b3df60c

mistral-7b-v03
  mistralai/Mistral-7B-Instruct-v0.3
  c170c708c41dac9275d15a8fff4eca08d52bab71
```

The following checkpoints are reserved for a possible future protocol and are
not inputs to SPQ0:

```text
olmo2-7b-held
  allenai/OLMo-2-1124-7B-Instruct
  470b1fba1ae01581f270116362ee4aa1b97f4c84

granite31-8b-held
  ibm-granite/granite-3.1-8b-instruct
  4009206d5fc95d2e65a7b7633e159d6e97e25d35
```

SPQ0 may not resolve their snapshots, list or open their files, instantiate a
model adapter, load weights, tokenize a task prefix, or execute inference. The
checkpoint preflight records zero for each operation and the verifier checks
that neither reserved model appears in a capture. Merely finding either model
in a shared cache is not access, but SPQ0 code must not inspect that cache
entry.

Before active model construction, local-only snapshot resolution must identify
the exact requested commit for each founder. Every active snapshot file is
hashed and its byte count recorded. The aggregate file registry is hashed, and
independent verification reads and rehashes every recorded file. Network
fallback and an unpinned branch or tag are forbidden.

## Controlled stochastic systems

The family consists of three full-support controlled hidden-state systems:

```text
latent states:             4
actions:                   3
observations:              3
fit systems:               2
validation-only systems:  1
minimum row probability:  0.04
system seed:               2026084101
```

The two fit systems are generated independently. The validation-only system is
a prospectively frozen 10% independent full-support perturbation of one fit
law. Its transition and emission matrices are distinct. No model response is
used in system construction. Calibration and selection contain only fit-system
histories; the held transition/emission law appears only in validation.

The public future-test candidate registry contains horizons 1 through 4. Core
and target tests are selected using fit systems only. Replacing the
validation-only matrices cannot change either registry, and a focused test
enforces this fact. The held system is used only to verify that the already
selected registry and executor are valid and to score frozen transfer.

## Exact predictive core and rank census

Each system has exact linear predictive rank four. The implementation selects
one common typed core of four future tests, then four additional typed tests for
an overcomplete public bank. For every system, the four core test vectors must
have full rank, condition number at most 5, and an exact target executor with
maximum absolute error at most `1e-10`.

Twenty-four target future tests are selected prospectively from fit systems,
balanced over horizon and observation strata. Their rank-four executor must
have maximum column L1 norm at most 4 on every system. Ranks are evaluated in
the frozen order:

```text
1, 2, 3, 4, 6, 8
```

For rank `r`, the source-local semantic encoder predicts the first `r` typed
public tests and the system-local, parameter-free executor maps that packet to
the target-test probabilities. Ranks below four are intentionally
undercomplete. Ranks four, six, and eight must reproduce target probabilities
exactly when given oracle public coordinates. The primary packet is rank four.

Candidate condition scores are rounded to 14 decimal places before comparison,
with candidate-registry order as the declared tie break. This prevents
Accelerate/OpenBLAS/MKL last-bit differences from selecting different members
of a mathematically tied symmetry class. Runtime arrays retain float64
precision. Only the plan's basis-registry digest and reported condition/L1
summaries use a 10-decimal canonical representation; the registered executor
error bound remains `1e-10`. A validator must reproduce the same typed
registries and canonical digest on every supported platform.

## Role and rendering freeze

The independent unit is one `(system, history)` group. Every renderer, model,
future-test response, surface, and control for that group remains in the same
role.

```text
role          systems           lengths       histories/system/length
calibration   fit only          8, 16         96
selection     fit only          8, 16         48
validation    fit + held law    8, 16, 32     64
test          none              none          0
```

This produces 384 calibration histories, 192 selection histories, and 576
validation histories. Narrative and table renderers occur in fit roles. The
symbolic grammar and length 32 occur only in validation. Validation conditions
are reported separately as seen, unseen renderer, unseen system, length
transfer, and joint OOD.

All role seeds are disjoint. Calibration fits local probability temperatures
and training maps. Selection chooses surfaces, depths, encoder methods,
regularization, target-reader regularization, and residual rank. Validation is
read once after those choices are frozen. There is no cross-validation that
returns validation outcomes to selection.

## Causal capture and categorical forecasting

The full history and system description form the first user turn. The captured
prefix ends exactly at that turn, before an assistant generation header. Every
post-capture branch appends the same fixed assistant acknowledgement and then a
new user turn containing the future test. This registered
`user_prefix_fixed_assistant_ack_user_query` shape satisfies Mistral's strict
role alternation and Qwen's context-dependent thinking template. The future
test, target observation, probability bins, candidate labels, and answers do
not occur in the captured prefix. The model is run once on that prefix. Every
future-test query uses an exclusive clone of that literal KV cache. Replay
fallback, mixed caches, and a second prefix forward are forbidden.

For every query branch the workflow verifies exact token-prefix continuity:
the formatted prefix tokens must be the leading tokens of the formatted
prefix-plus-new-user-query conversation. Fast-tokenizer offsets locate every
history event boundary in the formatted prefix, and the capture records those
checks. Failure is terminal.

Before either active model is constructed, a tokenizer-only preflight repeats
the turn-continuity and event-offset checks across every registered
role/system/length/renderer stratum and all 32 future tests. It records exact
candidate token IDs and the maximum prefix-plus-query length. This may inspect
only the already hashed active snapshots; it performs no model load or
inference and never resolves a reserved checkpoint.

The historical stochastic true/false protocol is rejected. SPQ0 forecasts ten
categorical probability bins:

```text
A=.05 B=.15 C=.25 D=.35 E=.45
F=.55 G=.65 H=.75 I=.85 J=.95
```

It scores the complete candidate-label token sequences by conditional log
likelihood from the same cloned query cache, length-normalizes those scores,
applies a model-local temperature selected only on calibration histories, and
takes the categorical expectation. It does not sample a true/false answer and
does not interpret one A/B next-token logit as a probability.

The development tomography asks all 8 public and 24 target tests after each
captured prefix. Those 32 interactive queries diagnose source behavior and
train local readers. They are not the primary interface and cannot support a
communication claim.

## Query-blind surfaces and controls

At normalized depths `.25`, `.50`, `.75`, and `1.00`, SPQ0 captures:

- the final-prefix-token residual;
- mean and maximum residual summaries at exact event boundaries;
- mean and maximum summaries over every prefix token;
- the mean input embedding;
- token IDs, attention mask, and exact event indices for a deterministic
  parameter-matched token-sequence surface.

No selected-KV cross-architecture summary is registered. KV is used for causal
branching, not treated as a comparable vector across model families.

Complete model-local semantic encoders map each candidate activation surface
to every requested public coordinate. Ridge and reduced-rank regression are
evaluated over the frozen rank and regularization grids. Surface, depth,
method, and regularization are selected on the selection role, then refit on
calibration plus selection. No public coordinate has an unfitted or oracle
substitution in the learned packet.

The deterministic token-sequence encoder uses token identities, positions,
and event boundaries. Its feature width is chosen so its learned linear readout
has exactly the same number of trainable coefficients as the selected
activation encoder. It is a primary activation-specificity control, not a
weaker token-hash baseline.

Additional frozen controls are history prior, last-observation and empirical
filters, deterministic token-hash ridge, mean embedding, final token, direct
categorical forecast, exact Bayes filter, exact public core, overcomplete test
bank, shuffled renderer, shuffled/wrong history, and zero packet.

## Frozen target-local future readers

For each target model, one target-local future reader maps:

```text
(oracle exact public core,
 frozen public-executor output,
 typed future-test descriptor)
    -> target categorical probability-bin distribution.
```

It is fit on calibration histories, regularization-selected on selection, and
refit on calibration plus selection. It is frozen before any source packet is
evaluated. The executor output is a deterministic feature of the public core:
it is exact while fitting from oracle cores and is recomputed from the learned
source packet during transfer. No oracle target value is supplied at source
evaluation. The reader contains no source identity and no source-target pair
parameter. The oracle-core reader is the ceiling against which source transfer
retention is measured.

Every ordered cross-family composition is mandatory:

```text
qwen3-4b       -> mistral-7b-v03
mistral-7b-v03 -> qwen3-4b
```

The source model's complete local semantic encoder produces the typed packet;
the target model's already frozen local reader consumes it. Pair-specific
alignment, translation, calibration, or mapper parameters are zero by
construction.

## Behavioral residual census

After conditioning target behavior on the semantic public core, SPQ0 may fit a
shared low-rank behavioral residual with ranks `0, 1, 2, 4` using the registered
MAXVAR-GCCA procedure. Local residual encoders are fit on calibration data,
rank is chosen on selection, and cross-family gains are reported on validation.

This census is non-promotional. Its outcome cannot turn a failed semantic core,
activation-specificity, rank, rate, or transfer gate into a pass. It cannot
authorize a hidden shared gauge, pair-specific mapper, receiver execution, or
claim. It is preserved only as evidence about model-family behavioral
remainder after semantic conditioning.

## Rate and compute comparison

The primary learned message is the four-coordinate core quantized to four bits
per coordinate: 16 payload bits, with framing explicitly uncounted. It requires
zero post-capture source queries. Its consumer is the frozen target-local
future reader.

The direct baseline asks the source separately for each target future. The
comparison reports a frontier at 1, 4, 16, and 32 future queries, charging both
source-query compute and packet bits using the frozen config exchange rates.
The conjunctive primary gate is the grouped lower-95 utility advantage at 16
future queries. A one-query direct advantage is reported but is not a
conjunctive requirement. Development tomography queries are disclosed
separately and are never counted as the primary packet.

## Gates and decision tree

All confidence intervals use 2,000 deterministic bootstrap replicates grouped
by system/history. All registered conditions and both active models must pass.
The conjunctive checks are:

1. exact predictive system and executor;
2. calibrated categorical source forecasts beat the history prior;
3. the learned semantic quotient beats the history prior;
4. activation surfaces strictly beat the parameter-matched token sequence;
5. correct history strictly beats wrong history;
6. renderer, held-system, length-32, and joint-OOD stability;
7. both ordered cross-family transfers beat the target prior and retain at
   least 70% of oracle-reader gain;
8. exact rank four is better than ranks 1--3 and noninferior to 6 and 8;
9. four-bit packets retain at least 95% of cross-family gain;
10. sender identity is at most 0.15 above chance;
11. the 16-query amortized rate utility has strictly positive lower 95% bound.

The machine diagnoses are evaluated in causal order:

```text
PREDICTIVE_SYSTEM_OR_EXECUTOR_INVALID
SOURCE_PROBABILITY_PROTOCOL_NOT_QUALIFIED
SEMANTIC_PREDICTIVE_QUOTIENT_NOT_READABLE
NO_ACTIVATION_SPECIFIC_PREDICTIVE_ADVANTAGE
PREDICTIVE_QUOTIENT_NOT_RENDERER_OR_LENGTH_STABLE
NO_CROSS_FAMILY_PUBLIC_STATE_TRANSFER
TRANSFER_RANK_NOT_IDENTIFIED
PUBLIC_PREDICTIVE_QUOTIENT_CANDIDATE_SUPPORTED
```

Any failure stops this source/task contract. A pass opens only the right to
draft one new SPQ1 protocol with fresh claim-bearing roles and separately
authorized reserved-checkpoint access. It does not open an execution.

## Immutable execution and verification

The sole permitted stage is:

```text
--profile full --stages audit
```

Before any model run, all local checks in `docs/26_SPQ0_OLIVIA_RUNBOOK.md` must
pass. The committed config, protocol, registration, and inspected plan must
match independent recomputation byte-for-byte. The source tree must be clean
and committed. The source archive, config, plan, runtime image, active revision
registry, active checkpoint file registries, and reserved non-access registry
are SHA-256 bound.

The workflow writes immutable manifests for systems, basis, panels, captures,
training, fitted checkpoints, predictions, metrics, rate/compute, decision,
and all artifacts. The independent verifier regenerates systems and panels,
rehashes active checkpoint files, validates causal capture counters, refits all
encoders/readers/residuals, recomputes all predictions and metrics, and
reproduces the machine decision. Machine artifacts outrank prose and W&B.

No Olivia submission is part of the implementation PR. Submission requires a
separate explicit operator action after review of a clean, deterministic dry
run.
