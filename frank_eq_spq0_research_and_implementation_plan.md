# Frank-EQ SPQ0: Shared Predictive Quotient Census

## Status

Prospective development-only successor to the consumed graph campaigns. This plan does not authorize reuse, retuning, or reinterpretation of Stage-A v1/v2/v3, RC0, or Stage M0. It authorizes only implementation review and, after a green immutable dry run, one development-only Olivia audit.

## Scientific decision

Stop the explicit deterministic-graph line. The next object is the **minimal shared predictive quotient** of a genuinely partially observed stochastic process.

The headline hypothesis is:

> The part of an LLM latent state that transfers across architectures is not a shared hidden coordinate system and not an exhaustive semantic event vocabulary. It is a low-dimensional, future-predictive quotient whose dimension is controlled by the predictive rank of the underlying process. The remaining state is model-private execution and calibration state.

## Why the graph line stops

1. The graph is fully stated in the prompt. Its semantic state is a deterministic point, so the directed edge indicators already form a sufficient basis.
2. Stage M0 added 318 joint events to a deterministic semantic state. Those events were algebraically redundant but separately noisy. The moment executor therefore had Brier 0.05310, versus 0.02744 for the marginal executor.
3. Stage-A v3-2 learned useful seen-renderer and behavioral structure, but failed every unseen-renderer semantic gain and did not beat the parameter-matched token-ID compiler.
4. This is not evidence that high-order uncertainty is never needed. It is evidence that model confidence scores from separately queried deterministic propositions are not a coherent posterior over worlds.
5. The correct next task must make uncertainty part of the external state itself and require sequential filtering rather than text parsing.

## Mathematical object

Let a controlled stochastic process have history

\[
h_t=(a_1,o_1,\ldots,a_t,o_t),
\]

and let \(\tau\) be a future action-observation test. Define

\[
p_\tau(h_t)=\Pr(\tau\mid h_t).
\]

Two histories are operationally equivalent for test bank \(\mathcal T\) when

\[
h\sim_{\mathcal T}h'\quad\Longleftrightarrow\quad
p_\tau(h)=p_\tau(h')\quad\forall \tau\in\mathcal T.
\]

For a finite linear predictive-state system, choose core tests \(B\) whose state-conditional test matrix \(Q_B\) has full rank \(r\). The public state is

\[
s_B(h)=b(h)^\top Q_B,
\]

where \(b(h)\) is the exact Bayesian belief. Every target test factors through the same public state:

\[
p_\tau(h)=s_B(h)W_\tau.
\]

The rank \(r\), equivalently the appropriate system-dynamics/Hankel rank, is the minimal exact predictive dimension. This gives a testable prediction:

- dimensions below \(r\) are undercomplete;
- dimension near \(r\) should maximize cross-model transfer;
- extra dimensions can carry model identity and finite-sample nuisance without improving formal future sufficiency.

## Transfer decomposition

For model \(m\), let \(H_m(h)\) be the query-blind captured state, \(S(h)\) the exact semantic predictive core, and \(Y_m(h)\) the model's own future-response signature under a frozen probability protocol.

Use the decomposition

\[
Y_m(h)=R_m^{\mathrm{sem}}S(h)+A_m Z_{\mathrm{shared}}(h)+U_m(h)+\epsilon_m.
\]

- \(S\): externally grounded semantic predictive state;
- \(Z_{\mathrm{shared}}\): optional low-rank behavioral residual common across models after conditioning on \(S\);
- \(U_m\): model-private execution/calibration residual.

The primary transfer path is

\[
H_m\xrightarrow{E_m}\widehat S_m
\xrightarrow{R_n^{\mathrm{sem}}}\widehat Y_{m\to n},
\]

where every source encoder and target reader is model-local, the public core-test meanings are frozen, and no pair-specific mapper is trained.

## Stage SPQ0 question

Can independently trained source-local readouts recover the same rank-matched public predictive state from cross-family LLM activations, and can that state drive a frozen target-local future reader better than prior and token-only controls under renderer and history-length shift?

SPQ0 is a development census, not a claim-bearing confirmation.

## Models

Development founders:

- Qwen/Qwen3-4B, revision `1cfa9a7208912126459214e8b04321603b3df60c`;
- mistralai/Mistral-7B-Instruct-v0.3, revision `c170c708c41dac9275d15a8fff4eca08d52bab71`.

Same-family diagnostic, optional only if compute permits:

- Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`.

Reserved for a later unopened establishment stage:

- allenai/OLMo-2-1124-7B-Instruct, revision `470b1fba1ae01581f270116362ee4aa1b97f4c84`;
- ibm-granite/granite-3.1-8b-instruct, revision `4009206d5fc95d2e65a7b7633e159d6e97e25d35`.

Do not load the reserved checkpoints in SPQ0.

## Controlled environments

Use a family of four-state controlled HMMs, not one fixed automaton.

- 3 actions;
- 3 observations;
- full-support transitions and emissions;
- exact Bayesian filtering;
- candidate test horizons 1--4;
- rank-selected core bank with rank 4;
- condition number at most 5;
- target-executor coefficient L1 norm at most 4.

Roles:

- calibration systems/histories;
- architecture-selection systems/histories;
- frozen validation systems/histories.

At least one transition/emission system is validation-only. All renderings and all model rows for one underlying system/history share the same role.

History lengths:

- calibration/selection: 8 and 16;
- validation: 8, 16, and 32;
- length 32 is never used for fitting or selection.

Renderers:

- fit: narrative and table;
- validation-only: symbolic compact grammar.

The prefix explicitly instructs the model to maintain the current predictive state for an unknown future test, but reveals no specific future test or answer candidate before capture.

## Probability protocol correction

Do not ask whether an unsampled stochastic future event is simply `true` or `false`. That prompt has no deterministic semantic target.

Use a frozen categorical probability forecast:

- candidate bins: 0.05, 0.15, ..., 0.95;
- score all candidate strings from one post-reveal query;
- normalize candidate likelihoods into a categorical distribution;
- predicted probability is the candidate expectation;
- retain the full categorical distribution as the model-behavior signature;
- calibrate only on calibration histories;
- select no candidate vocabulary or temperature on validation.

A threshold-query audit may be included as a secondary control, but not as the primary teacher.

## Capture surfaces

Predeclare a small state census rather than one final-token probe or indiscriminate full-state transport:

1. final-token residual at normalized depths 0.25/0.50/0.75/1.00;
2. event-boundary residuals at the final token of each action-observation step;
3. all-token mean/max summaries at the same depths;
4. selected K/V summaries at event boundaries, if the frozen runtime exposes a model-independent tensor contract;
5. mean input embedding;
6. parameter-matched token/event sequence encoder with no source activations.

No capture surface is selected on validation outcomes. Use the architecture-selection role only.

## Readouts

### Semantic core encoder

For each model and capture surface, fit a low-flexibility reduced-rank/ridge map

\[
E_m:H_m\rightarrow S.
\]

Sweep public bottleneck rank \(d\in\{1,2,3,4,6,8\}\), selected on the architecture-selection role. Rank 4 is the exact-system prediction.

### Target-local reader

For each model, fit

\[
R_m:S\rightarrow Y_m
\]

using exact oracle core states and that model's probability-bin future signature on calibration histories. Freeze before source encoders are evaluated.

Evaluate every ordered source-target pair:

\[
R_n(E_m(H_m)).
\]

There is no pair-specific alignment, target hidden state, target gradient, or joint sender-receiver training.

### Behavioral residual census

After fitting \(R_m\), form residuals

\[
\Delta_m=Y_m-R_m(S).
\]

On calibration/selection data only, estimate a pooled residual PCA or reduced-rank shared residual with rank \(k\in\{0,1,2,4\}\). Report whether a source-local residual encoder improves a different model's frozen reader. This is diagnostic in SPQ0; semantic transfer may pass even when the residual is private.

## Baselines

Mandatory:

1. per-test training prior;
2. last-observation and empirical-observation filters;
3. deterministic token hash plus ridge;
4. parameter-matched recurrent/Transformer token-sequence encoder;
5. mean input-embedding encoder;
6. final-token residual ridge;
7. historical all-token compiler, parameter matched where possible;
8. direct probability forecast after future-test reveal;
9. exact Bayesian belief;
10. exact public core;
11. overcomplete future-test bank;
12. shuffled-history, wrong-history, renderer-shuffled, and zero packets.

The exact Bayes filter is a compute ceiling, not a latent comparator. The hidden-state claim requires beating the parameter-matched token-sequence encoder under matched data and output rate.

## Primary metrics

World/history-grouped intervals with at least 2,000 bootstrap replicates:

- semantic core Brier and R2;
- target-test Brier after exact public execution;
- activation minus token-sequence Brier gain;
- wrong-history specificity;
- renderer and length transfer;
- source-to-target future-signature gain over target prior;
- source-to-target retention relative to oracle-core target reader;
- rank-performance curve;
- model-identity leakage from the public packet;
- shared-residual incremental gain;
- 2/4/8-bit quantization retention;
- rate and producer/consumer compute;
- amortized comparison against direct post-reveal queries at 1, 4, 16, and 32 future tests.

## SPQ0 machine gate

A candidate is supported only if all hold:

1. exact core rank/executor checks pass;
2. source native probability forecasts beat prior on calibration-independent validation for both founder families;
3. semantic core readout beats prior in every model × renderer-role × length-role group;
4. the selected activation surface beats the parameter-matched token-sequence control on joint OOD with lower-95 gain > 0 for both founders;
5. wrong-history specificity lower95 > 0;
6. every ordered cross-family source-target composition beats the target prior with lower95 > 0;
7. every cross-family direction retains at least 0.70 of the oracle-core reader gain;
8. rank-4 performance is non-inferior to every higher rank, and ranks below 4 show the predicted undercomplete loss in aggregate;
9. four-bit quantization retains at least 0.95 of float cross-family gain;
10. sender-ID leakage from the semantic packet is no more than 0.15 above chance;
11. all authorizations remain false except permission to draft SPQ1.

Do not require the reusable packet to beat one direct query. Require a positive amortized utility margin at a frozen multi-query count, preferably 16, and report the complete frontier.

## Machine diagnoses

- `PREDICTIVE_SYSTEM_OR_EXECUTOR_INVALID`
- `SOURCE_PROBABILITY_PROTOCOL_NOT_QUALIFIED`
- `SEMANTIC_PREDICTIVE_QUOTIENT_NOT_READABLE`
- `NO_ACTIVATION_SPECIFIC_PREDICTIVE_ADVANTAGE`
- `PREDICTIVE_QUOTIENT_NOT_RENDERER_OR_LENGTH_STABLE`
- `NO_CROSS_FAMILY_PUBLIC_STATE_TRANSFER`
- `TRANSFER_RANK_NOT_IDENTIFIED`
- `PUBLIC_PREDICTIVE_QUOTIENT_CANDIDATE_SUPPORTED`

Only the last diagnosis permits drafting SPQ1. It does not authorize running it.

## SPQ1 after a pass

SPQ1 is the claim-bearing confirmation:

- fresh stochastic systems and histories;
- one unseen model family reserved before implementation;
- nonlinear model-local compilers selected only from SPQ0 development evidence;
- public rank-4 packet and frozen exact executor;
- target-local readers frozen before held-sender onboarding;
- one-way held-sender establishment;
- no receiver retraining;
- state swaps and matched wrong-history controls;
- source-free target execution;
- formal natural-language sequential-evidence domain as a second task;
- complete rate--compute frontier.

## Paper thesis

Working title:

**Shared Futures, Private States: The Transferable Predictive Quotient of Language Models**

Primary claim, if prospectively confirmed:

> Across heterogeneous language models, transferable latent information concentrates near the predictive rank of the task. Model-local compilers can export this compact future-sufficient quotient through fixed public tests, while extra latent dimensions primarily encode model-private execution state and reduce held-sender transfer.

Figures:

1. transfer quality versus packet dimension, with the system predictive rank marked;
2. semantic quotient, shared behavioral residual, and private residual variance;
3. cross-family source-target transfer matrix plus held sender;
4. renderer/length and wrong-history controls;
5. payload-rate versus amortized future-query utility;
6. controlled stochastic task and formal natural-language confirmation.

## Repository implementation sequence

1. Rebase/supersede `agent/predictive-state-psr0` onto current `main` after Stage M0.
2. Preserve all existing Stage M evidence unchanged.
3. Rename the protocol to `spq0` and mark the old PSR0 branch historical/unexecuted.
4. Replace the ill-defined stochastic true/false teacher with probability-bin scoring.
5. Add cross-family Mistral founder and reserve OLMo/Granite checkpoints.
6. Add calibration/selection/validation roles and held transition/emission systems.
7. Add capture-surface census and parameter-matched token-sequence control.
8. Add target-local readers and ordered source-target composition.
9. Add rank sweep and behavioral-residual census.
10. Replace direct-query conjunctive gate with the rate-aware amortized gate.
11. Add exact independent recomputation, source hashes, and Olivia dispatch.
12. Run compile, Ruff, complete pytest, historical evidence validation, SPQ static validation, shell syntax checks, and a content-addressed dry run.
13. Launch no model inference until the dry-run source/config/image hashes and reserved-model non-access contract are inspected.

## Independent review corrections before launch

The implementation review made four prospective corrections before any model
execution. First, the transcript comparison is named precisely: it is a fixed
causal token sketch with an independently selected, parameter-matched linear
readout, not a learned sequence model. Second, every cross-family reader now
evaluates both the activation-derived packet and the source-token-derived packet;
a positive activation-over-token lower bound is conjunctive. Third, predictive
rank is identified at the cross-family target-reader endpoint with paired
rank-four comparisons, rather than from source-local semantic point estimates.
Fourth, the Brier-equivalent rate scalarization is diagnostic only because its
exchange rates are conventional rather than empirically identified. The
non-promotional behavioral remainder is pooled residual PCA, not MAXVAR-GCCA.

These corrections do not alter the systems, histories, model roster, public test
registry, access boundary, or protected authorizations. They prevent a favorable
SPQ0 outcome from being interpreted more strongly than the executed controls
support.
