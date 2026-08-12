# Frank-EQ

**Public operational interfaces for future-defined LLM state.**

Frank-EQ asks whether a state formed before a future operation is revealed can
be compiled into an externally identifiable description of the computations it
supports. Hidden-coordinate similarity is not treated as interoperability.

For frozen model `M`, query-blind state `h`, future operation `k`, and declared
post-reveal compute contract `c`:

```text
Sigma_M(h; k, c) = p_M(y | h, k, c).
```

## Evidence so far

### Historical shared-code architectures: exact-pipeline negatives

Real Stage-A v1/v2, Stage Q, and the scale screens rule out the tested
final-token/private-chart/shared-head constructions. The v2 public code had weak
world specificity and held-sender establishment while retaining strong model
identity. The corrected Stage-Q comparison also showed that proper user-turn
placement did not repair immediate source competence.

These are architecture-specific negatives. They do not establish that the full
runtime state lacks a reusable operational quotient.

### RC0: positive interactive public-basis result

The adopted Olivia RC0 capture and artifact-only recovery returned:

```text
PUBLIC_BASIS_COMPOSITION_SUPPORTED
```

For hard graph operations:

```text
compiled public-basis Brier:       0.0408
direct frozen-model Brier:         0.2035
train-world prior Brier:           0.2181
lower95 gain over direct/prior:     0.1542 / 0.1661
weakest basis balanced accuracy:   0.9246
oracle executor mismatches:        0
```

Semantic sequence scoring improved over the historical answer-token channel.
Generated reasoning did not beat an equal-token pause control. RC0 therefore
supports a typed public basis plus deterministic composition, not a chain-of-
thought claim.

RC0 is interactive tomography: the source is queried separately for every basis
coordinate after capture. It is an upper bound, not a one-shot message.

### Stage-A v3-2: one-shot graph compiler negative

The sole registered one-shot compiler ran on Olivia job `1899057`, passed every
integrity check, and returned:

```text
ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED
```

The exact result is more informative than the aggregate failure:

- behavioral self-future prediction passed;
- public alignment, held-sender retention, quantization, and exact execution
  passed;
- seen-renderer semantic decoding was strong;
- every unseen canonical-renderer semantic gain was negative;
- activations did not beat the matched token-ID compiler;
- aggregate composition improved, but held-model and operation-family strata
  failed.

The independent verifier refusal was caused only by reduction-order differences
at most `5.55e-17`; exact-runtime recomputation preserved the same machine
scientific decision. The v3-2 registration is consumed and terminal.

## Scientific diagnosis

The data now reject the original primary framing:

```text
map one model's hidden vector into another model's hidden vector
```

Three recurrent separations are more defensible:

```text
predictable geometry       != receiver-native utility
interactive basis access   != one-shot sender compilation
visible semantic facts     != latent predictive state
```

The graph task has become the wrong final application. Its semantic basis is
printed in the prefix, so a parser or token-only model has an intrinsically
complete solution. The v3 unseen-renderer reversal further shows that the
learned compiler bound itself to grammar rather than recovering a stable public
state.

The strongest positive clue is the behavioral channel: frozen-model future
responses remain predictable even when externally grounded graph semantics do
not transfer. The next question should therefore use a state that must be
computed by filtering or closure, not copied from visible text.

## Current next experiment: PSR0

PSR0 is a fresh development-only predictive-state census. It asks whether frozen
LLM activations expose an amortized belief state for a noisy controlled automaton
and whether that state transfers across grammar and history length.

For history `h` and future test `tau`:

```text
p_tau(h) = P(future event tau | h).
```

A rank-selected public core-test bank `B` has matrix `Q_B`. When `Q_B` has full
predictive rank:

```text
s_B(h) = b(h)^T Q_B
p_tau(h) = s_B(h) Q_B^{-1} q_tau.
```

Thus the core-test probabilities are an externally identifiable separating
basis and every registered target test has an exact public executor.

Frozen PSR0 facts:

```text
models:                   Qwen3-4B and Qwen3-8B, exact revisions
latent process:           four-state controlled HMM
public predictive rank:   4
core tests / targets:     4 / 18
core condition number:    1.8311
train lengths:            8 and 16
validation lengths:       8, 16, and unseen length 32
fit grammars:             narrative and table
unseen grammar:           symbolic
held sender:              none
claim-bearing test role:  none
```

The primary probe is linear and selected on training histories only. It is
compared against:

- an order-sensitive token-only hash probe at matched width;
- a mean input-embedding probe;
- the training-history prior;
- calibrated interactive future-test queries;
- direct target-test prediction;
- the exact oracle public basis.

The decisive condition is joint OOD: unseen symbolic grammar at history length
32. A pass requires every model to show positive core readability, positive
activation-specific advantage over tokens, positive wrong-history specificity, renderer and length transfer, and
compiled target-test advantage over both prior and direct prediction.

Read:

```text
docs/22_PREDICTIVE_STATE_PSR0.md
docs/23_PSR0_OLIVIA_RUNBOOK.md
HANDOFF.md
AGENTS.md
```

Frozen artifacts:

```text
configs/predictive_state/real_olivia_psr0.yaml
configs/predictive_state/inspected_plan.json
```

## Olivia dry run

```bash
python olivia/cli.py submit \
  --job-name frank-eq-psr0-olivia-20260812a \
  --config configs/predictive_state/real_olivia_psr0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

No PSR0 model run has been launched from this branch.

## Decision after PSR0

A negative diagnosis stops or localizes the corresponding branch:

```text
PREDICTIVE_BASIS_OR_EXECUTOR_INVALID
ACTIVATION_PREDICTIVE_STATE_NOT_READABLE
NO_ACTIVATION_SPECIFIC_PREDICTIVE_STATE_ADVANTAGE
PREDICTIVE_STATE_NOT_RENDERER_INVARIANT
PREDICTIVE_STATE_NOT_LENGTH_TRANSFERABLE
PUBLIC_PREDICTIVE_STATE_NOT_COMPOSITIONALLY_USEFUL
```

Only

```text
PUBLIC_PREDICTIVE_STATE_CANDIDATE_SUPPORTED
```

permits drafting one fresh PSR Stage 1 protocol. It does not authorize executing
that protocol, onboarding a held sender, opening a claim-bearing test role,
running a receiver, or making a claim.

## Commands

Historical workflows remain available through `frank-eq`. PSR0 uses an isolated
operator CLI:

```text
python scripts/predictive_state_cli.py validate
python scripts/predictive_state_cli.py plan
python scripts/predictive_state_cli.py run
python scripts/predictive_state_cli.py verify
```

## Local validation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[real,dev]'

python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
python scripts/validate_predictive_state.py
```

## Claim boundary

Frank-EQ currently establishes:

- reproducible causal-order and cluster infrastructure;
- exact negatives for several shared/private continuous-code pipelines;
- a positive interactive typed-basis composition result;
- a terminal one-shot graph-compiler negative with strong failure localization;
- a prospective, theorem-backed predictive-state experiment.

It does not yet establish an activation-specific public interface, a qualified
held sender, receiver-native execution, a rate advantage over text/tokens, or a
positive ICLR claim.
