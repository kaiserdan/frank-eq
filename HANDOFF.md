# Frank-EQ handoff

Snapshot: 2026-08-12

## Current authority

The adopted evidence contains:

```text
Synthetic Stage 0          implementation evidence only
Real Stage-A v1/v2         exact-pipeline negatives
Stage Q + scale screens    development negatives
Stage R / RC0              development pass
Stage-A v3-2               terminal one-shot compiler negative
```

RC0 demonstrated that interactively recovered public edge coordinates compose
far better than direct frozen-model answers. Stage-A v3-2 then showed that the
registered one-shot graph compiler does not produce a renderer-invariant,
activation-specific semantic interface. Its test role was consumed exactly once
and the registration must not be retried or tuned.

The currently authorized prospective execution is **PSR0 only**:

```text
config: configs/predictive_state/real_olivia_psr0.yaml
plan:   configs/predictive_state/inspected_plan.json
stage:  audit
role:   development-only
```

PSR0 contains no held sender, claim-bearing test split, receiver stage, or claim
authorization.

## What the v3 result actually says

The v3 result is not a generic failure of public operational state.

Seen-renderer validation established that the frozen activations contain enough
information for low-Brier semantic edge prediction under the training grammars.
The outcome failed because:

1. every model reversed under the unseen canonical grammar;
2. activations did not beat the matched token-ID compiler;
3. the held-model and operation-family composition strata did not pass;
4. the graph basis was explicitly visible in the input, so token/text methods
   have a complete solution.

The correct conclusion is that the graph task and compiler did not isolate a
latent, grammar-invariant public state. Do not respond by widening the graph
compiler, adding target-state reconstruction, or trying another hidden gauge.

## New scientific object

PSR0 uses a controlled noisy automaton. The query-blind prefix contains the
known dynamics and an action-observation history. The required state is the
posterior predictive state obtained by sequential Bayesian filtering.

For public future test `tau`:

```text
p_tau(h) = b(h)^T q_tau.
```

A deterministic rank-selected core bank `B` has full-rank matrix `Q_B`, hence:

```text
s_B(h) = b(h)^T Q_B
p_tau(h) = s_B(h) Q_B^{-1} q_tau.
```

The four core probabilities are therefore an exact public separating basis for
the 18 registered target tests.

## Frozen PSR0 design

Models:

```text
Qwen/Qwen3-4B
1cfa9a7208912126459214e8b04321603b3df60c

Qwen/Qwen3-8B
b968826d9c46dd6066d109eabc6255188de91218
```

Data:

```text
train:       128 histories at lengths 8 and 16
validation:   64 histories at lengths 8, 16, and 32
fit grammar: narrative + table
OOD grammar: symbolic
```

Per model:

```text
query-blind prefixes:       1,088
future tests per prefix:       22
KV response branches:      23,936
```

The models are loaded sequentially. Every future test is revealed only after
prefix capture and is executed from an exclusive cloned KV branch. Replay
fallback is forbidden.

## Primary comparisons

A train-only linear activation probe predicts the four public core-test
probabilities. It is compared with:

- train-history prior;
- order-sensitive token-hash probe at matched activation width;
- mean input-embedding probe;
- calibrated interactive source queries;
- direct target-test queries;
- exact oracle predictive state.

Mandatory conditions:

```text
seen
unseen_renderer
length_transfer
joint_ood = symbolic grammar at length 32
```

The machine gate requires every model to pass semantic readability,
activation-over-token specificity, wrong-history specificity, renderer transfer, length transfer, and
compiled target-test advantage over both prior and direct query prediction,
including every target horizon.

## Implemented surfaces

```text
src/frank_eq/predictive_state/automaton.py
src/frank_eq/predictive_state/panel.py
src/frank_eq/predictive_state/config.py
src/frank_eq/predictive_state/probes.py
src/frank_eq/predictive_state/workflow.py
src/frank_eq/predictive_state/verify.py
scripts/predictive_state_cli.py
scripts/validate_predictive_state.py
configs/predictive_state/real_olivia_psr0.yaml
configs/predictive_state/inspected_plan.json
docs/22_PREDICTIVE_STATE_PSR0.md
docs/23_PSR0_OLIVIA_RUNBOOK.md
```

The ridge implementation switches to the mathematically equivalent dual solve
when hidden width exceeds the number of development histories. The independent
PSR0 verifier tolerates only `1e-12` numerical reduction differences while
recomputing panels, probes, predictions, metrics, and decisions.

## Next command

```bash
python olivia/cli.py submit \
  --job-name frank-eq-psr0-olivia-20260812a \
  --config configs/predictive_state/real_olivia_psr0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

Inspect the content-addressed source SHA, clean Git status, checkpoint cache, and
plan SHA before submitting the same package without `--dry-run`.

## Decision after PSR0

A pass authorizes drafting one fresh PSR Stage 1 registration only. That future
stage would require fresh histories, a genuinely unopened sender, model-local
one-shot core-test compilers, rate-matched text/token/recurrent-filter baselines,
and a frozen executor.

A PSR0 failure is terminal for the corresponding diagnosis. Do not use its
validation histories to adapt thresholds, automaton matrices, renderer grammar,
probe family, layer set, or model roster and rerun under the same identity.

## Prohibited actions

- Retrying or tuning Stage-A v3-2.
- Using the consumed v3 test outcomes as a selection set for another graph
  registration.
- Treating RC0 runtime tomography as one-shot communication.
- Adding another shared private latent space.
- Mapping directly into receiver hidden coordinates as the primary method.
- Opening a PSR claim-bearing split or held sender during PSR0.
- Promoting a PSR0 aggregate when any model or joint-OOD gate fails.
- Starting receiver execution.
- Committing generated captures, model checkpoints, scheduler state,
  `.agents/state/`, or credentials.

## Validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
python scripts/validate_predictive_state.py
```
