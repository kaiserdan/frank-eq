# Implementation status

Snapshot: 2026-08-12

## Adopted evidence

| Stage | Status | Scientific role |
|---|---|---|
| Synthetic Stage 0 | complete | implementation evidence only |
| Real Stage-A v1/v2 | complete | exact-pipeline negatives |
| Stage Q + scale screens | complete | development negatives |
| Stage R / RC0 | complete | interactive public-basis development pass |
| Stage-A v3-2 | complete | terminal one-shot graph-compiler negative |
| PSR0 | implemented, not run | next development-only census |

The original shared/private continuous-code paths must not be resumed. RC0
showed that a calibrated, interactively queried public graph basis can be
recovered and composed reliably. Stage-A v3-2 showed that the frozen one-shot
graph compiler does not produce a renderer-invariant or activation-specific
semantic interface.

## Stage-A v3-2 interpretation

The v3 machine decision is:

```text
ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED
```

Integrity, behavioral prediction, public alignment, held-sender retention,
quantization, and oracle execution pass. Semantic Brier, unseen-renderer
transfer, activation-over-token specificity, and conjunctive composition fail.
The consumed graph registration is terminal.

The verifier refusal is numerical rather than scientific: exact-runtime
recomputation localizes 46 reducer-order differences to at most `5.55e-17` and
preserves the same decision. Future independent reducers use explicit numerical
tolerance rather than byte equality for floating metrics.

## Implemented successor: PSR0

PSR0 changes the task from visible graph reformatting to noisy sequential
filtering. It defines a four-dimensional public predictive state through
probabilities of rank-selected future tests in a controlled hidden Markov model.
An exact linear public executor maps four core-test probabilities to 18 target
tests.

Implemented surfaces:

```text
src/frank_eq/predictive_state/__init__.py
src/frank_eq/predictive_state/automaton.py
src/frank_eq/predictive_state/config.py
src/frank_eq/predictive_state/panel.py
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

## PSR0 frozen design

```text
models:                   pinned Qwen3-4B and Qwen3-8B
predictive rank:          4
core tests / targets:     4 / 18
core condition number:    1.8311192670
train histories:          256 at lengths 8 and 16
validation histories:     192 at lengths 8, 16, and 32
fit renderers:            narrative and table
unseen renderer:          symbolic
prefixes per model:       1,088
KV branches per model:    23,936
```

The primary train-only linear activation probe is compared with an
order-sensitive token-only hash probe, mean input embeddings, priors, calibrated
interactive future-test queries, direct target queries, and the oracle basis.
History is the paired independent unit.

High-dimensional ridge fitting uses the dual system when feature width exceeds
history count. The workflow loads models sequentially, forbids replay fallback,
uses exact `chat_turn` prefix continuity, and opens no held or test role.

The independent verifier regenerates automaton, basis, panels, probes,
predictions, metrics, and decision from hash-bound artifacts. Floating reducer
comparisons use `atol=rtol=1e-12`; decision and protected authorization fields
remain fail closed.

## PSR0 promotion boundary

All of the following must pass for every model:

- public core gain over prior on seen, unseen-renderer, length-transfer, and
  joint-OOD conditions;
- activation gain over token-only control on joint OOD;
- unseen-renderer transfer and a seen/unseen Brier gap at most `0.02`;
- length-32 transfer;
- compiled target-test gain over both prior and direct prediction on joint OOD,
  in aggregate and at every target horizon;
- exact rank/conditioning/executor checks.

A pass permits drafting one fresh PSR Stage 1 registration only. Execution,
held-sender onboarding, test access, receiver work, scientific claims, and paper
claims remain unauthorized.

## Next executable action

One content-addressed Olivia dry run is ready:

```bash
python olivia/cli.py submit \
  --job-name frank-eq-psr0-olivia-20260812a \
  --config configs/predictive_state/real_olivia_psr0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

No PSR0 model run has been launched from the implementation branch.
