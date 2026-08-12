# Agent operating contract

## Mission

Frank-EQ studies whether a query-blind frozen-model state can be compiled into
an externally identifiable predictive or operational interface. Do not optimize
hidden-coordinate similarity as an end in itself. Every experiment must preserve
causal order, use a frozen question, and fail closed.

## Reading order

1. `README.md`
2. `HANDOFF.md`
3. `docs/22_PREDICTIVE_STATE_PSR0.md`
4. `docs/23_PSR0_OLIVIA_RUNBOOK.md`
5. `evidence/real_stagea_v3_olivia/AUDIT.md`
6. `docs/20_STAGEA_V3_PROTOCOL.md`
7. `evidence/real_stage_r_olivia_rc0/AUDIT.md`
8. `docs/18_RATE_COMPUTE_OPERATIONAL_BASIS.md`
9. `docs/09_IMPLEMENTATION_STATUS.md`
10. `docs/10_DECISION_LOG.md`

For Olivia work, also read `.agents/skills/olivia-cluster-runner/SKILL.md`.

## Current authority

RC0 is an adopted development pass for interactive typed-basis recovery and
composition. Stage-A v3-2 is an adopted terminal negative for the registered
one-shot graph compiler. Its test access is consumed and cannot be reopened.

The only authorized prospective execution is PSR0:

```text
config: configs/predictive_state/real_olivia_psr0.yaml
plan:   configs/predictive_state/inspected_plan.json
stage:  audit
role:   development-only
```

No held sender, claim-bearing test role, receiver, scientific claim, or paper
claim is authorized.

## Scientific interpretation

### Preserve the v3 negative without broadening it

V3 succeeded on seen-renderer semantic decoding and on the behavioral channel,
but failed unseen grammar, activation-over-token specificity, and conjunctive
composition. This rules out the exact graph/compiler contract. It does not prove
that LLM activations lack predictive state.

### Do not continue on visible graph facts

The graph edge basis is printed in the prefix. A parser or token-only model has a
complete semantic solution. Do not use another graph compiler as the primary
positive paper direction.

### Predictive state is the next object

For a noisy controlled process, future-test probability is

```text
p_tau(h) = b(h)^T q_tau.
```

A full-rank public core bank `B` gives

```text
s_B(h) = b(h)^T Q_B
p_tau(h) = s_B(h) Q_B^{-1} q_tau.
```

PSR0 tests whether query-blind activations expose this compact predictive state
more cleanly than matched token-only controls.

## PSR0 invariants

- Use only the committed config and inspected plan.
- Run exactly `audit`.
- Use the pinned Qwen3-4B and Qwen3-8B revisions.
- Form the prefix state before any future test or candidate answer is revealed.
- Use corrected `chat_turn`, exact prefix continuity, exclusive cloned KV reuse,
  and no replay fallback.
- Train/select probes on training histories only.
- Keep all renderer views of one history in the same role and bootstrap history
  units, not rows.
- Treat the symbolic grammar and history length 32 as validation-only transfer.
- Keep oracle-semantic and model-behavioral predictive states separate.
- Count runtime future-test queries as tomography, not packet rate.
- Require every model and joint-OOD gate; no aggregate override.
- Keep all protected authorization fields false.

## PSR0 decision tree

### `PREDICTIVE_BASIS_OR_EXECUTOR_INVALID`

Repair only engineering or mathematical defects. Do not interpret model results.

### `ACTIVATION_PREDICTIVE_STATE_NOT_READABLE`

Stop the registered model/task contract. Do not add a larger probe search.

### `NO_ACTIVATION_SPECIFIC_PREDICTIVE_STATE_ADVANTAGE`

The transcript controls explain the result. Do not claim a hidden-state
interface.

### `PREDICTIVE_STATE_NOT_RENDERER_INVARIANT`

The readout is grammar-bound. Do not tune on PSR0 validation histories.

### `PREDICTIVE_STATE_NOT_LENGTH_TRANSFERABLE`

The readout is regime-specific rather than a stable filter state.

### `PUBLIC_PREDICTIVE_STATE_NOT_COMPOSITIONALLY_USEFUL`

The core state does not outperform direct future-test prediction.

### `PUBLIC_PREDICTIVE_STATE_CANDIDATE_SUPPORTED`

Draft one fresh PSR Stage 1 registration. Do not execute it automatically.

## Prohibited shortcuts

- Retrying or tuning Stage-A v3-2.
- Using the consumed v3 test role to select a successor graph architecture.
- Resuming shared private latent alignment.
- Mapping directly into target/receiver hidden state as the primary method.
- Mixing replay and KV branches.
- Opening a held sender or claim-bearing PSR split during PSR0.
- Changing automaton matrices, histories, renderers, gates, models, layers, or
  probes after reading PSR0 validation outcomes and rerunning the same protocol.
- Starting receiver execution.
- Committing generated runs, model snapshots, `.agents/state/`, or credentials.

## Development validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
python scripts/validate_predictive_state.py
```

Machine decisions, hashes, and independent recomputation outrank prose and W&B.
