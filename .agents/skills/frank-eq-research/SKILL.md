---
name: frank-eq-research
description: Audit and extend Frank-EQ public predictive-state experiments while preserving causal order, activation-specific controls, and immutable negative evidence.
---

# Frank-EQ research skill

## Current authority

Read `AGENTS.md`, `HANDOFF.md`,
`docs/22_PREDICTIVE_STATE_PSR0.md`, and
`docs/23_PSR0_OLIVIA_RUNBOOK.md` before acting.

RC0 is an adopted interactive public-basis pass. Stage-A v3-2 is a terminal
one-shot graph-compiler negative. The only authorized prospective execution is
PSR0 under its committed config and plan.

## Scientific object

PSR0 studies a public predictive state:

```text
p_tau(h) = b(h)^T q_tau
s_B(h) = b(h)^T Q_B
p_tau(h) = s_B(h) Q_B^{-1} q_tau
```

The sufficient state requires filtering a noisy history; it is not explicitly
printed as graph facts. Core coordinates are named future events and the target
executor is exact and public.

## Mandatory workflow

1. Validate repository, PSR0 contract, shell, and tests.
2. Confirm pinned Qwen3-4B/Qwen3-8B snapshots.
3. Confirm the committed plan SHA.
4. Dry-run the content-addressed Olivia submission.
5. Submit exactly `audit`.
6. Fetch and run both cluster and PSR0-specific verifiers.
7. Inspect every model and joint-OOD stratum.
8. Adopt only a compact hash-verified evidence package.
9. Follow the machine diagnosis; do not adapt and rerun PSR0.

## Invariants

- State formation precedes future-test reveal.
- Use corrected `chat_turn`, exclusive cloned KV branching, and no replay.
- Fit/select probes on training histories only.
- Group all views by history.
- Keep semantic and behavioral predictive states separate.
- Compare activations with full transcript token-only and embedding controls.
- Require activation-specific advantage on unseen grammar and unseen length.
- Treat interactive future-test scoring as tomography.
- Keep all held/test/receiver/claim authorizations false.
- Machine decisions and hashes outrank W&B and prose.

## Prohibited actions

- Retrying or tuning consumed Stage-A v3-2.
- Another visible-graph compiler as the primary continuation.
- Shared private latent alignment or direct target-hidden reconstruction.
- Opening a held sender or claim-bearing split in PSR0.
- Changing the task, renderers, models, probes, layers, or gates after reading
  PSR0 validation outcomes and rerunning the same identity.
- Promoting an aggregate while a model or joint-OOD stratum fails.
- Starting receiver execution.
- Committing generated runs, checkpoints, `.agents/state/`, or credentials.

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
