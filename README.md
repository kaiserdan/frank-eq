# Frank-EQ

**Future-defined operational equivalence quotients for cross-model LLM state interfaces.**

Frank-EQ tests a different object from conventional hidden-state alignment. Instead of asking whether two model states occupy similar coordinates, it asks whether they preserve the same family of future computations.

For a cached state `h` and a frozen operation family `K`, the operational signature is

```text
Sigma_K(h) = { p(y | h, k) : k in K }.
```

States are equivalent when their future signatures agree. The intended transferable object is the quotient of hidden states by this operational equivalence, not a reconstruction of another model's native hidden vector.

## Current implementation

The repository contains a complete, executable **synthetic Stage 0**:

- model-local hidden-state charts with heterogeneous widths;
- renderer-specific nuisance and model-private variation;
- state capture before operation reveal;
- a public bank of held-out future operations;
- a gauge-fixed public code consisting of grounded fact coordinates plus an operational residual;
- a frozen, parameter-free public operation decoder;
- founder training followed by held-sender onboarding with the shared decoder frozen;
- renderer invariance, model-ID leakage, cross-model retrieval, wrong-world specificity, facts-only residual gain, quantization retention, and packet round-trip tests;
- grouped bootstrap intervals and a fail-closed machine decision;
- deterministic typed operational packets;
- backend-agnostic contracts for real LLM future-branch caches.

The full synthetic reference run passes its frozen implementation gate and writes `PROMOTE_REAL_MODEL_CANARY`. This authorizes only a real-model canary; it never authorizes a scientific claim. See `evidence/reference_stage0/` and `docs/09_IMPLEMENTATION_STATUS.md`.

## Architecture

```text
model-local hidden trajectory
        |
        v
private model-local chart
        |
        v
gauge-fixed public causal state
  grounded facts | operational residual
        |
        +---- frozen future-operation decoder
        |
        +---- query-conditioned typed packet
        |
        v
receiver-native execution (next phase)
```

The code intentionally separates:

1. a private model-local chart used only to extract state;
2. a gauge-fixed public quotient with externally defined coordinates;
3. a frozen operation decoder that can execute held-out operations;
4. a query-conditioned wire representation;
5. a future receiver-native executor, which is not yet implemented or claimed.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

frank-eq validate-config --config configs/stage0/synthetic_smoke.yaml
frank-eq run-stage0 \
  --config configs/stage0/synthetic_smoke.yaml \
  --out runs/synthetic-smoke
```

Run the stricter reference experiment:

```bash
frank-eq run-stage0 \
  --config configs/stage0/synthetic_full.yaml \
  --out runs/synthetic-stage0-v1
```

The authoritative outputs are:

```text
runs/.../train/final.pt
runs/.../train/training_summary.json
runs/.../eval/metrics.json
runs/.../eval/decision.json
runs/.../eval/artifact_manifest.json
runs/.../run_summary.json
```

Only `eval/decision.json` carries promotion semantics.

## Commands

```text
frank-eq validate-config  Validate a YAML contract.
frank-eq make-synthetic   Materialize the controlled Stage-0 bundle.
frank-eq train-stage0     Train founder charts and onboard a held sender.
frank-eq eval-stage0      Evaluate, bootstrap, and reduce the gate.
frank-eq run-stage0       Run the complete synthetic workflow.
```

## Start here for agents

Read in this order:

1. `AGENTS.md`
2. `docs/00_PROJECT_CHARTER.md`
3. `docs/01_SCIENTIFIC_HYPOTHESIS.md`
4. `docs/02_ARCHITECTURE.md`
5. `docs/03_INFORMATION_ACCESS_CONTRACT.md`
6. `docs/04_STAGE0_PROTOCOL.md`
7. `docs/05_GATES_AND_STOP_RULES.md`
8. `docs/06_REAL_MODEL_PLAN.md`
9. `HANDOFF.md`

The reusable agent skill is `.agents/skills/frank-eq-research/SKILL.md`.

## Validation

```bash
python -m compileall -q src scripts
pytest -q
python scripts/validate_repo.py
```

## Claim boundary

The repository currently establishes that the proposed contracts and learning problem are executable on a controlled benchmark. It does **not** establish:

- a universal latent space in real LLMs;
- cross-model hidden-state execution;
- autonomous receiver generation;
- held-sender establishment on real checkpoints;
- superiority to text communication;
- causal or safety claims about natural-language models.

Those claims remain locked behind the real-model protocol in `docs/06_REAL_MODEL_PLAN.md`.
