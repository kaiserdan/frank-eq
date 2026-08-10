# Frank-EQ

**Future-defined operational equivalence quotients for cross-model LLM state interfaces.**

Frank-EQ tests a different object from conventional hidden-state alignment. Instead of asking whether two model states occupy similar coordinates, it asks whether they preserve the same family of future computations.

For a cached state `h` and a frozen operation family `K`, the operational signature is

```text
Sigma_K(h) = { p(y | h, k) : k in K }.
```

States are equivalent when their future signatures agree. The intended transferable object is the quotient of hidden states by this operational equivalence, not a reconstruction of another model's native hidden vector.

## Current implementation

The repository has two complete implementation layers.

### Synthetic Stage 0

The synthetic reference includes heterogeneous model-local charts, renderer nuisance, a gauge-fixed facts-plus-residual public code, a parameter-free operation decoder, held-operation evaluation, frozen-decoder held-sender onboarding, packet quantization, grouped bootstrap intervals, and a fail-closed machine decision. The adopted reference under `evidence/reference_stage0/` returns `PROMOTE_REAL_MODEL_CANARY`; it is implementation evidence only.

### Real-checkpoint Stage A

The real canary is now cluster-runnable and includes:

- a frozen controlled relational-world panel with two renderer families;
- hidden-state capture at normalized depths before any operation is revealed;
- exact-prefix replay and physical KV-reuse branch modes, with fallback audited;
- all registered future branches stored as `FutureSignatureRecord` objects;
- source-model A/B branch probabilities recorded separately from formal oracle outcomes;
- a parameter-free differentiable graph interrogator for held-out operations;
- founder chart training and held-sender onboarding with the public executor frozen;
- renderer invariance, cross-model retrieval, wrong-world specificity, facts-only residual, quantization, model leakage, and branch-behavior diagnostics;
- content-addressed Olivia and LUMI submit/status/fetch/verify tooling;
- workflow manifests and status files that preserve a scientifically negative gate as a valid completed job.

No real-model result has been adopted yet. The implementation authorizes a cache/train/eval canary, not a scientific claim or receiver execution.

## Architecture

```text
frozen source prefix, before operation reveal
        |
        v
selected source hidden layers
        |
        v
private model-local chart
        |
        v
gauge-fixed public causal state
  directed facts | operational residual
        |
        +---- frozen graph operation interrogator
        |
        +---- query-conditioned typed packet
        |
        v
receiver-native execution (locked next phase)
```

The source model's actual post-reveal behavior is retained as a diagnostic future signature. The primary public executor uses formal operation semantics rather than a learned target hidden-state decoder.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

frank-eq validate-config --config configs/stage0/synthetic_smoke.yaml
frank-eq run-stage0 --config configs/stage0/synthetic_smoke.yaml --out runs/synthetic-smoke
```

Real extraction additionally requires:

```bash
pip install -e '.[real,dev]'
frank-eq validate-real-config --config configs/stage0/real_smoke.yaml
frank-eq run-real-stagea \
  --config configs/stage0/real_smoke.yaml \
  --out runs/real-stagea-smoke
```

A scientific gate failure from `run-real-stagea` does not make the scheduler job fail. Workflow integrity is recorded in `workflow_status.json`; scientific promotion remains exclusively in `eval/decision.json`.

## Olivia

```bash
python olivia/cli.py submit \
  --job-name frank-eq-stagea-v1 \
  --config configs/stage0/real_olivia.yaml \
  --profile full \
  --stages cache,validate,train,eval \
  --dry-run --json

python olivia/cli.py submit \
  --job-name frank-eq-stagea-v1 \
  --config configs/stage0/real_olivia.yaml \
  --profile full \
  --stages cache,validate,train,eval --json
python olivia/cli.py status --job-name frank-eq-stagea-v1 --json
python olivia/cli.py fetch --job-name frank-eq-stagea-v1 --json
python olivia/cli.py verify --job-name frank-eq-stagea-v1 --json
```

See `docs/OLIVIA.md`. The operator skill is `.agents/skills/olivia-cluster-runner/SKILL.md`.

## LUMI

```bash
python lumi/cli.py submit \
  --job-name frank-eq-stagea-lumi-v1 \
  --config configs/stage0/real_lumi.yaml \
  --profile full \
  --stages cache,validate,train,eval \
  --dry-run --json
```

See `docs/LUMI.md`. The operator skill is `.agents/skills/lumi-cluster-runner/SKILL.md`.

## Authoritative artifacts

```text
runs/run_manifest.json
runs/workflow_status.json
runs/cache/dataset.npz
runs/cache/panel.json
runs/cache/future_signature_records.jsonl
runs/cache/cache_validation.json
runs/train/final.pt
runs/train/training_summary.json
runs/eval/metrics.json
runs/eval/decision.json
runs/eval/artifact_manifest.json
runs/run_summary.json
```

Only `eval/decision.json` carries scientific promotion semantics.

## Commands

```text
frank-eq validate-config       Validate a synthetic Stage-0 config.
frank-eq make-synthetic        Materialize the controlled synthetic bundle.
frank-eq train-stage0          Train founder charts and onboard a held sender.
frank-eq eval-stage0           Evaluate and reduce synthetic Stage 0.
frank-eq run-stage0            Run the complete synthetic workflow.
frank-eq validate-real-config  Validate a real-checkpoint Stage-A contract.
frank-eq make-real-cache       Capture states and all future branches.
frank-eq validate-real-cache   Audit the causal boundary and cache hashes.
frank-eq run-real-stagea       Run selected real cache/train/eval stages.
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
9. `docs/11_REAL_STAGEA_IMPLEMENTATION.md`
10. `HANDOFF.md`
11. `docs/10_DECISION_LOG.md`

## Validation

```bash
python -m compileall -q src scripts olivia lumi
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
```

## Claim boundary

Frank-EQ currently establishes that the synthetic and real-checkpoint workflows are implemented, causally audited, and cluster-runnable. It does **not** establish:

- that a future-defined quotient exists across real LLM families;
- receiver-native execution or autonomous generation;
- superiority to text communication;
- real held-sender establishment;
- safety, causal utility, or interoperability on locked data.

Those claims remain closed until an immutable real Stage-A decision passes and a separately frozen receiver protocol is activated.
