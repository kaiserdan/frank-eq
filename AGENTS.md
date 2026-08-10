# Agent operating contract

## Mission

Frank-EQ studies whether LLM hidden states admit a compact, future-defined operational equivalence quotient that can be populated by independently trained senders and eventually executed by frozen receivers.

The project is not an open-ended hidden-state architecture search. Every experiment must answer a frozen scientific question and fail closed.

## Reading order

1. `README.md`
2. `docs/00_PROJECT_CHARTER.md`
3. `docs/01_SCIENTIFIC_HYPOTHESIS.md`
4. `docs/02_ARCHITECTURE.md`
5. `docs/03_INFORMATION_ACCESS_CONTRACT.md`
6. `docs/04_STAGE0_PROTOCOL.md`
7. `docs/05_GATES_AND_STOP_RULES.md`
8. `docs/06_REAL_MODEL_PLAN.md`
9. `docs/11_REAL_STAGEA_IMPLEMENTATION.md`
10. `docs/09_IMPLEMENTATION_STATUS.md`
11. `HANDOFF.md`
12. `docs/10_DECISION_LOG.md`

For cluster work, also read `.agents/skills/olivia-cluster-runner/SKILL.md` or `.agents/skills/lumi-cluster-runner/SKILL.md`.

## Non-negotiable scientific invariants

### State formation precedes operation reveal

A claim-bearing state must be captured before the future operation, query, intervention, verification request, or target is revealed. The operation may enter only the frozen interrogator or query-conditioned selector after capture.

Real cache rows are invalid unless `FutureSignatureRecord.validate()` verifies `operation_reveal_step > capture_step` for every branch.

### Exact-prefix replay is not silently called physical cache reuse

The real backend records how many branches used cloned KV state and how many used deterministic prefix replay. Both preserve causal order, but only KV reuse is literal branching from the cached runtime state. Reports must distinguish them.

### Split by world, never by row

All renderers, model views, operations, branches, and seeds belonging to one world remain in one split. A duplicated world under another surface form is the same grouping unit.

### Local charts, public semantics

Model-local charts may be expressive. The shared quotient must use externally defined facts, residual coordinates, and operations. Do not force private chart coordinates to match and do not credit private geometry as interoperability.

### Oracle and source behavior remain separate

Formal oracle operation outcomes define the public state semantics. Source-model branch probabilities are a diagnostic. Never replace oracle labels with model agreement after observing outcomes, and never repair model branches with oracle probabilities.

### Held-sender onboarding freezes the shared executor

The held sender may train its own source-local chart using source-side labels. The public graph decoder, packet schema, operation registry, rates, gates, founder checkpoints, and test worlds remain frozen.

### No target-private rescue

The strict path may not consume target hidden states, target logits, target labels, pair IDs, or receiver gradients. Assisted upper bounds require a separate namespace and cannot promote the primary path.

### Facts-only is mandatory

Every operational residual claim must beat a grounded-facts-only execution under the same operation bank and rate. Remove the residual if its held-out margin is not positive.

### Machine decisions are authoritative

README prose, W&B summaries, scheduler success, and training loss do not authorize continuation. Only a complete decision artifact with required hashes and a passing parent gate can unlock the next phase.

## Prohibited shortcuts

- Selecting checkpoints, layers, operation families, renderers, rates, or thresholds after test outcomes.
- Treating hidden-state R2, CKA, cosine, or reconstruction MSE as execution evidence.
- Revealing or tokenizing the operation inside the captured prefix.
- Splitting renderer variants or branches of one world across roles.
- Jointly training a sender and receiver in the primary establishment condition.
- Replacing a failed compiled packet or hidden chart with oracle fields at evaluation time.
- Reporting synthetic or smoke runs as evidence about real LLMs.
- Reclassifying exact-prefix replay as KV reuse.
- Committing checkpoints, generated caches, private datasets, access tokens, API keys, or W&B credentials.

## Development workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m compileall -q src scripts olivia lumi
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
```

Real backend development additionally uses `pip install -e '.[real,dev]'`.

For a behavioral change:

1. add or update a focused test;
2. update the protocol before any outcome-bearing run;
3. validate synthetic backward compatibility;
4. validate real config and cache contracts without model calls where possible;
5. run cluster submission dry-run;
6. append a decision-log entry;
7. update `HANDOFF.md` and `docs/09_IMPLEMENTATION_STATUS.md`.

## File map

```text
src/frank_eq/config.py              Shared synthetic/training contract
src/frank_eq/real_config.py         Real checkpoint, panel, and capture contract
src/frank_eq/contracts.py           Causal-boundary cache records
src/frank_eq/data/synthetic.py      Synthetic Stage-0 generator
src/frank_eq/data/real_panel.py     Frozen graph worlds, renderers, operations
src/frank_eq/data/hf_backend.py     HF hidden capture and future branches
src/frank_eq/data/real.py           Real bundle builder and validator
src/frank_eq/models/encoder.py      Model-local charts and public quotient
src/frank_eq/models/layers.py       Synthetic and graph frozen interrogators
src/frank_eq/workflow.py            Auditable real Stage-A workflow
src/frank_eq/cluster.py             Content-addressed cluster operations
olivia/                             Olivia Slurm and operator surface
lumi/                               LUMI Slurm and operator surface
configs/stage0/                     Synthetic and real frozen configurations
```

## Artifact hierarchy

Use evidence in this order:

1. frozen protocol/config and source hash;
2. cache validation and access audit;
3. machine decision and reducer;
4. held-out metric artifact;
5. prediction artifact;
6. training summary/history;
7. source-model branch diagnostics;
8. W&B as secondary telemetry;
9. prose documentation.

## Current authority

Synthetic Stage 0 passes. Real Stage A is implemented and may be launched on Olivia or LUMI. No real-model cache, quotient, or receiver result has yet been adopted. The only authorized scientific action is the frozen real Stage-A cache/train/eval canary; receiver execution remains locked.
