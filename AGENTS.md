# Agent operating contract

## Mission

Frank-EQ studies whether LLM hidden states admit a compact, future-defined
operational equivalence quotient that can be populated by independently trained
senders and eventually executed by frozen receivers.

The project is not an open-ended hidden-state architecture search. Every
experiment must answer a frozen question and fail closed.

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
10. `docs/12_STAGEA_V1_AUDIT_AND_V2_PROTOCOL.md`
11. `docs/09_IMPLEMENTATION_STATUS.md`
12. `HANDOFF.md`
13. `docs/10_DECISION_LOG.md`

For cluster work, also read
`.agents/skills/olivia-cluster-runner/SKILL.md` or
`.agents/skills/lumi-cluster-runner/SKILL.md`.

## Current authority

Synthetic Stage 0 passes. Real Stage-A v1 is an adopted, engineering-valid
negative result for the exact v1 pipeline. Its broader failure localization is
unresolved.

The only authorized next scientific action is the non-promotional
train/validation-only cache diagnostic:

```bash
frank-eq diagnose-real-cache \
  --cache <v1-run>/runs/cache \
  --out <v1-run>/runs/diagnostics
```

No new outcome-bearing Stage-A run and no receiver experiment is authorized.

## Non-negotiable scientific invariants

### State formation precedes operation reveal

A claim-bearing state must be captured before the future operation, query,
intervention, verification request, or target is revealed.

### Self-future state and oracle semantics are distinct

`model_signatures` are the frozen source model's own future branch
distributions. `signatures` are external oracle outcomes. They answer different
questions and must remain in separate metric namespaces.

Do not call an oracle-supervised world-state decoder a pure operational
equivalence quotient.

### The complete causal state is not the final-token residual

Stage-A v1 captured only final-token residual vectors at three depths while
literal branching used the complete KV cache. A negative v1 result cannot be
generalized to the full causal runtime state.

Any v2 capture expansion must be frozen before fresh test access and must
distinguish residual, token-sequence, and selected-KV streams.

### Exact-prefix replay is not physical cache reuse

Both preserve operation ordering, but only cloned KV reuse is literal branching
from the cached runtime state. Future cache claims require a registered
KV-versus-replay numerical parity audit.

### Split by world, never by row

All renderers, model views, operations, branches, and seeds belonging to one
world remain in one split.

### Complete local compilers, public semantics

Only public coordinates and their operation semantics need to be shared.
Model-local charts and public-coordinate heads may be fully local. A held sender
must not be forced into a founder-private bottleneck gauge merely because a
head was shared for convenience.

The historical v1 architecture uses shared public heads and remains available
for checkpoint compatibility. `model.public_head_scope: local` is a dormant v2
implementation option, not an authorized experiment.

### Held-sender onboarding freezes public execution

The held sender may train a complete source-local compiler using source-side
labels. The public operation registry, public interrogator, packet schema,
rates, gates, founder checkpoints, and test worlds remain frozen.

### No target-private rescue

The strict path may not consume target hidden states, target logits, target
labels, pair IDs, or receiver gradients.

### Facts-only comparisons must be nontrivial

A residual comparison is invalid as evidence for hidden operational information
when the residual target is explicitly printed in the prefix. Such coordinates
must be called declared-global controls or removed from the scientific gate.

### Invariance requires non-collapse

Renderer cosine alone cannot pass an invariance claim. It must be accompanied
by positive world specificity/retrieval and low model identity leakage.

### Machine decisions are authoritative

README prose, W&B summaries, scheduler success, and training loss do not
authorize continuation. Only a complete decision artifact with required hashes
and a passing parent gate can unlock the next phase.

## Post-outcome diagnostic rules

The v1 cache diagnostic:

- fits on training worlds;
- reports only validation-world outcomes;
- consumes zero test labels;
- may compare layers and capture concatenations descriptively;
- cannot promote an architecture;
- cannot change v1 gates;
- can only motivate a separately versioned v2 with fresh test worlds.

## Prohibited shortcuts

- Reusing the v1 test role to select layers, prompts, capture streams, rates,
  heads, losses, or thresholds.
- Treating hidden-state R2, CKA, cosine, or reconstruction MSE as execution
  evidence.
- Concluding that facts are not linearly readable without an independent
  readability probe.
- Treating good renderer cosine as invariance when codes collapse.
- Crediting explicitly rendered density/reciprocity labels as an irreducible
  residual.
- Jointly training sender and receiver in the primary condition.
- Committing checkpoints, generated caches, `.agents/state/`, private datasets,
  API keys, or W&B credentials.

## Development workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
```

For a behavioral change:

1. add a focused test;
2. update protocol documentation before an outcome-bearing run;
3. preserve synthetic and v1 checkpoint compatibility;
4. validate real config/cache contracts without model calls where possible;
5. append a decision-log entry;
6. update `HANDOFF.md` and `docs/09_IMPLEMENTATION_STATUS.md`.

## Artifact hierarchy

Use evidence in this order:

1. frozen protocol/config and source identity;
2. cache validation and access audit;
3. machine decision and reducer;
4. held-out metrics;
5. prediction artifact;
6. training summary/history;
7. source-model branch diagnostics;
8. non-promotional localization;
9. W&B telemetry;
10. prose.

Adopted evidence belongs under `evidence/<run>/` with a content hash manifest.
Operational cluster state remains ignored under `.agents/state/`.
