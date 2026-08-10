# Agent operating contract

## Mission

Frank-EQ studies whether an LLM state formed before an operation is revealed can
be compiled into a public, future-defined operational state. The project is not
an unrestricted hidden-state architecture search. Every experiment must answer
a frozen question, preserve information boundaries, and fail closed.

## Reading order

1. `README.md`
2. `HANDOFF.md`
3. `evidence/real_stagea_lumi_v2/REVIEW.md`
4. `docs/16_STAGEA_V2_INTERPRETATION_CORRECTION.md`
5. `docs/15_STAGEA_V2_REVIEW_AND_STAGEQ.md`
6. `docs/03_INFORMATION_ACCESS_CONTRACT.md`
7. `docs/05_GATES_AND_STOP_RULES.md`
8. `docs/12_STAGEA_V1_AUDIT_AND_V2_PROTOCOL.md`
9. `docs/09_IMPLEMENTATION_STATUS.md`
10. `docs/10_DECISION_LOG.md`

For cluster work, also read
`.agents/skills/lumi-cluster-runner/SKILL.md` or
`.agents/skills/olivia-cluster-runner/SKILL.md`.

## Current authority

Synthetic Stage 0 passes as implementation evidence.

Real Stage-A v1 and v2-1 are adopted exact-pipeline negatives. Their machine
`STOP_OR_REVISE_STAGE0` decisions remain authoritative. Neither permits receiver
execution or a scientific claim.

The only authorized next execution is **Stage Q**, a development-only paired
source-competence qualification:

```text
configs/stageq/real_lumi_legacy_chat.yaml
configs/stageq/real_lumi_chat_turn.yaml
```

Run `cache,validate` only. Then use:

```bash
python scripts/qualify_real_cache.py --cache <cache> --out <out>
python scripts/compare_stageq_caches.py \
  --baseline-cache <legacy-cache> \
  --candidate-cache <chat-turn-cache> \
  --out <out>
```

Stage-Q worlds are permanently development-only. A pass permits protocol design,
not a fresh outcome run by itself.

## Interpretation invariants

### Preserve exact negatives without broadening them

V2-1 failed under a legacy chat construction where the cached prefix ended at
an assistant-generation marker and the operation was appended as assistant
content. Do not call this a general native-chat falsification.

V1 and v2 used different panel seeds. Their point estimates are not a paired
prompt ablation and must not be described as unchanged within noise.

### State formation precedes operation reveal

No operation, query, target, or future label may affect the captured state.

For `prompt_format: chat_turn`, the legal prefix is:

```text
system contract
user world statement
assistant fixed acknowledgement
```

The operation is revealed afterward as a new user turn. The backend must verify
that the cached token IDs are an exact prefix of the full branch conversation.

Historical `prompt_format: chat` is reproduction-only.

### Competence is a prerequisite, not an evaluator side metric

Before representation training or claim-bearing test access, frozen founder
branches must beat an operation-wise training prior on validation worlds. The
paired unit is a world, and the lower grouped 95% bound must be non-negative.

A point estimate computed after test evaluation does not satisfy this rule.

### Paired comparisons require identical panels

A prompt/capture contrast is valid only when the two caches have identical:

- model roster and revisions;
- world IDs and renderer IDs;
- fact, residual, and oracle labels;
- operation registry and descriptors;
- split manifest;
- branch execution mode.

`src/frank_eq/stageq.py` fails closed on any mismatch.

### Self-future state and oracle semantics are distinct

`model_signatures` describe what the frozen source model will do.
`signatures` describe externally correct outcomes. They require separate metric
namespaces and may support different conclusions.

Do not call an oracle-supervised world decoder a pure operational quotient.

### The complete causal state is not the final-token residual

V1/v2 captured only final-token residual vectors at three depths while literal
branching used the complete KV cache. Negative results cannot be generalized to
the full runtime state.

Any future capture census must be selected on development worlds and frozen
before fresh Stage-A test access.

### KV reuse and exact replay are non-equivalent on the current stack

The observed branch-probability differences reached approximately 0.11. A pure
KV cache is internally valid; exact replay must not be substituted or mixed.
The 0.33 stack threshold is an alarm, not an equivalence claim.

### Complete local compilers, public semantics

Only public coordinate meaning and the interrogator need to be shared.
`model.public_head_scope: local` gives each model a complete local compiler.
It remains dormant until source competence passes and one fresh Stage-A
registration is frozen.

### Invariance requires specificity

Renderer cosine cannot pass alone. Positive world retrieval/wrong-world margin
and low model identity leakage are required to rule out collapse.

### Declared controls are not discoveries

Density and reciprocity are printed in the current prefixes. Their residual
readability is a declared-global control, not evidence for hidden irreducible
operational state.

### Machine artifacts outrank prose

Use evidence in this order:

1. frozen protocol/config and source identity;
2. causal cache validation;
3. machine qualification or decision;
4. grouped metric artifact;
5. predictions/training history;
6. W&B telemetry;
7. prose.

## Stage-Q operating procedure

1. Confirm the two Stage-Q configs differ only in run identity, telemetry tags,
   and `capture.prompt_format`.
2. Submit `cache,validate` for the legacy condition.
3. Submit `cache,validate` for the `chat_turn` condition.
4. Fetch and verify both caches.
5. Run single-cache qualification on each.
6. Run the paired cache comparison.
7. Adopt a small hash-verified evidence package before drawing a conclusion.
8. Do not run quotient training or evaluation for either Stage-Q config.

Stage Q passes only if:

```text
chat-turn competence lower95 >= 0
paired chat-turn-minus-legacy improvement lower95 >= 0
```

Every Stage-Q output must keep test, receiver, new-outcome, and claim
authorization false.

## Decision tree

### Stage Q fails

Screen stronger checkpoints or a simpler task on development-only caches. Do
not change the quotient architecture yet.

### Competence passes but paired prompt improvement fails

Treat the passing condition as a prerequisite only. Make no prompt-mechanism
claim.

### Both checks pass

Draft exactly one fresh Stage-A registration with:

- fresh claim-bearing worlds;
- complete model-local compilers;
- behavioral self-future and oracle semantic channels separated;
- a development-selected, frozen capture stream;
- receiver work still locked.

## Prohibited shortcuts

- Running `train` or `eval` with Stage-Q configs.
- Reusing Stage-Q worlds as confirmation worlds.
- Calling legacy `chat` a proper user-operation turn.
- Comparing unpaired v1/v2 estimates as a prompt ablation.
- Selecting models, tasks, layers, or thresholds on a new Stage-A test role.
- Rescuing the shared-head oracle quotient with unregistered variants.
- Mixing replay and KV branches.
- Jointly training sender and receiver in the primary condition.
- Committing generated caches, checkpoints, `.agents/state/`, API keys, or W&B
  credentials.

## Development validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
```

For any behavioral change, add a focused test and update the relevant protocol
before running an outcome-bearing job.
