---
name: frank-eq-research
description: Audit and extend Frank-EQ future-defined operational-state experiments while preserving causal order, development-only source gates, and immutable negative evidence.
---

# Frank-EQ research skill

## Use this skill when

- auditing an adopted Stage-A result;
- building or comparing source-competence caches;
- changing prompt/capture contracts;
- implementing model-local compilers or operational channels;
- preparing a frozen Stage-A registration or evidence package.

## Current authority

Read `AGENTS.md`, `HANDOFF.md`, and
`docs/17_STAGEQ_EXECUTION_AND_GATE_CONTRACT.md` before acting.

Stage-A v1 and v2-1 are valid exact-pipeline negatives. V2-1 does not falsify
native chat prompting generally: its operation was appended inside the assistant
turn, and its panel was not paired with v1.

The only authorized execution is the development-only Stage-Q pair:

```text
configs/stageq/real_lumi_legacy_chat.yaml
configs/stageq/real_lumi_chat_turn.yaml
```

Run `cache,validate` only. The workflow rejects `diagnose`, `train`, and `eval`
for these configs. Do not access a fresh Stage-A test role.

## Mandatory Stage-Q workflow

1. Confirm both configs have identical checkpoints, panel seed, operation
   registry construction, split, renderer count, branch mode, and template
   kwargs.
2. Confirm the candidate uses `prompt_format: chat_turn`.
3. Verify exact token-prefix continuity during cache construction.
4. Fetch and verify both caches.
5. Run:

```bash
python scripts/qualify_real_cache.py --cache <legacy> --out <legacy-out>
python scripts/qualify_real_cache.py --cache <candidate> --out <candidate-out>
python scripts/compare_stageq_caches.py \
  --baseline-cache <legacy> \
  --candidate-cache <candidate> \
  --out <paired-out>
```

6. Check that every output reports zero test worlds, excludes the held sender,
   and keeps all authorization fields false.
7. Adopt only a small hash-verified evidence package.

## Stage-Q decisions

The candidate source contract qualifies only if:

```text
aggregate competence lower95 >= 0
every individual founder competence lower95 >= 0
```

This permits one fresh Stage-A protocol to be drafted, not run.

The paired candidate-minus-legacy interval controls a separate prompt-effect
claim. If its lower bound is negative, make no turn-placement claim; this does
not invalidate an independently passing source competence gate.

## Causal and statistical invariants

- Form state before any operation/query reveal.
- A proper chat branch reveals the operation as a new user turn, never as
  assistant content.
- Prompt comparisons must use identical world/model/operation rows.
- Average renderer/model views within world before bootstrap resampling.
- Use training worlds to fit priors and validation worlds for qualification.
- Require every founder to pass; do not let an aggregate mask one sender.
- Keep test worlds and the held sender unopened during Stage Q.
- Keep self-future behavior separate from oracle semantic correctness.
- Do not treat renderer cosine as invariance without specificity and low model
  leakage.
- Do not treat explicitly printed density/reciprocity labels as a discovered
  residual.
- Do not mix KV-reuse and replay branches.

## After Stage Q

If source competence fails, qualify stronger models or a simpler task on
development data. Do not revise the latent architecture first.

If source competence passes, draft one fresh Stage-A registration using complete
local compilers, separate behavioral/semantic channels, fresh worlds, and a
frozen capture representation. Record a prompt-effect claim only if the paired
interval also passes. Receiver execution remains locked.

## Validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
python scripts/validate_repo.py
```

## Evidence standard

Every adopted artifact set must include:

```text
frozen config/source identity
causal cache validation
machine qualification or decision
grouped intervals and paired units
compact verification summary
SHA-256 manifest
explicit authorization boundary
```

W&B is secondary telemetry and cannot authorize continuation.
