# Frank-EQ handoff

Snapshot: 2026-08-10

## Current authority

Synthetic Stage 0 passes as implementation evidence only.

Two real Stage-A runs are adopted negatives:

```text
v1: frank-eq-stagea-devg-v2     LUMI 20942127   STOP_OR_REVISE_STAGE0
v2: frank-eq-stagea-lumi-v2     LUMI 20952565   STOP_OR_REVISE_STAGE0
```

Both workflows completed and their exact-pipeline negative decisions remain
valid. Neither authorizes a scientific claim or receiver execution.

The current broad interpretation is narrower than the prose originally adopted
with v2-1. Read, in order:

1. `evidence/real_stagea_lumi_v2/REVIEW.md`
2. `docs/16_STAGEA_V2_INTERPRETATION_CORRECTION.md`
3. `docs/15_STAGEA_V2_REVIEW_AND_STAGEQ.md`

## What v2-1 establishes

The following exact pipeline failed:

```text
chat-formatted world prefix ending at assistant generation header
→ operation appended as assistant content
→ final-token residuals at depths 0.35 / 0.60 / 0.85
→ model-local chart
→ shared fact/residual heads
→ frozen graph interrogator
```

Key values:

```text
native competence gain vs operation prior:  -0.0521
held-out signature Brier:                     0.2065 (upper95 0.2421)
fact accuracy:                                0.5509 (lower95 0.5120)
cross-model retrieval:                        0.1528 (lower95 0.0972)
wrong-world margin:                          -0.0607
held-sender retention:                       -0.3445
model-ID leakage over chance:                 0.6389
```

The shared-head oracle quotient should not proceed to a receiver.

## What v2-1 does not establish

It does not isolate native chat prompting as the failure source.

- `prompt_format: chat` ended the cache at the assistant-generation header;
  the operation was appended inside the assistant turn rather than as a new
  user message.
- Changing `panel.seed` between v1 and v2 changed worlds, operation instances,
  and split/holdout assignments. The two competence point estimates are not a
  paired prompt comparison.
- Native competence used a point estimate and was evaluated after training and
  test-world scoring, so it did not function as a prerequisite.
- KV reuse was internally consistent, but replay differed by as much as 0.1089;
  the two execution paths are not interchangeable on this stack.

The exact v2-1 negative is preserved. The statements that native chat was
falsified, the prompt surface was not the bottleneck, or the values were
unchanged within noise are withdrawn.

## Current authorized action: Stage Q only

Stage Q is a development-only competence qualification. It compares two caches
on exactly the same models, worlds, renderers, operation registry, split, and KV
branch path:

```text
configs/stageq/real_lumi_legacy_chat.yaml
configs/stageq/real_lumi_chat_turn.yaml
```

The candidate `chat_turn` contract caches:

```text
system contract
user world statement
assistant fixed acknowledgement
```

and reveals the operation after capture as:

```text
user operation
assistant generation boundary
```

The backend fails closed unless the cached prefix token IDs are an exact prefix
of the full conversation.

Build only `cache,validate` for each condition. Then run:

```bash
python scripts/qualify_real_cache.py \
  --cache <legacy>/runs/cache \
  --out runs/stageq/legacy-qualification

python scripts/qualify_real_cache.py \
  --cache <chat-turn>/runs/cache \
  --out runs/stageq/chat-turn-qualification

python scripts/compare_stageq_caches.py \
  --baseline-cache <legacy>/runs/cache \
  --candidate-cache <chat-turn>/runs/cache \
  --out runs/stageq/paired-comparison
```

The qualification uses founder models and train/validation worlds only. It
reports a world-grouped 95% interval. Test worlds and the held sender are not
used.

Stage Q passes only when:

```text
candidate competence lower95 >= 0
paired candidate-minus-legacy Brier improvement lower95 >= 0
```

Every Stage-Q artifact keeps all claim, receiver, fresh-test, and outcome-run
authorization fields false.

## Decision after Stage Q

### Stage Q fails

Do not revise the latent architecture. Screen stronger checkpoints or a simpler
formal task on development-only competence caches. Freeze a source/task pair
only after its lower confidence bound is non-negative.

### Competence passes but paired prompt improvement fails

The candidate is usable as a source prerequisite, but no prompt-mechanism claim
is supported.

### Both checks pass

Draft one fresh Stage-A registration. It should use:

- fresh claim-bearing worlds never used by Stage Q;
- complete model-local compilers (`public_head_scope: local`);
- a behavioral self-future channel distinct from oracle semantic grounding;
- a development-selected and then frozen capture representation;
- no receiver experiment until the representation gate passes.

## Implemented surfaces

```text
src/frank_eq/data/hf_backend.py       legacy chat + proper chat_turn capture
src/frank_eq/qualification.py         development native-competence interval
src/frank_eq/stageq.py                paired identical-panel comparison
scripts/qualify_real_cache.py         single-cache qualification CLI
scripts/compare_stageq_caches.py      paired Stage-Q comparison CLI
```

Historical `prompt_format: chat` remains supported only to reproduce v2-1.
Future chat competence work must use `chat_turn`.

## Evidence boundary

The adopted v2 package remains immutable:

```text
evidence/real_stagea_lumi_v2/
  AUDIT.md
  decision.json
  metrics.json
  run_manifest.json
  verification_summary.json
  manifest.json
```

The independent correction is supplemental:

```text
evidence/real_stagea_lumi_v2/
  REVIEW.md
  review.json
  review_manifest.json
```

Generated caches, checkpoints, scheduler state, and source archives remain
outside Git under `.agents/state/` and `runs/`.

## Prohibited next actions

- Running Stage-Q configs through `train` or `eval`.
- Reusing Stage-Q worlds as a Stage-A confirmation role.
- Treating v1/v2 point differences as a prompt ablation.
- Resuming the shared-head oracle quotient unchanged.
- Selecting layers, tasks, checkpoints, or thresholds on a new Stage-A test set.
- Starting receiver-native execution.
