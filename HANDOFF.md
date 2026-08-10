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
4. `docs/17_STAGEQ_EXECUTION_AND_GATE_CONTRACT.md`

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

## Current authorized action: Stage Q completed — candidate stopped

Stage Q ran both development conditions on LUMI dev-g (panel seed 20260811,
cache,validate only, both verified). Machine verdict `STOP_STAGEQ_CANDIDATE`:

```text
source competence (world-grouped 95% CI, founders, held-out ops):
  legacy    aggregate -0.098  [-0.152, -0.040]   FAIL
  chat_turn aggregate -0.118  [-0.162, -0.068]   FAIL
  every founder and every operation family negative in both conditions
prompt effect: paired improvement -0.020 [-0.107, 0.071] -> NOT identified
```

The proper new-user-turn reveal is not better than the legacy
assistant-continuation construction, and neither is oracle-competent. The
prompt-surface question is now answered under correct paired methodology.

Next per `docs/15` §6: screen stronger source checkpoints or a simpler formal
task on development-only competence caches; freeze the first combination whose
aggregate and every founder lower confidence bound are non-negative before any
Stage-A representation training. This is a user decision (checkpoint roster
and/or task change); the latent architecture is not modified. Stage-Q
artifacts: `runs/stageq/` (ignored); record in `docs/13_STAGEA_V1_CORRECTION_LOG.md`.

## Screening series: stopped without a passing combination

Three development screens ran (Stage-Q pair plus qwen3-4b/qwen3-8b rosters);
none passed the frozen aggregate + per-founder gate:

```text
Stage-Q (0.6B+1.7B)      aggregate lower95 -0.162
screen-strong (1.7B+4B)  aggregate lower95 -0.262   (qwen3-1.7b -0.454)
screen-8b (4B+8B)        aggregate lower95 -0.129   (qwen3-8b -0.112)
```

Wall: multi-edge structural operations fail even at 8B (mutual -0.758,
compose -0.299) while single-edge/global ops pass at 8B (inverse +0.136,
reciprocity +0.147). Scale helps slowly; the 6-entity task depth is the
binding constraint. Series stopped by user decision; next options for a future
decision: 4-entity task simplification, Qwen3-14B scale push, or query-contract
revision — all must pass the same development qualification first. Staged on
LUMI: Qwen3-1.7B/4B/8B (pinned revisions in `configs/stageq/real_lumi_screen_*.yaml`).

## Decision after Stage Q

### Source competence fails

Do not revise the latent architecture. Screen stronger checkpoints or a simpler
formal task on development-only competence caches. Freeze a source/task pair
only after the aggregate and every founder lower confidence bound are
non-negative.

### Source competence passes but paired prompt improvement fails

The candidate may be frozen as the source prerequisite for one Stage-A
registration, but no prompt-mechanism claim is supported.

### Source competence and paired prompt improvement pass

Draft one fresh Stage-A registration and retain the paired prompt result as
secondary evidence. The new registration should use:

- fresh claim-bearing worlds never used by Stage Q;
- complete model-local compilers (`public_head_scope: local`);
- a behavioral self-future channel distinct from oracle semantic grounding;
- a development-selected and then frozen capture representation;
- no receiver experiment until the representation gate passes.

## Implemented surfaces

```text
src/frank_eq/data/hf_backend.py       legacy chat + proper chat_turn capture
src/frank_eq/qualification.py         aggregate and per-founder competence intervals
src/frank_eq/stageq.py                paired identical-panel prompt comparison
src/frank_eq/workflow.py              Stage-Q cache-only role enforcement
scripts/qualify_real_cache.py         single-cache qualification CLI
scripts/compare_stageq_caches.py      paired Stage-Q comparison CLI
```

Historical `prompt_format: chat` remains supported only to reproduce the legacy
turn placement. Future chat competence work must use `chat_turn`.

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

- Running Stage-Q configs through `diagnose`, `train`, or `eval`.
- Reusing Stage-Q worlds as a Stage-A confirmation role.
- Treating v1/v2 point differences as a prompt ablation.
- Resuming the shared-head oracle quotient unchanged.
- Selecting layers, tasks, checkpoints, or thresholds on a new Stage-A test set.
- Starting receiver-native execution.
