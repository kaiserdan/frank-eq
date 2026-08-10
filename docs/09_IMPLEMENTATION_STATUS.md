# Implementation status

Snapshot: 2026-08-10

## Completed

| Component | Status |
|---|---|
| Synthetic Stage-0 implementation and adopted reference | complete |
| Real graph panel, causal cache records, validator, and frozen interrogator | complete |
| Founder training and held-sender onboarding | complete |
| Olivia/LUMI content-addressed workflows and W&B telemetry | complete |
| Real Stage-A v1 outcome and independent audit | complete, valid negative |
| Real Stage-A v2-1 outcome | complete, valid exact-pipeline negative |
| Supplemental v2-1 interpretation review | complete |
| Train/validation-only readability diagnostic | complete |
| Complete model-local public-head option | implemented, dormant |
| Proper new-user-turn `chat_turn` capture | implemented |
| Aggregate and per-founder native-competence qualification | implemented |
| Paired identical-panel prompt/capture comparison | implemented |
| Stage-Q cache-only workflow enforcement | implemented |
| Stage-Q LUMI development configs | frozen, not run |

## Adopted real outcomes

### Stage-A v1

`frank-eq-stagea-devg-v2` returned `STOP_OR_REVISE_STAGE0` for final-token
residual capture, local charts, shared public heads, and the frozen graph
interrogator. Its exact-pipeline negative remains valid.

### Stage-A v2-1

`frank-eq-stagea-lumi-v2` returned `STOP_OR_REVISE_STAGE0`:

| Metric | Value |
|---|---:|
| Native competence gain versus prior | -0.0521 |
| Held-out signature Brier | 0.2065; upper95 0.2421 |
| Fact accuracy | 0.5509; lower95 0.5120 |
| Cross-model retrieval | 0.1528; lower95 0.0972 |
| Wrong-world margin | -0.0607 |
| Held-sender retention | -0.3445 |
| Model-ID leakage over chance | 0.6389 |
| Renderer cosine | 0.9793 |

This is a decisive negative for the exact v2-1 shared-head oracle quotient.
Receiver execution remains blocked.

## v2-1 interpretation correction

The original adoption text overreached in calling native chat prompting
falsified.

Historical `prompt_format: chat` produced a cache ending at an assistant
-generation header. The operation string was appended after that header and was
therefore assistant content, not a new user turn. In addition, v1 and v2 used
different panel seeds, which changed worlds, operations, and split assignments.
No paired prompt contrast or interval for the v1/v2 difference existed.

The correct conclusion is:

> v2-1 falsifies the exact legacy chat-assistant-continuation pipeline, not all
> native chat prompting.

The competence gate also used a point estimate and ran during final evaluation,
so it did not stop training/test use as a true prerequisite.

See:

```text
evidence/real_stagea_lumi_v2/REVIEW.md
docs/16_STAGEA_V2_INTERPRETATION_CORRECTION.md
docs/17_STAGEQ_EXECUTION_AND_GATE_CONTRACT.md
```

## Stage-Q implementation

Stage Q corrects these deficiencies before another Stage-A registration.

### Proper conversation contract

`capture.prompt_format` now accepts:

- `raw`: historical raw text;
- `chat`: historical v2-1 assistant-continuation path, retained for reproduction;
- `chat_turn`: complete system/user/assistant prefix followed by a new user
  operation turn after capture.

For `chat_turn`, the backend tokenizes the full conversation and fails closed if
the cached prefix is not an exact token prefix of the branch conversation.
Optional `capture.chat_template_kwargs` are passed to the model template; Stage Q
freezes `enable_thinking: false`.

### Development competence qualifier

`src/frank_eq/qualification.py` computes founder competence on train/validation
worlds only:

```text
operation-prior Brier - frozen-source Brier
```

Model/renderer rows are averaged within world before a 2,000-replicate grouped
bootstrap. The held sender and test worlds are excluded. Both aggregate and
every individual founder lower confidence bound must be non-negative. The
output authorizes nothing beyond protocol design.

CLI:

```bash
python scripts/qualify_real_cache.py --cache <cache> --out <out>
```

### Paired Stage-Q comparison

`src/frank_eq/stageq.py` rejects caches unless their models, world IDs,
renderers, labels, operation registry, descriptors, and split manifest are
identical. It then bootstraps the paired candidate-minus-baseline Brier
improvement by world.

The paired result is a prompt-attribution diagnostic, not a source-qualification
prerequisite. It reports `prompt_effect_identified` separately from
`source_contract_qualified`.

CLI:

```bash
python scripts/compare_stageq_caches.py \
  --baseline-cache <legacy> \
  --candidate-cache <chat-turn> \
  --out <out>
```

### Frozen development configs

```text
configs/stageq/real_lumi_legacy_chat.yaml
configs/stageq/real_lumi_chat_turn.yaml
```

They share panel seed `20260811` and differ only in `prompt_format` plus run
identity/telemetry tags. They are for `cache,validate` only; their worlds are
permanently development-only. `src/frank_eq/workflow.py` rejects `diagnose`,
`train`, and `eval` for Stage-Q identities.

## Current continuation gate

The source contract qualifies only when:

```text
aggregate candidate competence lower95 >= 0
every individual founder competence lower95 >= 0
```

This permits drafting one fresh Stage-A registration. It does not authorize
running that registration, a receiver experiment, or a scientific claim.

The paired candidate improvement controls only a claim that corrected turn
placement helped. If it fails while source competence passes, the candidate may
still be frozen as the source prerequisite without a prompt-mechanism claim.

**Stage-Q result (2026-08-11):** both development conditions failed the gate
with fully negative world-grouped intervals (legacy −0.152 lower95, chat_turn
−0.162 lower95; all founders and families negative), and the paired prompt
effect was not identified (−0.020, lower95 −0.107). Machine decision
`STOP_STAGEQ_CANDIDATE`; artifacts under `runs/stageq/`; record in
`docs/13_STAGEA_V1_CORRECTION_LOG.md`.

Since source competence fails, the next work is development-only
checkpoint/task qualification (stronger checkpoints or a simpler formal task).
Do not modify the quotient architecture until a source/task pair passes both
aggregate and per-founder gates.

## Dormant future architecture

`model.public_head_scope: local` provides one complete compiler per source model
(chart plus fact/residual heads) while retaining externally fixed public
coordinates and a frozen interrogator. It should be considered only after
source competence passes.

A later Stage-A registration should separate:

- behavioral self-future prediction;
- oracle semantic grounding.

## Not authorized

- Stage-Q `diagnose`, training, or evaluation stages;
- reuse of Stage-Q worlds for confirmation;
- another Stage-A test run before Stage-Q source qualification;
- shared-head oracle-quotient rescue variants;
- receiver-native execution;
- rate-matched communication, confirmation, safety, or harm-tail claims.
