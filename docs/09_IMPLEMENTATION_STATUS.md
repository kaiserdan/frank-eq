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
| Development native-competence qualification with grouped interval | implemented |
| Paired identical-panel Stage-Q comparison | implemented |
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
bootstrap. The held sender and test worlds are excluded. The output authorizes
nothing beyond diagnosis.

CLI:

```bash
python scripts/qualify_real_cache.py --cache <cache> --out <out>
```

### Paired Stage-Q comparison

`src/frank_eq/stageq.py` rejects caches unless their models, world IDs,
renderers, labels, operation registry, descriptors, and split manifest are
identical. It then bootstraps the paired candidate-minus-baseline Brier
improvement by world.

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

They share panel seed `20260811` and differ only in `prompt_format`. They are
for `cache,validate` only; their worlds are permanently development-only.

## Current continuation gate

Stage Q must pass both:

```text
candidate competence lower95 >= 0
paired candidate improvement lower95 >= 0
```

A pass permits drafting one fresh Stage-A registration. It does not authorize a
receiver experiment or scientific claim.

If Stage Q fails, the next work is development-only checkpoint/task
qualification. Do not modify the quotient architecture until a source/task pair
beats the operation prior with a non-negative lower confidence bound.

## Dormant future architecture

`model.public_head_scope: local` provides one complete compiler per source model
(chart plus fact/residual heads) while retaining externally fixed public
coordinates and a frozen interrogator. It should be considered only after
source competence passes.

A later Stage-A registration should separate:

- behavioral self-future prediction;
- oracle semantic grounding.

## Not authorized

- Stage-Q training or evaluation stages;
- reuse of Stage-Q worlds for confirmation;
- another Stage-A test run before Stage-Q qualification;
- shared-head oracle-quotient rescue variants;
- receiver-native execution;
- rate-matched communication, confirmation, safety, or harm-tail claims.
