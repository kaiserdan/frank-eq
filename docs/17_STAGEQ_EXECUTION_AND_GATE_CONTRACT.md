# Stage-Q execution and promotion contract

Status: frozen development protocol. This document governs the next permitted
execution and supersedes any prose implying that another Stage-A outcome run is
already authorized.

## Execution role

The two configs under `configs/stageq/` are development-only:

```text
real_lumi_legacy_chat.yaml
real_lumi_chat_turn.yaml
```

The workflow infers the `stageq` role from the registered config path or
`frank-eq-stageq-*` run identity and permits only:

```text
cache,validate
```

Requests containing `diagnose`, `train`, or `eval` fail before the run manifest
is created. Stage-Q analysis is performed afterward with the dedicated
qualification scripts, which consume only train/validation worlds.

## Primary source qualification

For each held-out operation, estimate an oracle prior using training worlds.
For each founder on validation worlds, compute:

```text
Brier gain = Brier(oracle, training-world operation prior)
           - Brier(oracle, founder branch probability)
```

Average renderer views within model/world, then bootstrap worlds with 2,000
replicates.

The candidate source contract qualifies only if both conditions hold:

```text
aggregate founder lower95 >= 0
every individual founder lower95 >= 0
```

This prevents one competent founder from masking another source that is not
usable under the registered operation contract.

A source-qualification pass permits drafting exactly one new Stage-A
registration with fresh worlds. It does not authorize running that
registration, opening its test role, building a receiver experiment, or making
a scientific claim.

## Secondary paired prompt/capture comparison

The legacy and candidate caches must be identical in:

- checkpoint roster and requested/observed revisions;
- world, model, and renderer row identities;
- fact, residual, and oracle labels;
- operation registry and descriptors;
- split manifest;
- selected layers, answer labels, and branch-mode accounting.

The paired unit is again a world. Positive improvement means the proper
`chat_turn` candidate has lower Brier than the legacy assistant-continuation
baseline on the same model/world/renderer/operation rows.

A prompt-effect claim is identified only if:

```text
paired candidate-minus-baseline improvement lower95 > 0
```

This strict inequality prevents identical conditions from being labeled an
effect. The contrast is not a prerequisite for using an independently competent
source contract. If source competence passes but paired improvement does not,
the candidate may be used as a frozen prerequisite while making no claim that
corrected turn placement caused an improvement.

## Stage-Q decisions

The paired artifact reports two independent decisions:

1. `source_contract_qualified`: aggregate and every founder competence gate;
2. `prompt_effect_identified`: strictly positive paired candidate-minus-legacy
   lower confidence bound.

Only the first controls whether one Stage-A protocol may be drafted. The second
controls only whether a prompt-placement effect may be claimed.

## Data and model roles

Stage-Q uses:

- training worlds to estimate operation priors;
- validation worlds for all qualification intervals;
- zero test-world labels;
- founder rows only in the qualification statistic.

The current cache format still extracts the model registered in the historical
`held` slot, although its rows are excluded from every Stage-Q statistic. That
checkpoint is therefore **development-exposed** and may not be presented as the
held sender in a later Stage-A registration. A future Stage-A protocol must
choose a new unopened held checkpoint/model role.

The Stage-Q panel seed `20260811` is permanently development-only and may not be
reused for a later Stage-A test or confirmation role.

## Frozen versus exploratory settings

The primary CLIs read thresholds, bootstrap replicate counts, and seeds from the
cache's frozen config. They expose no command-line threshold override. Internal
exploratory overrides are marked `protocol_mode=exploratory_override` and are
automatically demoted to `NO_PROMOTION_EXPLORATORY_OVERRIDE`.

## Required artifacts

```text
legacy cache validation
candidate cache validation
legacy qualification.json
candidate qualification.json
paired comparison.json
compact verification summary
SHA-256 evidence manifest
explicit authorization block with all fields false
```

W&B is optional telemetry and never carries promotion authority.
