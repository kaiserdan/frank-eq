# Implementation status

Snapshot: 2026-08-10

## Completed

| Component | Status |
|---|---|
| Synthetic Stage-0 implementation and adopted reference | complete |
| Real relational panel, capture backend, cache records, and validator | complete |
| Founder training and held-sender onboarding | complete |
| Frozen graph interrogator and query-conditioned packet | complete |
| Olivia/LUMI content-addressed workflows | complete |
| Fail-open W&B telemetry | complete |
| First real Stage-A v1 outcome | complete, valid negative |
| Stage-A v1 independent audit and evidence manifest | complete |
| Train/validation-only real-cache localization diagnostic | implemented |
| Complete model-local public-head option | implemented, dormant |
| Real-specific future decision metadata | implemented |
| Operational `.agents/state/` exclusion | implemented |

## Adopted real Stage-A v1 outcome

`frank-eq-stagea-devg-v2` completed on LUMI job `20942127` and returned
`STOP_OR_REVISE_STAGE0`.

Engineering integrity passed. The decision is scientifically valid for:

```text
last-token residual at three depths
→ local chart
→ shared fact/residual heads
→ frozen graph interrogator
```

The result does not falsify a future-defined quotient over the complete runtime
state.

### Main observations

| Metric | Value | Interpretation |
|---|---:|---|
| Fact accuracy | 0.5296 | only +0.0019 over reconstructed global-majority baseline |
| Held-out signature Brier | 0.1729 | better than operation-prior 0.2080, but gate upper bound fails |
| Cross-model retrieval | 0.2083 | weak world identity across models |
| Wrong-world margin | -0.0575 | collapse/wrong-world preference |
| Held retention | 0.4717 | poor establishment |
| Model leakage over chance | 0.6528 | public code remains strongly model-specific |
| Source branch accuracy | 0.4392 | below operation-prior 0.6354 |
| Renderer cosine | 0.9925 | not meaningful without non-collapse |
| Quantization retention | 0.9962 | preserves failed code |
| Residual gain | positive | confounded by explicit density/reciprocity tags |

## Audit correction

The earlier status described the failure as localized to capture sufficiency and
said facts were not linearly readable. That claim was not tested. The executed
fact head is nonlinear and jointly optimized with seven other objectives, and
no independent layer/model readability probe exists.

The failure is currently compatible with:

- insufficient last-token capture;
- inadequate pooling of a distributed KV/token state;
- weak native task competence under raw prompts;
- shared-head/private-gauge mismatch;
- sample/parameter mismatch;
- joint-objective interference;
- absent semantic grounding despite readable own-future state.

## Localization outcome (diagnostic executed)

The existing-cache diagnostic ran on train/validation worlds only
(`runs/diagnostics/frank-eq-stagea-devg-v2/localization.json`; full record in
`docs/13_STAGEA_V1_CORRECTION_LOG.md`). Machine recommendation:

```text
FIX_NATIVE_COMPETENCE_BEFORE_LATENT_REVISION
```

Founder native branch Brier gain over the operation-wise prior is −0.060 on
validation worlds (negative in every operation family), while facts
(+0.006..+0.035 gain), oracle signatures (+0.062..+0.067), and own-future
signatures (+0.0008..+0.0115) are all readable from the existing capture.
Renderer-transfer fact probes lose to the coordinate prior (−0.18..−0.26),
so the v1 renderer-cosine pass is treated as invariance without specificity.
Residual R2 0.64..0.69 is a declared-global control (density/reciprocity tags
are rendered in the prefix) and carries no hidden-operational evidence.

Per the frozen decision tree the next hypothesis is a model/task competence
prerequisite (e.g. native chat template or stronger checkpoints); it still
requires a versioned Stage-A v2 registration with a fresh test role.

## New localization implementation

`frank-eq diagnose-real-cache` fits fixed-ridge probes using training worlds and
reports only validation worlds. It separately tests facts, oracle signatures,
own future signatures, residual coordinates, renderer transfer, and native
competence. It touches zero test labels and cannot promote.

The real workflow now accepts the optional stage:

```text
cache,validate,diagnose,train,eval
```

## Dormant v2 architecture support

`model.public_head_scope` accepts:

- `shared`: exact historical v1 behavior and checkpoint compatibility;
- `local`: complete local compiler per model (chart + fact/residual heads).

The public operation semantics and interrogator remain shared. The local option
is implementation readiness only; no outcome-bearing v2 config is frozen.

## Stage-A v2-1 registration (frozen, awaiting ratification)

Per decision-tree branch A (native competence prerequisite), exactly one
versioned v2 hypothesis is frozen: same checkpoints and panel geometry, capture
prefix wrapped in each model's native chat template. Protocol:
`docs/14_STAGEA_V2_PROTOCOL.md`; config: `configs/stage0/real_lumi_v2.yaml`
(new world seed 20260810, required revision pins, chat capture,
KV-versus-replay parity audit, frozen native-competence gate, residual check as
declared-global control). Implementation and tests are complete (50/50 green,
dry-run submission succeeds); no outcome-bearing run is submitted until the
registration is ratified.

## Not authorized

- reuse of the v1 test role for selection;
- a new Stage-A outcome run (localization completed; a v2 registration must be
  frozen and versioned first);
- receiver-native execution;
- rate-matched transcript/summary experiments;
- confirmation or locked roles;
- safety or harm-tail claims.

## Known v1 evidence limitations

- checkpoint revisions were not pinned in YAML;
- run manifest `git_commit` is null;
- KV reuse was counted but not numerically checked against replay;
- the small adopted evidence copy omits cache metadata, predictions, training
  history, and evaluator artifact manifest;
- v1's decision scope string incorrectly says synthetic due to the historical
  reducer. Future decisions use a real-specific schema.

See `docs/12_STAGEA_V1_AUDIT_AND_V2_PROTOCOL.md`.
