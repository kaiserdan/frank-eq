# Frank-EQ handoff

Snapshot: 2026-08-10

## Current decision

Synthetic Stage 0 passes as implementation evidence.

Real Stage-A v1 has one adopted outcome:

```text
run: frank-eq-stagea-devg-v2
cluster/job: LUMI dev-g / 20942127
decision: STOP_OR_REVISE_STAGE0
workflow integrity: passed
scientific claim authorized: false
```

Real Stage-A v2-1 (native chat-template competence) is also an adopted
negative:

```text
run: frank-eq-stagea-lumi-v2
cluster/job: LUMI dev-g / 20952565
decision: STOP_OR_REVISE_STAGE0
native competence gain: -0.0521 (vs -0.0603 diagnostic) — prompt surface
is not the competence bottleneck
```

The v1 negative is immutable for the exact v1 pipeline. The v2-1 negative is
immutable for the exact v2-1 registration; the chat-template hypothesis is
falsified. The failure source is unresolved beyond "source task competence is
the binding constraint; hidden-state capture carries readable self-future
signal". Read `docs/12_STAGEA_V1_AUDIT_AND_V2_PROTOCOL.md`,
`docs/14_STAGEA_V2_PROTOCOL.md`, and `evidence/real_stagea_lumi_v2/AUDIT.md`.

## What v1 indicates

Observed failures:

```text
fact accuracy:                     0.5296
reconstructed majority baseline:  0.5278
held-out signature Brier:          0.1729  (upper 95% 0.1978)
cross-model retrieval top-1:       0.2083
wrong-world margin:               -0.0575
held-sender retention:             0.4717
model-ID leakage over chance:      0.6528
source branch accuracy to oracle:  0.4392
operation-prior accuracy:          0.6354
```

Renderer cosine `0.9925`, residual gain, and quantization retention passed, but
none is a positive quotient result:

- renderer stability coexists with collapse/model identity;
- density and reciprocity targets are explicitly printed in the prefix;
- quantization preserves an already-failing code.

The v1 public compiler uses model-local charts but shared fact/residual heads.
Held onboarding updates only the held chart, forcing it into a founder-induced
private gauge. The capture is also limited to the final-token residual at three
depths although literal future branching uses the full KV cache.

Finally, the primary target is the formal oracle signature, while the project's
definition of operational equivalence concerns the model's own future branch
distribution. V2 must separate those objects.

## Immediate next action

Stage-A v2-1 (native chat-template competence) ran on LUMI dev-g and is
**falsified** as an adopted negative:

```text
run:      frank-eq-stagea-lumi-v2 (Slurm 20952565, source 1aa741d5df31)
decision: STOP_OR_REVISE_STAGE0
native competence gain: -0.0521  (v1 diagnostic: -0.0603)
held-out Brier upper:   0.2421   (v1: 0.1978) | held retention: -0.34 (v1: 0.47)
```

The prompt surface is not the competence bottleneck: the frozen source models'
branches are oracle-incompetent on the 6-entity graph task under both raw and
chat prompts (anti-predictive vs operation priors). Chat made own-future
signatures more readable (qwen3 balacc 0.89) but linear fact readability
dropped. Evidence: `evidence/real_stagea_lumi_v2/`; decision log + correction
log entries appended.

Next hypothesis candidates (branch A continuation — a user decision):
stronger checkpoints (e.g. Qwen3-4B-class founders) or a task revision. Either
requires a new versioned registration with a fresh untouched test role;
receiver execution stays locked. No tuning occurred under this outcome.

## Decision tree after localization

```text
native branches do not beat priors
  → freeze a prompt/task competence prerequisite

native competence passes but own future signature unreadable
  → expand capture to token-sequence or selected-KV state

own future signature readable but oracle facts unreadable
  → separate behavioral operational state from semantic grounding

raw targets readable but trained quotient fails
  → use complete model-local compilers and revise the joint objective
```

The code supports the last architecture behind:

```yaml
model:
  public_head_scope: local
```

This makes chart, fact head, and residual head model-local while leaving public
coordinates and the interrogator shared. It is not yet an authorized v2 config.

## Requirements for Stage-A v2

Before any fresh test run:

1. new world seed and untouched test role;
2. exact checkpoint revisions in config;
3. frozen native-competence gate;
4. explicit capture stream/pooling contract;
5. sampled KV-reuse versus exact-replay numerical parity;
6. self-future and oracle metrics in separate namespaces;
7. prior-relative fact/signature metrics;
8. renderer invariance conditioned on world specificity;
9. local/shared compiler scope frozen in advance;
10. real-specific decision schema.

Receiver-native execution remains locked.

## Evidence and repository hygiene

Adopted v1 evidence is under:

```text
evidence/real_stagea_devg_v2/
  decision.json
  metrics.json
  run_manifest.json
  AUDIT.md
  audit.json
  manifest.json
```

The evidence package is intentionally small; generated caches and checkpoints
remain external. `.agents/state/` is local operator state and is ignored. Do
not commit source archives, stale scheduler snapshots, or fetched run trees.

## Known integrity gaps to repair in v2

- v1 YAML omitted exact checkpoint revision pins;
- v1 run manifest had no Git commit identity;
- all KV branches executed, but no registered replay-parity sample was stored;
- the adopted evidence package does not contain the full training history,
  cache metadata, predictions, or evaluation artifact manifest;
- the historical v1 decision has a synthetic-scope metadata label due to the
  old reducer. The outcome values remain valid; future real decisions use a
  real-specific schema.

## Do not do next

Do not tune the v1/v2 test gates, rerun the same deterministic tests, add a
target hidden-state decoder, or start receiver execution. The chat-template
hypothesis is falsified (v2-1 adopted negative). The next step is a user
decision: one new versioned registration raising source task competence
(stronger checkpoints) or revising the task, with a fresh untouched test role
and all ten registration prerequisites. Any capture/prompt/checkpoint change
must satisfy state-capture-before-reveal and the frozen parity regime
(exclusive kv_reuse, registered stack-noise audit).
