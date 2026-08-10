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

The negative decision is correct for the exact v1 pipeline and remains
immutable. The previous prose localization to “capture sufficiency” was too
strong. The failure source is unresolved because v1 did not run independent
readability/capacity probes.

Read `docs/12_STAGEA_V1_AUDIT_AND_V2_PROTOCOL.md` and
`evidence/real_stagea_devg_v2/AUDIT.md`.

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

Use the existing fetched cache. Do not recapture and do not touch test worlds.

```bash
frank-eq diagnose-real-cache \
  --cache .agents/state/lumi/frank-eq-stagea-devg-v2/remote/runs/cache \
  --out runs/diagnostics/frank-eq-stagea-devg-v2
```

The diagnostic is now also available as workflow stage `diagnose`:

```bash
frank-eq run-real-stagea \
  --config configs/stage0/real_lumi.yaml \
  --out <run-root> \
  --stages cache,validate,diagnose
```

Prefer the standalone command for the existing cache.

The diagnostic uses training/validation worlds only and measures per model:

- fact readability;
- oracle-signature readability;
- own-future-signature readability;
- residual readability;
- individual layer and concatenated capture;
- renderer-transfer fact readability;
- native branch competence against operation-wise priors.

Its machine recommendation is non-promotional and authorizes no run.

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

Do not tune the v1 test gate, rerun the same deterministic test, add a target
hidden-state decoder, or start receiver execution. First run the existing-cache
localization and freeze exactly one versioned v2 hypothesis.
