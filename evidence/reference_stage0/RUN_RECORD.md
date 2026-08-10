# Synthetic Stage-0 reference run

Date: 2026-08-10

## Command

```bash
frank-eq run-stage0 \
  --config configs/stage0/synthetic_full.yaml \
  --out <run-root>
```

The adopted run used CPU execution with `training.num_threads: 1`.

## Result

```text
status: pass
decision: PROMOTE_REAL_MODEL_CANARY
authorizes_real_model_canary: true
authorizes_scientific_claim: false
```

Selected metrics are recorded in `metrics.json`; the complete gate is in `decision.json`. The evidence copy intentionally excludes checkpoints, hidden states, and row-level predictions.

## Interpretation

This run demonstrates that the repository's synthetic contracts, learning path, held-sender onboarding, uncertainty, packetization, and reducer execute coherently. It does not establish an operational quotient in any real language model.
