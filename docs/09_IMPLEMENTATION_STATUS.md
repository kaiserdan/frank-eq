# Implementation status

Snapshot: 2026-08-10

## Completed

| Component | Status |
|---|---|
| Configuration schema and validation | complete |
| Deterministic synthetic generator | complete |
| World-grouped and family-stratified splits | complete |
| Model-local private charts | complete |
| Gauge-fixed facts + residual public code | complete |
| Frozen public operation decoder | complete |
| Renderer/cross-model invariance objectives | complete |
| World contrastive objective | complete |
| Model-ID adversary | complete |
| Held-sender frozen-decoder onboarding | complete |
| Facts-only residual baseline | complete |
| Raw-hidden Ridge geometry baseline | complete |
| Query-conditioned typed packet | complete |
| Quantization and checksum round-trip | complete |
| Grouped bootstrap and machine reducer | complete |
| Real-cache information contracts | complete |
| Unit and end-to-end tests | complete |
| CI and repository validator | complete |

## Synthetic reference

The full frozen config passes every implementation gate. The evidence copy under `evidence/reference_stage0/` is intentionally small and contains no checkpoint or prediction rows.

This result authorizes only the real-model representation canary.

## Not implemented

- Hugging Face/vLLM hidden-state capture backend;
- real future-operation branching harness;
- real-model source chart training;
- receiver-native packet execution;
- rate-matched transcript and summary controls;
- W&B telemetry;
- Olivia multi-node real-model jobs;
- confirmation or locked data.

## Known implementation caveats

- The optional workspace gate remains effectively open in the reference run; no sparse-workspace claim is made.
- The synthetic decoder exactly matches the generator's public operation algebra. Real tasks require a separately frozen public decoder.
- The synthetic hidden charts are generated from the public state by construction. Their purpose is contract validation, not realism.
