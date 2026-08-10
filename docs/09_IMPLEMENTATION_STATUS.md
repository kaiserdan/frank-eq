# Implementation status

Snapshot: 2026-08-10

## Completed

| Component | Status |
|---|---|
| Synthetic configuration and controlled generator | complete |
| World-grouped and family-stratified splits | complete |
| Model-local private charts | complete |
| Gauge-fixed facts + residual public code | complete |
| Synthetic parameter-free coefficient decoder | complete |
| Founder training and held-sender onboarding | complete |
| Facts-only, raw-hidden, leakage, retrieval, quantization, and bootstrap evaluation | complete |
| Query-conditioned typed packet | complete |
| Synthetic reference evidence and machine gate | complete |
| Real checkpoint configuration and model roster contract | complete |
| Frozen real relational-world panel | complete |
| Two query-blind renderer families | complete |
| Parameter-free differentiable graph interrogator | complete |
| Hugging Face hidden-state capture at normalized depths | complete |
| Physical KV-reuse branch path | complete |
| Exact-prefix replay fallback with explicit accounting | complete |
| Canonical single-token outcome mapping per tokenizer | complete |
| Real `FutureSignatureRecord` serialization | complete |
| Formal oracle / source branch separation | complete |
| Real bundle serialization and cache validation | complete |
| Real cache/train/eval workflow manifest and status | complete |
| Olivia content-addressed submit/status/fetch/verify | complete |
| LUMI content-addressed submit/status/fetch/verify | complete |
| Cluster Slurm entrypoints and agent skills | complete |
| Fail-open W&B telemetry for real Stage-A runs | complete |
| amd64 Olivia runtime container (H200 nodes) | complete |
| Focused real-panel, graph-decoder, config, HF utility, cluster, and telemetry tests | complete |

## Synthetic reference

`configs/stage0/synthetic_full.yaml` passes every synthetic implementation gate. The adopted evidence under `evidence/reference_stage0/` authorizes only the real-model representation canary.

## Real Stage-A first outcome (negative)

The first full frozen canary ran on LUMI `dev-g` (`frank-eq-stagea-devg-v2`, source `bc2bff426e1c`) and returned a valid negative decision: `STOP_OR_REVISE_STAGE0`. Engineering integrity passed end to end; the scientific gate failed on fact accuracy, retrieval, wrong-world specificity, held-sender retention, held-out Brier, and model-ID leakage. Renderer invariance, quantization retention, and the operational residual gain passed. Failure is localized to capture sufficiency (facts not linearly readable from the captured states; global coordinates are). Evidence: `evidence/real_stagea_devg_v2/`.

The frozen cluster configs remain:

```text
founders: Qwen3-0.6B, SmolLM-1.7B-Instruct
held sender: Llama-3.2-1B-Instruct
worlds: 64
renderers: 2
operations: 16 across 8 families
capture depths: 0.35, 0.60, 0.85
public executor: parameter-free graph interrogator
```

A revised Stage A requires a new versioned protocol and a fresh untouched test role; no gates, layers, or panel were tuned under the negative outcome.

## Not implemented or not authorized

- adopted real-model Stage-A evidence (the adopted outcome is negative; no positive real evidence exists);
- receiver-native packet execution;
- rate-matched transcript and summary controls;
- a second task family;
- confirmation or locked roles;
- safety or harm-tail evaluation;
- multi-node or multi-GPU data parallelism.

## Known implementation caveats

- KV-cache object semantics vary across Transformers releases. `branch_mode=auto` attempts cloned KV reuse and records exact-prefix replay fallback counts. A literal cached-state claim requires an all-KV rerun or a predeclared tolerated fallback fraction.
- Raw prompts are used instead of model-specific chat templates to preserve an exact shared prefix/query token boundary. Source branch accuracy is diagnostic and may understate instruction-tuned capability.
- Density and reciprocity residuals are externally declared graph coordinates. They are not yet evidence that a natural LLM contains an irreducible operational residual beyond grounded facts.
- The real graph interrogator is task-specific. Passing it is evidence for an operational quotient on this family, not a universal latent language.
- The cluster workflow is sequential by model to limit VRAM and preserve simple source isolation; multi-node scaling is unnecessary until the canary passes.
- The amd64 Olivia container lacks transformers and wandb; the job installs them from PyPI (or a pre-downloaded wheel directory via `FRANK_EQ_PIP_FIND_LINKS`) when `FRANK_EQ_ALLOW_PIP_INSTALL=1`.
- W&B telemetry is fail-open and does not gate anything; a missing key or network simply disables the stream and is noted in `run_summary.json`.
