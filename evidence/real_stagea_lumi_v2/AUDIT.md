# Stage-A v2-1 adopted negative — independent audit

Run: `frank-eq-stagea-lumi-v2` (LUMI dev-g, Slurm 20952565, source archive
`1aa741d5df31`, config `configs/stage0/real_lumi_v2.yaml`, protocol
`docs/14_STAGEA_V2_PROTOCOL.md`).

## Outcome

- Workflow: `completed` — `cache,validate,train,eval`, zero failures,
  276.3 s wall.
- Engineering: verification passed; cache validation passed; all branches
  ran `kv_reuse` (2048 per model); the registered 32-branch parity audit
  recorded the measured stack noise floor (max abs diff qwen3 0.0757,
  smollm 0.1089, llama-held 0.0517) within the amended tolerance 0.33;
  W&B run synced with zero telemetry failures.
- Decision: `STOP_OR_REVISE_STAGE0` (fail), `authorizes_scientific_claim`
  false. Adopted as a valid terminal negative for the v2-1 hypothesis.

## Hypothesis result

v2-1 (native chat-template competence) is falsified:

| Metric | v1 (raw) | v2 (chat) | Reading |
|---|---|---|---|
| native competence gain | −0.060 (diag) | −0.0521 | unchanged; prompt surface is not the bottleneck |
| fact accuracy | 0.5296 | 0.5509 | marginal |
| held-out signature Brier | 0.1729 | 0.2421 (upper 0.242) | worse |
| held-sender retention | 0.4717 | −0.3445 | much worse |
| model-ID leakage | 0.6528 | 0.6389 | unchanged |
| renderer invariance | 0.9925 | 0.9793 | unchanged (still collapse-ambiguous) |

Diagnostic probes (train/validation only) on the v2 cache: facts less
linearly readable than v1 (best gain −0.015..+0.007 vs +0.006..+0.035;
balanced accuracy 0.59–0.63 vs 0.66–0.68); own-future signatures more
readable (qwen3 balacc 0.89); residual R2 0.80–0.87 (declared-global
control); renderer-transfer of facts still loses to the prior in both
directions (−0.15..−0.26).

## Integrity notes

- Fresh world seed 20260810; v1 test role untouched.
- Exact revision pins honored (c1899de2…, 69f49d9c…, 92131767…).
- Two engineering amendments preceded this run and are recorded in
  `docs/13_STAGEA_V1_CORRECTION_LOG.md` (parity tolerance calibration and
  exclusive-kv freeze). They change no scientific gate.
- The historical v1 negative remains the adopted reference; this package
  does not alter `evidence/real_stagea_devg_v2/`.

## Authorization boundary

- `authorizes_scientific_claim`: false.
- No gate, layer, panel, prompt, or checkpoint selection may be informed by
  this outcome without a new versioned registration and fresh test role.
- Receiver-native execution remains locked.
