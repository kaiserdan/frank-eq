# Stage-A v1 correction log

This file continues the append-only scientific decision record for corrections
made after the first real Stage-A v1 outcome. It does not rewrite
`docs/10_DECISION_LOG.md`.

## 2026-08-10 — correct the v1 failure localization

- **Preserved decision:** LUMI job `20942127` remains an engineering-valid
  `STOP_OR_REVISE_STAGE0` result for the exact v1 pipeline.
- **Correction:** the statement that failure was localized to capture
  sufficiency, or that facts were not linearly readable, is withdrawn as an
  identified conclusion.
- **Reason:** the executed fact head was nonlinear and jointly trained with
  signature, residual, invariance, contrastive, adversarial, variance,
  workspace, and quantization objectives. No independent model/layer
  readability upper bound was run.
- **Additional confounds:** v1 captured only final-token residuals while future
  branches used the full KV cache; public fact/residual heads were shared
  across models; raw prompts yielded source-branch accuracy below an
  operation-prior baseline; density/reciprocity residual targets were printed
  explicitly in the prefix.
- **Resulting interpretation:** the exact v1 pipeline is falsified. The broader
  failure source remains unresolved.

## 2026-08-10 — freeze the existing-cache localization

- **Decision:** add a non-promotional diagnostic using only v1 training and
  validation worlds.
- **Measures:** model/layer fact readability, oracle-signature readability,
  own-future-signature readability, residual readability, renderer transfer,
  and native branch competence.
- **Access:** zero test labels; no gate change; no architecture promotion.
- **Implementation:** `frank-eq diagnose-real-cache` and optional workflow stage
  `diagnose`.
- **Authority:** the diagnostic may motivate one versioned Stage-A v2
  hypothesis. It cannot authorize a fresh outcome run or receiver execution.

## 2026-08-10 — implement complete model-local compiler support

- **Decision:** add `model.public_head_scope: local` while preserving
  `shared` as the historical default.
- **Reason:** v1 shared fact/residual heads force a held sender's chart into a
  founder-induced private bottleneck gauge. Only public coordinate semantics
  and the interrogator need to be shared.
- **Status:** implementation-only, dormant. No v2 config is frozen and no run is
  authorized.

## 2026-08-10 — correct real decision metadata and evidence hygiene

- **Decision:** future real evaluations emit a real-specific decision schema and
  authorization boundary. The historical v1 decision file remains immutable,
  including its erroneous synthetic scope string.
- **Evidence:** add a hash manifest and independent audit under
  `evidence/real_stagea_devg_v2/`.
- **Repository hygiene:** `.agents/state/` is local operational state and is
  removed from version control and ignored.
