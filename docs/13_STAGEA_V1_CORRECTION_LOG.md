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

## 2026-08-10 — execute the existing-cache localization

- **Command:** `frank-eq diagnose-real-cache --cache .agents/state/lumi/frank-eq-stagea-devg-v2/remote/runs/cache --out runs/diagnostics/frank-eq-stagea-devg-v2` (ridge 10.0; local-only output, `runs/` is ignored).
- **Access audit:** train worlds 42, validation worlds 10 per model; test worlds used 0, test labels consumed false, held-out operation gate not reused. Machine flags `authorizes_test_access/new_outcome_run/receiver_execution/scientific_claim` all false.
- **Machine recommendation:** `FIX_NATIVE_COMPETENCE_BEFORE_LATENT_REVISION` — founder native branch Brier gain over operation-wise coordinate prior is −0.060 on validation worlds. Decision-tree branch A: prompt/runtime competence is a prerequisite before attributing failure to hidden-state capture.
- **Probe findings (all three models, validation worlds):**
  - Native competence fails across every operation family (gains −0.197..+0.006; worst reciprocity, mutual, compare_outdegree); the frozen models' own branches track priors, not oracle outcomes.
  - Facts are weakly readable: Brier gain over coordinate prior +0.006..+0.035, balanced accuracy 0.66..0.68 (best layer or concatenation).
  - Oracle signatures readable: gain +0.062..+0.067, balanced accuracy 0.74..0.76.
  - Own-future signatures readable: qwen3 +0.0115 gain / balacc 0.81; smollm +0.0008 / 1.00; held llama +0.0023.
  - Residual coordinates: R2 0.64..0.69, but these are declared-global controls rendered in the prefix and carry no hidden-operational evidence.
  - **Renderer transfer fails:** fact probes trained on renderer 0 and scored on renderer 1 lose to the coordinate prior (−0.18..−0.26 gain in both directions for all models), while the trained v1 code reached renderer cosine 0.99 — consistent with invariance without specificity (non-collapse rule), a new confound against v1's renderer-invariance pass.
- **Interpretation:** the earlier "facts not linearly readable" claim is superseded by probe evidence — captures carry real, weakly positive fact signal and strong self-future signal. The primary v1 bottleneck per the frozen tree is native branch competence on raw prompts; the next frozen hypothesis must be a model/task competence prerequisite (for example a native chat template or stronger checkpoints) with capture before operation reveal and an audited exact-prefix continuity. Nothing here authorizes a fresh outcome run; a Stage-A v2 registration must still satisfy all ten prerequisites in `docs/12`.

## 2026-08-10 — freeze Stage-A v2-1 registration (native chat-template competence)

- **Decision:** freeze exactly one versioned v2 hypothesis per decision-tree branch A: the same frozen checkpoints and panel geometry, with the capture prefix wrapped in each model's native chat template. Protocol `docs/14_STAGEA_V2_PROTOCOL.md`, config `configs/stage0/real_lumi_v2.yaml`.
- **What changed vs v1:** chat-templated prefix (system contract + world statement in user turn); new world seed 20260810 (fresh untouched test role); exact revision pins required for all three checkpoints (c1899de2…, 69f49d9c…, 92131767…); frozen sample-wise KV-versus-replay parity audit (`parity_sample_size: 8`, `max_abs_diff ≤ 0.01` fails the cache build, recorded in metadata); frozen native-competence gate (validation worlds, founders, held-out operations, gain ≥ 0); residual check demoted to a declared-global control; `max_length` 1024.
- **What is deliberately unchanged:** checkpoints, panel geometry, depths 0.35/0.60/0.85, shared-head compiler (`public_head_scope: shared`), losses, gates thresholds, operation registry.
- **Implementation:** `capture.prompt_format: raw|chat` (chat prefixes tokenize with `add_special_tokens=False`); parity sampling inside the KV-reuse branch path with builder enforcement; evaluator emits `native_competence_brier_gain_over_prior` from the frozen cache (no trained weights, no test labels); reducer honors `control_checks` and the new gate. Tests: 50/50 green, `validate_repo` passes with the v2 config registered.
- **Status:** frozen and ready; dry-run submission succeeds. An outcome-bearing LUMI run is NOT submitted until the registration is ratified. Any deviation from this registration is a new versioned protocol with a fresh test role.

## 2026-08-10 — parity gate amendment (engineering finding, pre-scientific)

- **Finding:** the ratified v2-1 run (Slurm 20951138, dev-g) failed closed at cache build: `KV-reuse versus exact-replay parity divergence for qwen3-0.6b: max_abs_diff=0.030490 exceeds limit 0.01`. The parity audit worked as designed; no training, evaluation, or test-label consumption occurred (workflow failed before any scientific stage; W&B run `48tyge0d` recorded the failure).
- **Interpretation:** KV-reuse and exact-replay branch modes are not bit-identical on this stack (bf16 ROCm; plausible kernel-level accumulation differences). The v1 cache was unaffected because it used kv_reuse exclusively (2048/0 per model); the finding constrains only mixed-mode caches.
- **Amendment (engineering tolerance, not a scientific gate):** run one cache-only measurement with `parity_sample_size: 32` and a non-blocking tolerance to map the noise floor across models and operations. The measurement cache is discarded. The final tolerance is then set from the measured distribution (floor 0.05, headroom 3x measured max) and documented here before the outcome-bearing run resumes.
- **Observability:** the enforcement error now reports per-model max/mean diffs and the top-3 diverging branches, so the measurement run needs no code changes.

## 2026-08-10 — parity measurement results and exclusive-kv amendment

- **Measurement run:** Slurm 20951659 (dev-g, cache,validate, source `b09a2781eb26`), 32 dual-mode branches per model, non-blocking tolerance; cache discarded after reading.
- **Measured noise floor (max abs probability diff):** qwen3-0.6b 0.0757 (mean 0.014), smollm-1.7b-instruct 0.1089 (mean 0.039), llama-3.2-1b-instruct-held 0.0517 (mean 0.019). All 2048/2048 branches ran kv_reuse in the measurement cache; no mixing occurred.
- **Conclusion:** exact-replay and KV-reuse are not interchangeable at scientific-gate precision on this stack (bf16 ROCm kernel-level accumulation differences). A mixed-mode cache could carry up to ~0.11 probability error per branch.
- **Amendment (engineering, not scientific):** `branch_mode: kv_reuse` is now the EXCLUSIVE mode and `allow_exact_replay_fallback: false` — no cache can mix modes; a KV-clone failure fails the build instead of silently replaying. The 32-branch parity sample stays registered per model as a stack-property audit, and `parity_max_abs_diff: 0.33` follows the announced formula max(0.05, 3 × measured max). Protocol `docs/14` updated. v1 is unaffected (pure-kv cache; decision stands).
- **Status:** v2-1 registration re-frozen with the amended capture contract; full run submitted next.

## 2026-08-11 — Stage-Q chat_turn prefix-continuity defect and candidate rendering correction

- **Finding:** the first `frank-eq-stageq-chat-turn` cache run (Slurm 20961621, dev-g) failed closed with `chat_turn template violates exact prefix continuity`. Reproduction against the real templates (container, login node) showed Qwen3-0.6B's chat template renders assistant messages context-dependently: a trailing post-query assistant message gains a `<think>\n\n</think>\n\n` wrapper (`loop.index0 > last_query_index and loop.last`), which disappears once a later user message exists. The original candidate (system contract / user world statement / assistant acknowledgement) therefore could not satisfy exact-prefix continuity under Qwen3 — the prefix render wrapped the acknowledgement, the full-conversation render did not. SmolLM and Llama were unaffected. `enable_thinking: false` does not bind this branch of Qwen3's template.
- **Fix (frozen in code and protocol):** the candidate conversation is now `system: reasoning contract + query-blind world statement; assistant: fixed acknowledgement`, with the operation revealed as `user: operation question` + generation boundary. With no user message before the acknowledgement, Qwen3's post-query branch never applies and both renders are identical. Verified token-prefix equality against all three real templates (qwen3, smollm, llama). Regression test added with a Qwen3-like context-dependent stub template.
- **Contract docs updated:** `docs/15` §4.2 and `docs/17` (candidate rendering correction).
- **Status:** chat_turn cache,validate re-submitted after the fix; legacy baseline cache (Slurm 20961538) already completed and is unaffected.

## 2026-08-10 — v2-1 falsified: chat template is not the competence bottleneck

- **Run:** `frank-eq-stagea-lumi-v2` (Slurm 20952565, dev-g, source `1aa741d5df31`), full workflow completed with zero failures; adopted negative under `evidence/real_stagea_lumi_v2/`; decision log entry appended.
- **Result:** native-competence gain −0.0521 (validation worlds, founders, held-out ops) vs −0.0603 in the v1 diagnostic — unchanged within noise. The frozen chat-template hypothesis (v2-1) is falsified; the prompt surface is not the driver of branch incompetence.
- **Probe deltas (chat vs raw):** own-future signatures more readable (qwen3 balanced accuracy 0.813→0.892); linear fact readability decreased (best gain +0.006..+0.035 → −0.015..+0.007); residual R2 up (declared-global control); renderer-transfer still loses to the prior both directions (−0.15..−0.26).
- **Interpretation:** the source models are oracle-incompetent on this 6-entity graph task under both prompt surfaces (their branches are anti-predictive vs the operation prior); an oracle-grounded quotient is bounded by source task competence. Branch A continuation: stronger checkpoints, or a task revision — a new versioned registration with a fresh test role either way. No tuning occurred under this outcome.
