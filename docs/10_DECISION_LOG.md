# Append-only decision log

## 2026-08-10 — initialize Frank-EQ

- **Decision:** replace direct cross-model hidden-state reconstruction with future-defined operational equivalence quotients.
- **Scientific reason:** historical work repeatedly found shared descriptive structure without legal realization, continuation, semantic specificity, or sender-independent establishment.
- **Primary object:** a state cached before operation reveal and identified by its distribution of outcomes under a frozen family of future operations.
- **Architecture:** model-local private chart; gauge-fixed grounded facts plus operational residual; frozen public decoder; query-conditioned typed packet; receiver-native execution only after representation qualification.
- **Excluded:** pair-specific primary translators, target-hidden reconstruction, joint sender-receiver training, operation-conditioned state formation.
- **Initial gate:** synthetic Stage 0 under `configs/stage0/synthetic_full.yaml`.

## 2026-08-10 — adopt gauge-fixed public coordinates

- **Decision:** the chart's private bottleneck is not the cross-model code. The public code is constructed explicitly from grounded fact probabilities and bounded residual coordinates.
- **Reason:** an arbitrary shared bottleneck retained model identity and reproduced the same gauge ambiguity the project is intended to remove.
- **Consequence:** invariance, retrieval, packetization, leakage, and quantization are evaluated on the public state only.

## 2026-08-10 — adopt frozen public operation execution

- **Decision:** replace the learned operation decoder in the primary synthetic path with a parameter-free decoder over frozen public operation coefficients.
- **Reason:** held-out future operations should test state sufficiency, not whether a learned decoder inferred a new operation from a few examples.
- **Consequence:** operation instances can be held out while retaining exact public semantics. A learned operation decoder remains a potential baseline, not the primary executor.

## 2026-08-10 — synthetic Stage-0 pass

- **Decision:** the full reference run passes and authorizes implementation of a real-model representation canary.
- **Result:** `evidence/reference_stage0/decision.json` returns `PROMOTE_REAL_MODEL_CANARY` and `authorizes_scientific_claim=false`.
- **Next action:** implement real state capture and future-branch records without interpreting the synthetic result as LLM evidence.

## 2026-08-10 — freeze the first real Stage-A vertical slice

- **Decision:** use controlled closed-world relational graphs as the first real frozen-checkpoint task family.
- **Operation registry:** lookup, inverse, mutual relation, two-hop composition, out-degree comparison, counterfactual edge addition, graph-density class, and reciprocity class; two operation instances per family, with family-stratified holdout.
- **Public semantics:** exact graph facts plus two declared global coordinates, executed by a parameter-free differentiable graph interrogator. No target hidden state, target logit, learned receiver, or pair-specific decoder enters the primary path.
- **Source behavior:** record each source model's actual post-reveal A/B distribution, but keep it diagnostically separate from the formal oracle signature used to define the public quotient.
- **Causal boundary:** capture three normalized-depth hidden states from a prefix containing no operation. Attempt physical cloned-KV branches; allow exact-prefix replay fallback only with explicit counts.
- **Model roles:** two founder families and one final held-sender entry. Held onboarding freezes the public interrogator and all founder charts.
- **Gate scope:** a pass authorizes design of a receiver-native execution experiment only. It does not authorize a universal-latent, communication, safety, or natural-task claim.

## 2026-08-10 — make scientific failure scheduler-valid

- **Decision:** `run-real-stagea` exits successfully when the workflow and artifacts are complete even if `eval/decision.json` is negative.
- **Reason:** a preregistered scientific failure is a valid experiment and must not be conflated with an engineering or scheduler failure.
- **Authority:** `workflow_status.json` records execution integrity; `eval/decision.json` exclusively records scientific promotion.

## 2026-08-10 — adopt content-addressed Olivia/LUMI execution

- **Decision:** every cluster submission packages a deterministic source tarball, hashes it, records the exact config/stages/remote root, and writes local submission/status/fetch/verify state under `.agents/state/<cluster>/<job>`.
- **Reason:** mutable working-directory deployment and prose-only job identity are insufficient for a branch-heavy research program.
- **First authorized launch:** `cache,validate` using `configs/stage0/real_olivia.yaml` or `configs/stage0/real_lumi.yaml`, after checkpoint-cache preflight.

## 2026-08-10 — add fail-open W&B telemetry to real Stage-A

- **Decision:** log run identity, cache branch-mode accounting, per-epoch training losses, and evaluation metrics to a dedicated `frank-eq-stagea` W&B project. Credentials are forwarded through the environment (`WANDB_API_KEY`) and never enter source, configs, Slurm files, submission state, or logs.
- **Reason:** training and evaluation currently write only terminal artifacts; a secondary telemetry stream is needed to diagnose whether the quotient is learning, which phase or metric diverges, and how branch modes behaved, without waiting for a fetched run root.
- **Consequence:** a new `logging.wandb` config section on the real config; `WandbTelemetry` is fail-open (missing package, credentials, or network degrades to a stderr note and counter, never a workflow failure); the telemetry status is recorded in `run_summary.json`. W&B remains secondary telemetry: promotion authority stays exclusively with `eval/decision.json`.

## 2026-08-10 — build an amd64 runtime container for Olivia

- **Decision:** replace the arm64-only scratch PyTorch images with `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (amd64), since every `accel` node on Olivia is H200/x86_64 and the existing images cannot execute there. Runtime pip-installs `[real]` extras (transformers, wandb) inside the job.
- **Reason:** the previous images were built on an arm64 host and are unusable on Olivia compute nodes; the default image path in `olivia/run.slurm` was therefore broken.
- **Consequence:** `FRANK_EQ_OLIVIA_IMAGE` default now points at the amd64 image; `FRANK_EQ_ALLOW_PIP_INSTALL=1` and optional `FRANK_EQ_PIP_FIND_LINKS` (offline wheel directory) are forwarded through the submitter for runtime dependency installation.

## 2026-08-10 — switch the first canary launch to LUMI

- **Decision:** launch the first `cache,validate` canary on LUMI (`small-g`) instead of Olivia. The Olivia `accel` queue had 761 pending jobs with no idle GPU nodes; LUMI scheduled the same job with an ~8-hour estimate. The scientific configuration is identical (`configs/stage0/real_lumi.yaml`); only run identity and the operational partition differ.
- **Checkpoint staging:** `Qwen/Qwen3-0.6B` was already cached on LUMI. `HuggingFaceTB/SmolLM-1.7B-Instruct` was downloaded at the exact Olivia snapshot revision, and `meta-llama/Llama-3.2-1B-Instruct` was transferred from the Olivia cache (no token available on either cluster for gated re-download). All three load offline in the runtime container.
- **Operational notes:** huggingface_hub 1.x resolves only `$HF_HOME/hub/models--*`; `snapshot_download` with a SHA revision does not write `refs/main`, and a trailing newline in `refs/main` breaks offline resolution. The LUMI cache entries were repaired accordingly.

## 2026-08-10 — fix stage-list truncation in cluster submissions

- **Bug:** `sbatch --export` splits `KEY=VALUE` pairs on commas, so `FRANK_EQ_STAGES=cache,validate,train,eval` reached the job as `FRANK_EQ_STAGES=cache`; the first dev-g run executed only the cache stage and the workflow reported completed without eval artifacts. This would have affected any multi-stage submission on either cluster.
- **Fix:** the submitter encodes the stage list with `+` separators (`FRANK_EQ_STAGES=cache+validate+train+eval`) and both quickstart scripts decode `+` back to commas before invoking `run-real-stagea`. The submission plan and local state keep the human-readable comma form.
- **Consequence:** a new dev-g test job re-runs the full `cache,validate,train,eval` workflow before the standard-g campaign.

## 2026-08-10 — first real Stage-A outcome is a valid negative

- **Run:** `frank-eq-stagea-devg-v2` on LUMI `dev-g` (Slurm 20942127, source `bc2bff426e1c`), full frozen `configs/stage0/real_lumi.yaml`, stages `cache,validate,train,eval`. Engineering integrity verified locally (`verify: passed`; causal boundary, hashes, split coverage all valid; 2048/2048 KV-reuse branches per model with zero replay fallback; W&B telemetry synced with 0 failures).
- **Decision:** `eval/decision.json` returns `STOP_OR_REVISE_STAGE0`, `status=fail`, `authorizes_scientific_claim=false`. Adopted as the first real-model Stage-A outcome. No standard-g rerun: the frozen config is seeded and deterministic, so a rerun would reproduce the same decision without new evidence.
- **Passing gates:** renderer invariance 0.992, operational residual gain lower-95 0.117, quantization retention 0.996.
- **Failing gates:** held-out signature Brier upper-95 0.198 (≤0.18), fact accuracy lower-95 0.495 (≥0.70), cross-model retrieval lower-95 0.097 (≥0.30), wrong-world margin lower-95 −0.079 (≥0.03), held-sender retention 0.472 (≥0.70), model-ID leakage 0.653 over chance (≤0.25).

## 2026-08-10 — localize the Stage-A failure to capture sufficiency

- **Evidence:** training curves show the supervised fact head is unfittable — facts BCE moves 0.702→0.660 over 49 epochs (chance ≈ 0.693) with validation loss *increasing* (0.696→0.710), while the residual coordinates drop 1.09→0.35 and the signature loss improves. Seen-operation Brier (0.170) equals held-out (0.173), so operation generalization and the frozen decoder are not the limit; renderer code cosine is 0.99, so nuisance invariance is not the limit.
- **Interpretation:** the 30 per-edge facts of the 6-entity graphs are not linearly readable from residual-stream states at normalized depths 0.35/0.60/0.85 of Qwen3-0.6B, SmolLM-1.7B-Instruct, and Llama-3.2-1B-Instruct under raw (non-chat) prompts, while the declared global coordinates (density, reciprocity) are readable. Per-family Brier confirms the pattern: density 0.060 and reciprocity 0.109 are near-chance-optimal, while edge-derived operations (compare_outdegree 0.291, inverse 0.225) fail hardest. The held sender's chart also lands at roughly half the founder code scale, consistent with its retention shortfall.
- **Consequence:** a revised Stage A must change the capture contract (for example chat-templated prefixes, additional or different depths, larger chart capacity, or a stronger fact objective), and per the stop rules it requires a new versioned protocol with a fresh untouched test role. No gates, layers, or panel are tuned under this outcome.
