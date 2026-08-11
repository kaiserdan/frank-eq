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

## 2026-08-10 — Stage-A v2-1 (chat-template competence) is falsified

- **Run:** `frank-eq-stagea-lumi-v2` on LUMI `dev-g` (Slurm 20952565, source archive `1aa741d5df31`, frozen `configs/stage0/real_lumi_v2.yaml`), full `cache,validate,train,eval`, engineering verification passed, 276.3 s wall. Adopted as a valid terminal negative for v2-1; evidence under `evidence/real_stagea_lumi_v2/`.
- **Decision:** `STOP_OR_REVISE_STAGE0`, `authorizes_scientific_claim=false`. The native chat-template hypothesis is falsified: native-competence gain is −0.0521 (validation worlds, founders, held-out ops; gate ≥ 0.0) versus −0.0603 in the v1 diagnostic — the prompt surface is not the competence bottleneck. Quotient metrics worsened (held-out Brier upper 0.242 vs 0.198; held-sender retention −0.34 vs 0.47).
- **Localization (train/validation only):** under chat capture, own-future signatures are more readable (qwen3 balanced accuracy 0.89) but linear fact readability dropped (best gain −0.015..+0.007 vs +0.006..+0.035) and renderer-transfer of facts still loses to the prior in both directions. The source models' branches remain oracle-incompetent on this task; an oracle-grounded quotient is bounded by source task competence.
- **Consequence:** the next hypothesis must raise model task competence (branch A continuation: stronger checkpoints) or revise the task; per the stop rules it requires a new versioned registration with a fresh untouched test role. No gates, layers, panel, prompts, or checkpoints were tuned under this outcome.

## 2026-08-10 — localize the Stage-A failure to capture sufficiency

- **Evidence:** training curves show the supervised fact head is unfittable — facts BCE moves 0.702→0.660 over 49 epochs (chance ≈ 0.693) with validation loss *increasing* (0.696→0.710), while the residual coordinates drop 1.09→0.35 and the signature loss improves. Seen-operation Brier (0.170) equals held-out (0.173), so operation generalization and the frozen decoder are not the limit; renderer code cosine is 0.99, so nuisance invariance is not the limit.
- **Interpretation:** the 30 per-edge facts of the 6-entity graphs are not linearly readable from residual-stream states at normalized depths 0.35/0.60/0.85 of Qwen3-0.6B, SmolLM-1.7B-Instruct, and Llama-3.2-1B-Instruct under raw (non-chat) prompts, while the declared global coordinates (density, reciprocity) are readable. Per-family Brier confirms the pattern: density 0.060 and reciprocity 0.109 are near-chance-optimal, while edge-derived operations (compare_outdegree 0.291, inverse 0.225) fail hardest. The held sender's chart also lands at roughly half the founder code scale, consistent with its retention shortfall.
- **Consequence:** a revised Stage A must change the capture contract (for example chat-templated prefixes, additional or different depths, larger chart capacity, or a stronger fact objective), and per the stop rules it requires a new versioned protocol with a fresh untouched test role. No gates, layers, or panel are tuned under this outcome.

## 2026-08-11 — recover and adopt Stage R / RC0

- **Original capture:** `frank-eq-rc0-rate-compute-olivia-20260811c` (Olivia Slurm `1874736`, source `c24ae1eb...de48`) completed both pinned models and all 89,856 raw and calibrated response rows before failing in its first grouped metric with `too many values to unpack (expected 2)`. It produced no compiled predictions, metrics, decision, summary, or artifact manifest.
- **Repair:** restore the historical `aggregate_by_world -> (world_ids, world_means)` API and row-count guard, and add a regression with more than two worlds. The estimands, calibration, bootstrap seeds, gates, decision reducer, frozen config, and captured responses did not change.
- **Recovery:** `frank-eq-rc0-rate-compute-olivia-20260811d-recovery` (Slurm `1891471`, repaired source `84ea4112...c0b7`) reused the capture only through a separate SHA-256-bound input manifest. It copied rather than modified the original artifacts, executed no model inference, completed in 45 seconds, and passed repository verification, the RC0-specific verifier, fatal-log scanning, and an independent metric/decision recomputation.
- **Decision:** adopt `PUBLIC_BASIS_COMPOSITION_SUPPORTED` and `OPERATIONAL_BASIS_CANDIDATE_FOR_STAGEA_REGISTRATION` as a development result. Across 3,712 hard-family validation predictions, compiled Brier is `0.0408`, versus `0.2035` for the training-selected direct protocol and `0.2181` for the prior; lower-95 gains are `0.1542` and `0.1661`. Every frozen model/complexity and hard-family composition stratum passes, with zero hard-oracle executor mismatches.
- **Preserved diagnostic:** semantic sequence likelihood improves over the answer-token channel, but generated reasoning is worse than the matched 32-token pause condition (reasoning-minus-pause interval `[-0.00540, -0.00029]`). The positive endpoint is typed basis recovery plus deterministic composition, not contentful reasoning.
- **Interpretation:** RC0 is interactive source tomography requiring 12 or 30 basis queries, not a one-shot hidden-state compiler, cross-model communication result, or rate advantage over a single direct query. The compact adopted package is `evidence/real_stage_r_olivia_rc0/`.
- **Authority:** draft exactly one fresh Stage-A v3 registration with new worlds, a new unopened held sender, complete model-local token/slot compilers, separated behavioral and oracle-semantic channels, and strong baselines. No RC0 rerun, v3 launch, claim-bearing test access, receiver execution, or scientific claim is authorized.
- **Operational correction:** Olivia `accel` nodes are ARM64 NVIDIA Grace Hopper systems. The earlier 2026-08-10 AMD64/H200 container entry is superseded; RC0 used the pinned native ARM64 image with SHA-256 `a3ca46f0...aa3b1` and no runtime package installation.

## 2026-08-12 — freeze Stage-A v3-1 and authorize sequential execution

- **Human authority:** the user explicitly authorized all sequential next steps, requested results be documented along the way, and requested regular commits. This removes the prior operational pause after protocol drafting but does not waive scientific gates or causal/access boundaries.
- **Registration:** freeze `docs/20_STAGEA_V3_PROTOCOL.md` and `configs/stagea_v3/real_olivia_v3.yaml` before implementation. One representation run is permitted only after separate registration and implementation commits, full local validation, and an inspected content-addressed dry run.
- **Primary method:** one prefix forward; all-token residual capture at normalized depths 0.25/0.50/0.75/1.00; independent model-local semantic and behavioral coordinate-query resamplers; typed directed-edge packet; frozen deterministic executor; zero post-capture source queries in the primary condition.
- **Roles:** Qwen3-4B and Qwen3-8B remain development-exposed founders. Qwen3-14B at revision `40c069824f4251a91eefaf281ebe4c544efd3e18` is the new task-unopened held sender and may see train worlds only after founder freezing. Same-family establishment is an explicit limitation.
- **Access:** fresh independent train/validation/test seeds `2026081201/02/03`; test panels are created only after founder and held freeze manifests; one test access; worlds remain the independent unit.
- **Baselines:** train prior, parameter-matched token IDs, final-token public MLP, historical continuous quotient, train-selected direct protocols, interactive basis, deterministic text parser, rate-matched canonical text, oracle basis, and shuffled/wrong/zero packets are mandatory.
- **Interpretation boundary:** because a deterministic parser can recover the explicit controlled graph, the text and oracle controls are ceilings. A pass can establish one-shot public compilation and held onboarding, not hidden-over-text information or receiver utility.
- **Next gate:** only `STAGEA_V3_REPRESENTATION_QUALIFIED` may authorize drafting a receiver protocol. Receiver execution, receiver-world access, and scientific claims remain locked.
- **Pre-implementation completeness amendment:** before any v3 panel or model outcome existed, bind operation sampling to independent seed `2026081213` plus entity count so train, validation, and test share the same frozen operation registry. Role seeds vary worlds only; this restores the protocol's stated operation-holdout contract and does not change the hypothesis or gate.

## 2026-08-12 — implement the Stage-A v3 causal core

- **Panels:** implement independent role-specific world panels while deriving all roles' operation instances from the frozen complexity-specific operation seed. Public world IDs cannot collide across roles or complexities.
- **Compiler:** implement one model-local all-token token/slot resampler per channel. Four-entity rows activate the canonical induced 12-coordinate subset of the fixed 30-coordinate six-entity registry; the compiler API contains no operation input.
- **Channel separation:** semantic and behavioral modules are constructed independently, expose disjoint parameter identities, and retain separate forward namespaces.
- **Access:** add an OS-locked, hash-bound stage ledger. Test-panel creation is rejected until both founder and held freeze manifests validate, consumes exactly one access, and records every sanctioned test-file open.
- **Verification:** focused tests pass for panel determinism/freshness, unseen-renderer completeness, coordinate selection, padding isolation, channel disjointness, early/repeated access rejection, and artifact-hash drift. This is an implementation checkpoint, not permission to stage the held model or inspect test data.

## 2026-08-12 — implement Stage-A v3 capture and basis fitting

- **Capture:** store every unpadded prefix-token residual at all four registered depths in float32, with exact formatted prefix bytes, tokenizer offsets, token IDs, selected layers, residual hashes, and observed model revision. Prefix state is materialized before any basis or target query.
- **Teachers:** derive behavioral edge targets and all sequence/reason/pause direct responses only through query-exclusive cloned KV branches. Logical queries, batches, generated tokens, prefix-continuity checks, and the primary compiler's zero-query contract are separate counters.
- **Fitting:** train one source-local module per channel and seed with world-balanced batches that retain both renderer views. Semantic BCE and behavioral soft-label Brier remain separate; validation Brier plus the registered renderer variance selects checkpoints.
- **Controls:** use a deterministic public token-ID feature map with the exact token/slot architecture and parameter count. Solve the final-token MLP width against the primary parameter count and fail outside the 5% tolerance. Add an exact parser for all three renderer grammars.
- **Integrity:** detect both on-disk and in-memory config mutation, atomically write capture/checkpoint files, bind capture hashes into checkpoints, and test a full miniature fit/save/reload/predict cycle. The unopened held model remains unstaged; this checkpoint does not authorize test creation.

## 2026-08-12 — implement Stage-A v3 controls, packets, and reducer

- **Train-only controls:** reuse the RC0 affine-log-odds Platt method (`l2=1e-3`, 100 Newton steps, unconstrained slope) for coordinate-specific interactive-basis calibration and family/protocol-specific direct calibration. Select sequence/reason/pause only by calibrated training Brier with frozen-order tie breaking.
- **Continuous control:** reproduce the historical final-token private 32-dimensional quotient with a 160-wide chart and learned 96-wide operation head, trained and ensembled under the same registered roles, seeds, and early-stopping rule.
- **Wire contract:** serialize clipped-logit typed edge packets with exact bit packing, SHA-256 checksum, and distinct payload/framing counts. The four-bit primary carries 48 or 120 payload bits. The canonical-text ceiling is parsed first, then compressed into the same typed wire budget; it remains an oracle-like text control.
- **Prediction registry:** require every registered prior, token, final-token, continuous, direct, interactive, text, oracle, shuffled, wrong-world, zero, and 1/2/4/8-bit condition before reduction. Prediction arrays and metadata round-trip under hashes.
- **Reducer:** compute 2,000-replicate world-grouped intervals, model/complexity/renderer/family/seed strata, retrieval, wrong-world margin, train-to-test sender identity, held retention, quantization retention, oracle mismatches, and amortized rate/compute. Emit only the seven frozen diagnoses; even a representation pass opens protocol drafting only and leaves receiver execution and every claim field false.

## 2026-08-12 — void v3-1 test role and freeze the v3-2 access repair

- **Detected boundary violation:** implementation tests called the deterministic panel generator with the registered v3-1 test seed `2026081203` before founder and held freezes. No checkpoint was loaded, no model saw a task row, and no outcome was computed, but the worlds and labels were no longer unopened.
- **Disposition:** void v3-1 before execution. It cannot be launched or interpreted. Preserve this exposure in the append-only log rather than relabeling the seed as fresh.
- **V3-2 repair:** change only the test-world seed to previously uninstantiated `2026081297`; keep train/validation worlds, operation registry, models, architecture, losses, baselines, gates, and bootstrap contract fixed.
- **Enforcement:** the panel generator now rejects the registered test role unless it receives a config-matching access ledger already advanced to `evaluate` with its one access consumed. Tests use train/validation panels or synthetic role wrappers and assert that an ungranted test request fails before generation.
- **Authority:** this is a pre-outcome causal-access repair. The unopened held model remains unstaged and no Stage-A v3 model result exists.
