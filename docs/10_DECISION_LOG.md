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

## 2026-08-12 — complete the Stage-A v3-2 implementation surface

- **Workflow:** implement the exact `prepare -> founder_fit -> freeze -> held_onboard -> evaluate` sequence. The run refuses partial stages, a stale or uninspected plan, a nonempty output root, dirty source, wrong Olivia/image provenance, or any test artifact existing before the two freezes.
- **Freeze and access:** founder and held manifests bind config, protocol, implementation, panels, captures, checkpoints, calibrators, operation descriptors, and protocol selection by SHA-256. Only a validated held freeze can atomically consume the single test grant; the registered test file set and first-open hashes must be exact.
- **Accounting:** retain source query, prefix-forward, generated-reasoning, fixed-pause, packet payload/framing, compiler parameter/FLOP/wall-time, and deterministic executor operation/wall-time fields separately. Training wall time remains in checkpoint metadata.
- **Verification:** independently reload every prediction group, revalidate freeze/checkpoint/access/provenance hashes, recompute all world-grouped metrics, rate tables, and the machine decision, and require protected authorization fields to remain false.
- **Olivia:** add a repository-native one-GH200, 192-GiB, seven-day entrypoint. The launcher permits only v3-2 `full` with the complete stage sequence and a separately committed implementation-matching inspected plan. Any failure after test access consumes v3-2; there is no artifact-only same-registration retry.
- **Boundary:** no registered test world has been instantiated and the Qwen3-14B held revision remains task-unopened at this checkpoint.

## 2026-08-12 — repair the task-blind held-cache receipt helper

- **Failed staging job:** immutable Olivia cache job `1895296` (source archive `bbac2962...fab7b6`) failed in 15 seconds while importing the Python 3.11-only `datetime.UTC` name from the pinned Python 3.10 container.
- **Exposure audit:** the import failed before `snapshot_download`; the exact Qwen3-14B snapshot remained absent. The job executed zero task prompts, zero model inference, and no Stage-A panel generation or access.
- **Repair:** use the Python 3.10-compatible `datetime.now(timezone.utc)` spelling only. Preserve the failed immutable deployment and use a fresh source version and Slurm job for checkpoint staging.
- **Scientific boundary:** the frozen v3-2 implementation files and inspected plan are unchanged; this cache receipt repair does not alter any estimand, model interaction, seed, gate, or outcome path.

## 2026-08-12 — stage and verify the unopened held checkpoint

- **Staging:** immutable Olivia job `1895307` (source archive `76905ed3...be098`) completed in 60 seconds and resolved `Qwen/Qwen3-14B` exactly at `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- **Receipt:** hash all 18 resolved snapshot files totaling `29,552,613,776` bytes. An independent filesystem check finds eight referenced weight shards and zero missing or broken files.
- **Exposure:** the receipt records `task_prompts_executed=0` and `model_inference_executed=false`. The checkpoint is staged but remains task-unopened; only the ordered v3 workflow may expose it to registered train worlds after the founder freeze.
- **Plan:** the inspected plan remains byte-exact at internal SHA-256 `c71c1b26...11d0af`; cache staging changed none of its 64 bound implementation files.

## 2026-08-12 — reject a source archive containing fetched cluster artifacts

- **Detected in dry run:** the first v3 launcher plan produced deterministic source archive `f6103904...28b53`, but inspection found five ignored `.cluster-results/` receipt/log files inside it. No remote outcome job was created, no model was loaded, and no panel or test access existed.
- **Disposition:** reject that dry run and source archive. They are not eligible for submission even though the fetched files contain no credential or scientific outcome.
- **Repair:** add `.cluster-results` to the launcher's explicit archive exclusion set and a regression test. Because `src/frank_eq/cluster.py` is bound by the inspected plan, this pre-outcome engineering repair invalidates plan `c71c1b26...11d0af` and requires a freshly generated, inspected, separately committed plan.

## 2026-08-12 — refresh the plan after archive hygiene repair

- **Implementation repair:** commit `29e2cc3` adds only the fetched-artifact archive exclusion and its regression; it changes no Stage-A estimand, model path, panel, compiler, baseline, gate, or access order.
- **Replacement plan:** commit `8bc57fb` replaces the invalidated plan. The refreshed internal hash is `727348ba...34176`, the plan-file hash is `947e3dd5...9e320`, and the implementation-tree hash is `2ab50a10...15513` across the same 64 bound files.
- **Inspection:** exact model revisions, config hash, complete stage order, 1,824 prefix forwards, 213,408 logical source queries, one delayed test access, and all protected authorization fields are unchanged. The plan still records `held_model_task_opened=false` and `test_panel_instantiated=false`.

## 2026-08-12 — repair Stage-A import compatibility before model load

- **Failed smoke:** task-blind held runtime smoke `1895356` failed in 18 seconds while importing `src/frank_eq/stagea_v3/access.py`; the pinned Python 3.10 container does not expose the Python 3.11 `datetime.UTC` alias.
- **Exposure audit:** failure occurred during module import before tokenizer or model loading, neutral inference, registered worlds, operations, answers, panels, or test access. The held checkpoint remains task-unopened.
- **Repair:** replace `datetime.UTC` with the Python 3.10-compatible `timezone.utc` spelling and add a source-level regression covering every Stage-A v3 module. Timestamps and all scientific behavior are unchanged.
- **Plan consequence:** `access.py` is plan-bound. Plan `727348ba...34176` is therefore invalidated before outcome execution and must be regenerated, inspected, and committed again after the compatibility repair passes the full contract.

## 2026-08-12 — refresh the plan after the Python 3.10 repair

- **Replacement plan:** commit `a9a6b74` supersedes every earlier inspected plan. Its internal SHA-256 is `694408c6b9e6b63b68c393605da30c194b0feaad19fb737e93b091d5ff505922`, plan-file SHA-256 is `3f4f5740512fee3fb32169759e83ceec19bfec0b95877fe0a5de0d4f8edaf0b7`, and implementation-tree SHA-256 is `2fe48197d1ce513d21b95fbf87ecfbf9c69c481407b122298e4e330cc2141014` across 64 bound files.
- **Inspection:** exact model revisions, config hash `92d7ede...f5b3`, complete stage order, 1,824 prefix forwards, 213,408 logical source queries, one delayed test access, and every protected authorization field remain frozen. The plan records `held_model_task_opened=false` and `test_panel_instantiated=false`.
- **Boundary:** the compatibility change affects timestamp syntax only. No panel, task prompt, model response, test row, estimator, or gate was observed while replacing the plan.

## 2026-08-12 — pass the task-blind held runtime smoke

- **Execution:** immutable Olivia job `1895366` (source archive `f27a5fc6...22c05`) completed in 3 minutes 14 seconds on one GH200. It loaded `Qwen/Qwen3-14B` offline at exact revision `40c069824f4251a91eefaf281ebe4c544efd3e18` under the pinned image SHA-256 `a3ca46f0...aa3b1`.
- **Runtime checks:** a neutral engineering-only prefix produced a 5,120-wide hidden state, all requested layer states, and a cloneable KV cache; the frozen `chat_turn` construction also passed exact token-prefix continuity. Peak CUDA allocation was 29,648,355,328 bytes and measured smoke inference time was 175.4 seconds.
- **Exposure:** the receipt records `registered_worlds_loaded=0`, `registered_operations_scored=0`, `answers_scored=0`, `test_access_count=0`, and `inference_scope=neutral_prefix_only`. No registered panel or answer was constructed, and the held checkpoint remains task-unopened.
- **Evidence:** the fetched compact receipt has SHA-256 `bee594ac852729881e6a4dc1b6eefe36347ac1573e181366a29e589080779902`. It is retained outside Git under ignored `.cluster-results`; this log records its immutable identifiers without committing generated cluster artifacts.
- **Authority:** the smoke retires the known cache, image, Python-version, model-load, hidden-state, KV-clone, and chat-prefix runtime risks. It does not consume v3-2, open the official test role, authorize receiver execution, or change the frozen scientific contract.

## 2026-08-12 — launch the one authorized Stage-A v3-2 workflow

- **Local contract:** compile, Ruff, 110 tests, shell syntax, repository validation, and rate--compute validation all pass from a clean worktree at commit `91a0021b1bf336d0e40df1e4c6a68d68c08cc3f4`.
- **Archive inspection:** the repository launcher reproduced source archive SHA-256 `b81c704e8d2bf1efb402a8b42920d021dcc5463dac1ffe3747f24cf4a07cb47a` across two dry runs. An independent tar audit verified 208 regular files, every one of the 64 plan-bound file hashes, the config and plan hashes, required launcher/protocol files, no special or unsafe paths, and no Git state, agent state, fetched results, run data, environment credentials, W&B state, caches, or checkpoints.
- **Remote preflight:** the exact job and source targets were absent; all pinned 4B, 8B, and 14B snapshots contained their required tokenizer/config files and 3, 5, and 8 weight shards respectively with zero broken links. The W&B environment was mode `0600`; the account reported 2,907.58 available GPU-hours. Project storage was close to its file quota (`996,613/1,000,000`), but source and results are written under `/cluster/work` and all model cache files already exist.
- **Submission:** launch exactly `prepare,founder_fit,freeze,held_onboard,evaluate` with the frozen full profile as immutable Olivia Slurm job `1895410`, job name `frank-eq-stagea-v3-2-olivia-20260812a`, one GH200, 32 CPUs, 192 GiB memory, and a seven-day walltime. The scheduler placed it on `gpu-1-85` immediately.
- **Binding:** the job records config SHA-256 `92d7ede...f5b3`, inspected-plan internal SHA-256 `694408c6...505922`, plan-file SHA-256 `3f4f5740...af0b7`, image SHA-256 `a3ca46f0...aa3b1`, and source archive SHA-256 `b81c704e...cb47a`.
- **Authority:** this is the sole outcome-bearing v3-2 run. The process lock and access ledger must keep the test role unopened until both founder and held freezes exist and validate. A terminal failure after test access cannot be repaired by rerunning this registration; receiver execution and claim authorization remain false regardless of outcome.

## 2026-08-12 — preserve the pre-test float32 boundary failure

- **Failure:** immutable Olivia job `1895410` exited `1:0` after 58 minutes 2 seconds in `founder_fit`. It completed all 160 Qwen3-4B train/n4 rows and 17,280 logical teacher queries, then `V3CaptureShard.validate()` rejected a behavioral soft target with `behavioral targets must lie strictly inside (0,1)` before the shard was serialized.
- **Cause:** the teacher's finite semantic candidate log-odds were retained, but converting an extreme sigmoid probability to float32 can round it to exactly `1.0`. The capture schema already requires behavioral targets to remain in the open unit interval; the implementation lacked the final storage-boundary clamp needed to honor that invariant.
- **Exposure audit:** `workflow_status.json` records `test_access_consumed=false`; `access_ledger.json` records `test_access_count=0`, no registered test files, and no test-file opens. There is no capture file, founder freeze, held-onboarding manifest, test panel, decision, or scientific metric. The held Qwen3-14B model remained task-unopened.
- **Preservation:** fetch the failed run under ignored `.agents/state`. The fetched workflow-status SHA-256 is `84d1ed4526702cee4446be5bdf89862ba04f3f3ba1c7f4c7f2042f4ffe7c2996`, access-ledger SHA-256 is `3c57de50b5455bbeeff20f47f55c43de60650259c2270e625fb69fa47fe3495a`, stderr SHA-256 is `83dd616b2227022a9b4f71b3c4c82907704e7af8492431fd40a925050d2c1d7d`, and stdout SHA-256 is `cf647a79e4ede9a42e289a46d24a0fd33fca2ec79e1a3a1bd8602f8ee4179672`.
- **Disposition:** this is an engineering failure before test creation under Section 14 of the frozen protocol, so it does not consume the v3-2 outcome. Preserve job `1895410` immutably. A retry requires a minimal Stage-A-only numeric repair, focused regression, complete local validation, a new bound plan, a newly inspected deterministic archive, and a fresh job name. No scientific field, raw log-odds, model, panel, seed, architecture, loss, baseline, gate, or access order may change.

## 2026-08-12 — repair Stage-A behavioral probability storage

- **Numeric repair:** preserve every finite semantic-sequence teacher log-odds value unchanged, but clamp the Stage-A behavioral probability tensor after float32 conversion to `[1e-7, 1-1e-7]`. This implements the pre-existing open-unit-interval capture invariant and is explicitly a storage boundary, not calibration or target tuning.
- **Auditability:** add `behavioral_probability_storage_epsilon` and `behavioral_probability_clamp_count` to every capture summary. Both the runtime integrity reducer and independent verifier require the registered epsilon and a nonnegative integer count.
- **Regression:** drive the capture path with exact `0.0` and `1.0` teacher probabilities plus retained `-80/+80` log-odds; require strict open-interval targets, an exact clamp count, unchanged scores, successful validation, atomic serialization, and hash-checked reload.
- **Scientific boundary:** model revisions, panels, seeds, queries, response protocols, raw scores, architecture, training, baselines, gates, bootstrap, and access order are unchanged. No held task prompt or test artifact was accessed while making this repair.
- **Plan consequence:** `capture.py`, `workflow.py`, `verify.py`, and the protocol are plan-bound. Inspected plan `694408c6...505922` is invalidated and cannot authorize a retry; generate, inspect, and separately commit a new plan only after the repair commit.

## 2026-08-12 — rebind and freeze the numeric-repair plan

- **Registration binding:** update only the frozen protocol content hash in `configs/stagea_v3/registration.json` to `250f4e1d65d06c4b0ff86db165f958c11d6ab216f2a6f53db3f42799e8fda8fc`; the config hash and protocol version remain unchanged.
- **Validation:** compile, Ruff, 110 tests, shell syntax, repository validation, and rate--compute validation all pass after repair commit `d40cdf5` and the registration rebind.
- **Replacement plan:** internal SHA-256 `7b5098588c9368b64694eaf248713810ebc714a2d7f7e6891f93cb7cd16a113c`, plan-file SHA-256 `9a728f192b04035281e616822e9a468615a4969c8111a488ec58f28cd240cbf8`, and implementation-tree SHA-256 `bf8c87fa98116abff8d872b7fea0b6120d9337856d82f7d29778ccc3617812fc` across 64 bound files.
- **Inspection:** model revisions, config SHA-256 `92d7ede...f5b3`, stage order, 1,824 prefix forwards, 213,408 logical source queries, one delayed test grant, and every protected authorization are unchanged. The plan records `held_model_task_opened=false` and `test_panel_instantiated=false`.
- **Authority:** this plan supersedes all earlier v3-2 plans and may authorize only one fresh immutable retry after a clean deterministic source archive is independently inspected. Job `1895410` remains immutable failure evidence.

## 2026-08-12 — launch the fresh immutable pre-test retry

- **Dry-run archive:** reproduce source archive SHA-256 `b6203d03031d928f5006415f2a6b4dccd16bc27f7a4a0f7b298d50936f4a44e0` across two clean dry runs at Git commit `4f93143c357a4d77f905c41a3aeb1d7de1418473`. Independent tar inspection verifies all 208 regular files, all 64 plan-bound hashes, required repair/protocol/registration/launcher files, no unsafe or special entries, and no Git state, agent state, fetched results, run data, environment credentials, W&B state, caches, or checkpoints.
- **Remote preflight:** the retry job/source targets were absent; failed job `1895410` remained present; all three exact snapshots retained their 3/5/8 weight shards with zero broken links; W&B credentials remained mode `0600`; the account had 2,906.61 GPU-hours available and the accelerator partition had idle capacity.
- **Submission:** launch the identical full stage sequence as fresh Olivia Slurm job `1899057`, job name `frank-eq-stagea-v3-2-olivia-20260812b`, one GH200, 32 CPUs, 192 GiB memory, and seven-day walltime. It started immediately on `gpu-1-85`.
- **Binding:** config SHA-256 `92d7ede...f5b3`, plan internal SHA-256 `7b509858...a113c`, plan-file SHA-256 `9a728f19...0cbf8`, implementation-tree SHA-256 `bf8c87fa...812fc`, image SHA-256 `a3ca46f0...aa3b1`, and source archive SHA-256 `b6203d03...a44e0`.
- **Boundary:** this retry is permitted solely because the preserved prior failure occurred before any capture serialization, founder/held freeze, held-task exposure, or test access. All scientific fields remain frozen; job `1895410` is not resumed or overwritten.

## 2026-08-12 — cross the numeric failure and freeze the founders

- **Repair confirmation:** retry job `1899057` successfully validates, atomically writes, and hash-binds the formerly failing Qwen3-4B train/n4 shard. Its 160 rows and 17,280 logical queries complete under exact revision `1cfa9a...df60c`; the manifest SHA-256 `2210af85c6c589341dd5a6ded6dba220c0ac24cd9e2f83cbabdb5372923e5c3c` independently matches the 1,103,639,553-byte file.
- **Numeric audit:** the shard records storage epsilon `1e-7` and 1,115 clamped behavioral probability values. Stored float32 targets span `1.0000000116860974e-07` to `0.9999998807907104` and are strictly inside `(0,1)`, while raw log-odds remain `[-19.6875, 21.625]`. Exact-replay branches and primary compiler post-capture queries are both zero.
- **Founder freeze:** `freeze_manifest.json` records status `frozen` at `2026-08-12T06:14:41.959524+00:00`, exact config hash, founders Qwen3-4B/Qwen3-8B, 50 artifacts, and `test_files_existing=[]`. All eight expected founder capture paths are included, no held or test capture is included, and every frozen artifact exists.
- **Causal order:** the access ledger enters `freeze` at `06:14:47Z` and `held_onboard` at `06:14:52Z`. Only after that transition does Qwen3-14B see registered train/validation prefixes. Both held train captures are now durable; held validation is running. Test access count, registered test files, and test-file opens remain zero.
- **Authority:** the founder freeze establishes the repair and unlocks only the registered held-local onboarding step. It does not authorize test access early, receiver execution, a claim, or any adaptive change.

## 2026-08-12 — freeze the held sender and consume the single test grant

- **Held completion:** Qwen3-14B completes all train/validation captures and the 15 registered held-local checkpoint units: three seeds each for semantic, behavioral, token-ID, final-token, and continuous conditions.
- **Held freeze:** `held_onboarding_manifest.json` records status `frozen` at `2026-08-12T09:22:18.678176+00:00`, held model `qwen3-14b-held`, 22 artifacts, exact config hash, `test_files_existing=[]`, and founder-freeze SHA-256 `f6895ff9aaf4d4796ef06b40d10b1d14c961b85c5b75d91297d60743a1786eca`. An independent hash of `freeze_manifest.json` matches that value; every held artifact exists and none is a test artifact.
- **One-way grant:** only at `2026-08-12T09:22:32.879829+00:00`, 14 seconds after the held freeze, the access ledger records `test_access_consumed` and enters `evaluate`. The count is exactly one, with 21 registered panel/capture/prediction paths.
- **First opens:** `test_panel_manifest.json` opens at `09:22:32.918378Z` under SHA-256 `c4362a05...0e215`; `panels/test_n4.json` and `panels/test_n6.json` follow under independently matching hashes `ae0f4123...3a843` and `8c972ca7...126bd`. No test capture existed during the boundary audit; Qwen3-4B test capture then began.
- **Irreversibility:** v3-2 is now consumed. Any subsequent application, scheduler, verification, or artifact failure is part of the outcome and cannot be repaired or rerun under this registration. Compiler seeds, checkpoints, controls, calibrators, protocols, thresholds, renderers, and gates are frozen; receiver execution and every claim field remain unauthorized.

## 2026-08-12 — complete the Qwen3-4B test captures

- **n=4:** job `1899057` completes all 96 test worlds and 10,368 logical queries with zero recorded errors, then atomically writes `qwen3-4b-n4.pt`. Its 780,205,761 bytes independently hash to SHA-256 `c44e661d828c612526ec5aaa7d294b6527ff1eb44281461f0e889ef43d2e6c03`.
- **n=6:** the same immutable job completes all 96 test worlds and 12,096 logical queries with zero recorded errors, then atomically writes `qwen3-4b-n6.pt`. Its 1,397,854,209 bytes independently hash to SHA-256 `938ed4186033eda481f9c5d88ac5333a8048f5c3602468b76a71e664e7d758e0`.
- **Boundary:** these are capture-completion and integrity observations only. No raw test outcome is inspected or used adaptively; evaluation proceeds to the remaining frozen model/complexity strata. The consumed registration, no-retry rule, receiver lock, and claim locks are unchanged.
