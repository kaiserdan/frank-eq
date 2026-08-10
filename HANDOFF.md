# Frank-EQ handoff

Snapshot: 2026-08-10

## Current state

Synthetic Stage 0 is complete and its adopted reference returns `PROMOTE_REAL_MODEL_CANARY` with `authorizes_scientific_claim=false`.

The first real Stage-A canary has now run and produced a **valid negative decision**: `STOP_OR_REVISE_STAGE0` (`frank-eq-stagea-devg-v2` on LUMI `dev-g`, evidence under `evidence/real_stagea_devg_v2/`). Engineering integrity passed end to end (cache validation, 100% KV-reuse branches, W&B telemetry synced to project `frank-eq-stagea`). The failure is localized to capture sufficiency: per-edge graph facts are not linearly readable from the captured hidden states of the small frozen models under raw prompts, while the declared global coordinates are. The frozen config was not modified and no gates were tuned.

Real Stage-A now logs fail-open W&B telemetry (project `frank-eq-stagea`) covering run identity, cache branch-mode accounting, per-epoch training losses, and evaluation metrics. Olivia additionally uses a new amd64 PyTorch container (`pytorch-2.5.1-cuda12.4-runtime-amd64.sif`), because every `accel` node is H200/x86_64 and the previous arm64 images cannot execute there.

## Real Stage-A implementation

Implemented:

- controlled closed-world relational panel with exact facts and future operations;
- two renderer forms per world and world-grouped train/validation/test splits;
- two founder checkpoints plus exactly one held sender;
- normalized-depth hidden capture before operation reveal;
- physical KV-reuse branching where supported, exact-prefix replay fallback otherwise;
- single-token canonical A/B outcome mapping audited per tokenizer;
- formal oracle outcomes and source-model branch probabilities in separate arrays;
- immutable `FutureSignatureRecord` JSONL with prefix, hidden, and operation hashes;
- parameter-free graph interrogator over facts and two public global coordinates;
- existing founder training, frozen-executor held-sender onboarding, evaluator, bootstrap, and reducer reused without target hidden-state reconstruction;
- cache validator with causal-order, split, coverage, descriptor, hidden, and file-hash checks;
- generic real workflow with manifest/status and stage-resume boundaries;
- content-addressed source packaging and Olivia/LUMI submit/status/fetch/verify commands;
- real smoke and cluster configs, tests, and operator skills.

## Frozen canary configuration

Olivia:

```text
configs/stage0/real_olivia.yaml
founders: Qwen/Qwen3-0.6B, HuggingFaceTB/SmolLM-1.7B-Instruct
held sender: meta-llama/Llama-3.2-1B-Instruct
worlds/renderers/operations: 64 / 2 / 16
capture depths: 0.35 / 0.60 / 0.85
primary decoder: frozen graph interrogator
```

LUMI uses the same scientific configuration under `configs/stage0/real_lumi.yaml`; only run identity differs.

## Immediate launch sequence

1. Confirm all three pinned checkpoints exist under the selected cluster `HF_HOME` and that gated Llama access is available.
2. Run the dry-run command and inspect source hash, remote root, config, stages, and Slurm script.
3. Submit `cache,validate` first if cluster/model compatibility is uncertain.
4. Fetch and verify the cache before launching or resuming `train,eval`.
5. Treat a negative `eval/decision.json` as a valid scientific result. Do not tune gates or layers from test outcomes.

For Olivia, export before submitting:

```bash
export WANDB_API_KEY=<key>            # W&B telemetry for the frank-eq-stagea project
export FRANK_EQ_ALLOW_PIP_INSTALL=1   # container lacks transformers/wandb
export FRANK_EQ_PIP_FIND_LINKS=/cluster/projects/nn12027k/frank-eq-wheels  # offline wheels fallback
```

`WANDB_API_KEY` is forwarded to the job through `sbatch --export` and the Apptainer environment; it is never written to source, configs, submission state, or logs.

Olivia:

```bash
python olivia/cli.py submit \
  --job-name frank-eq-stagea-cache-v1 \
  --config configs/stage0/real_olivia.yaml \
  --profile full --stages cache,validate --dry-run --json
```

LUMI:

```bash
python lumi/cli.py submit \
  --job-name frank-eq-stagea-cache-lumi-v1 \
  --config configs/stage0/real_lumi.yaml \
  --profile full --stages cache,validate --dry-run --json
```

## Evidence boundary

The real cache stores two distinct targets:

- `signatures`: formal oracle operation outcomes used by the frozen public interrogator;
- `model_signatures`: the source model's actual post-reveal A/B probabilities, used only as a behavioral diagnostic.

This separation prevents a weak source model from redefining the public operation semantics while still allowing the paper to measure whether latent-state sufficiency and overt behavior diverge.

## Known risks

- Some Transformers cache implementations may not be safely cloneable. `auto` records physical `kv_reuse` where successful and exact-prefix replay where it falls back. A claim of literal cached-state branching must report this count and may require a no-fallback rerun.
- Raw prompts rather than model-specific chat templates are used to keep the prefix/query token boundary exact and shared. Branch accuracy may therefore be low; it is diagnostic, not the primary target.
- The graph residual coordinates are declared density and reciprocity state. They are not yet evidence for an irreducible natural-language operational residual.
- The public graph interrogator is task-specific. Passing it would justify a second task family, not a universal interface claim.
- Receiver-native execution, rate-matched text controls, confirmation, and locked data remain unimplemented and unauthorized.

## Do not do next

Do not add target-state reconstruction, pair-specific translators, receiver gradients, or a learned receiver rescue if Stage A fails. First localize failure among capture sufficiency, fact extraction, renderer invariance, operation generalization, and held-sender retention using the frozen artifacts.

The negative outcome has been localized to capture sufficiency (fact head unfittable, global coordinates readable). The next scientific action is a user decision on a revised Stage-A protocol — for example chat-templated prefixes, a different depth grid, larger chart capacity, or a stronger fact objective — frozen as a new versioned config with a fresh untouched test role before any outcome-bearing run. Do not tune the current gates, layers, or panel.
