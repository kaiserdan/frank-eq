# Stage R / RC0 cluster runbook

Status: completed on Olivia and adopted. This runbook is retained for exact
provenance and verification; no RC0 rerun or engineering replication is currently
authorized.

Read first:

```text
docs/18_RATE_COMPUTE_OPERATIONAL_BASIS.md
HANDOFF.md
AGENTS.md
```

## 1. Local validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
```

## 2. LUMI preflight

The frozen model revisions are already the revisions used in the latest source
screening. Confirm that both snapshots remain in the shared Hugging Face cache:

```text
Qwen/Qwen3-4B  1cfa9a7208912126459214e8b04321603b3df60c
Qwen/Qwen3-8B  b968826d9c46dd6066d109eabc6255188de91218
```

Dry run:

```bash
python lumi/cli.py submit \
  --job-name frank-eq-rc0-rate-compute \
  --config configs/rate_compute/real_lumi_rc0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

Submit the exact inspected source package:

```bash
python lumi/cli.py submit \
  --job-name frank-eq-rc0-rate-compute \
  --config configs/rate_compute/real_lumi_rc0.yaml \
  --profile full \
  --stages audit \
  --json
```

Monitor, fetch, and verify:

```bash
python lumi/cli.py status --job-name frank-eq-rc0-rate-compute --json
python lumi/cli.py fetch  --job-name frank-eq-rc0-rate-compute --json
python lumi/cli.py verify --job-name frank-eq-rc0-rate-compute --json
```

Run the RC0-specific verifier after fetch:

```bash
python scripts/verify_rate_compute_run.py \
  --run .agents/state/lumi/frank-eq-rc0-rate-compute/remote/runs
```

The exact fetched run root may include the job-specific run directory; use the
path containing `run_manifest.json`, `workflow_status.json`, `metrics.json`, and
`decision.json`.

## 3. Olivia alternative

Olivia `accel` is ARM64 Grace Hopper. Use the inspected native image and shared
offline cache; the old AMD64 image and x86_64 wheelhouse are incompatible:

```bash
export FRANK_EQ_OLIVIA_ROOT=/cluster/work/projects/nn12027k/dakai5365/frank-eq
export FRANK_EQ_OLIVIA_IMAGE=/cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif
export FRANK_EQ_IMAGE_SHA256=a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1
export FRANK_EQ_HF_HOME=/cluster/projects/nn12027k/hf-cache
```

For the completed RC0, an immutable source version passed the development-only
runtime smoke in `olivia/rc0_runtime_smoke.slurm`. The smoke exercises both
pinned models, exact prefix continuity, exclusive cloned-KV branches, semantic
candidate scoring, and the matched 32-token reasoning/pause paths. It does not
load an RC0 panel and is not a scientific result. See `docs/OLIVIA.md` for the
exact deployment and smoke commands.

Dry run:

```bash
python olivia/cli.py submit \
  --job-name frank-eq-rc0-rate-compute \
  --config configs/rate_compute/real_olivia_rc0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

Then submit, status, fetch, and verify with the same command pattern using
`olivia/cli.py`.

Olivia must have both exact revisions available before launch. Do not silently
substitute another checkpoint, permit network resolution to an unpinned head,
install packages at runtime, or submit if the development smoke fails.

## 4. Expected artifacts

```text
config.yaml
run_manifest.json
workflow_status.json
development_splits.json
panels/n4.json
panels/n6.json
models.json
records_raw.jsonl
calibration.json
records_calibrated.jsonl
compiled_predictions.jsonl
direct_protocol_selection.json
metrics.json
decision.json
artifact_manifest.json
run_summary.json
```

Generated reasoning text and exact token counts live in the response rows.
`models.json` records requested/observed revisions, answer-token IDs, semantic
candidate token IDs, prefix-continuity counts, and exclusive cloned-KV branch
accounting.

The verifier checks file hashes and preserves scientific failure as a successful
scheduler run.

### Artifact-only recovery after a post-capture engineering failure

A failed RC0 job may be recovered without repeating model inference only when all
of the following are true:

- both pinned models completed every registered branch;
- `records_raw.jsonl`, `calibration.json`, and `records_calibrated.jsonl` are
  complete and hash-frozen;
- no compiled predictions, metrics, machine decision, summary, or artifact
  manifest existed in the failed run;
- the original failed job remains immutable;
- a fresh content-addressed source snapshot contains only a mechanical repair
  that leaves the frozen config, estimands, bootstrap seeds, gates, and decision
  reducer unchanged;
- a recovery input manifest binds the original job ID, source archive, config,
  failure, remote run root, and SHA-256 of every reused artifact;
- the fresh recovery job still runs exactly `--stages audit`, copies rather than
  modifies the original artifacts, and records that model capture was not
  executed;
- the recovery job reaches scheduler `COMPLETED` and passes both repository and
  RC0-specific verification.

This is continuation of the same development audit, not a second scientific look.
If any post-calibration outcome already exists, recovery fails closed and no
adaptive repair or retry is permitted. On Olivia, use:

```bash
python olivia/cli.py submit \
  --job-name <fresh-recovery-job> \
  --config configs/rate_compute/real_olivia_rc0.yaml \
  --profile full \
  --stages audit \
  --recover-from-job <failed-job> \
  --dry-run --json
```

Inspect the content-addressed source and recovery-input hashes, then remove only
`--dry-run`. The recovered package must include `recovery_provenance.json`.

## 5. Operational integrity checks

Before interpreting metrics, verify:

- both entity-count panels exist and match the frozen hashes;
- only train/validation development worlds were used;
- both renderer views are present for every model/world;
- all branches use cloned KV reuse;
- `branch_batch_size: 8` is honored with query-exclusive cache slots and the
  model metadata records the observed response-batch accounting;
- no exact replay fallback occurred;
- every target protocol has the registered generated-token count;
- semantic candidate labels and tokenizer IDs are recorded;
- calibration and direct-protocol selection use training worlds only;
- the decision keeps new-outcome, receiver, test, and claim authorization false.

## 6. Reading the decision

### `BASIS_READOUT_NOT_QUALIFIED`

Stop. The models do not expose a stable directed-edge basis under this answer
and compute contract. Do not train Stage-A v3.

### `PUBLIC_BASIS_NOT_SUFFICIENT`

The source basis passes but the public executor does not beat the operation
prior. Inspect structured calibration and dependence assumptions on development
data only.

### `NO_COMPOSITION_ADVANTAGE_OVER_TRAIN_SELECTED_DIRECT_BASELINE`

The basis is usable but not a stronger constructive interface than direct
reasoning. Preserve as a diagnostic and do not claim an architecture result.

### `PUBLIC_BASIS_COMPOSITION_SUPPORTED`

The machine decision permits drafting one fresh Stage-A v3 registration. It does
not permit launching it. The v3 registration must freeze:

- new claim-bearing world seed;
- new unopened held model;
- model-local token/slot compilers into public edge coordinates;
- separate behavioral and semantic channels;
- text/token and direct-operation baselines;
- paired world units, intervals, and hard kills;
- receiver execution still locked.

## 7. Evidence adoption

Generated caches and responses remain outside Git until reviewed. Adopt only a
small package containing:

```text
frozen config and source identity
workflow verification summary
metrics.json
decision.json
compact independent audit
SHA-256 manifest
```

Do not commit model caches, raw checkpoints, scheduler state, W&B credentials,
or `.agents/state/`.

The adopted compact package is `evidence/real_stage_r_olivia_rc0/`. It binds the
original capture, artifact-only recovery, frozen config and runtime image,
repository and specialized verification, machine decision, world-grouped
metrics, and independent recomputation through SHA-256.
