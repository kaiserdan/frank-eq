# Stage R / RC0 cluster runbook

Status: ready for one development-only run on either LUMI or Olivia. The two
configs encode the same scientific contract. Run one cluster first; use the
second only as an engineering replication, not as an adaptive second chance.

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
substitute another checkpoint or permit network resolution to an unpinned head.

## 4. Expected artifacts

```text
run_manifest.json
workflow_status.json
panel_4.json
panel_6.json
responses.jsonl
response_metadata.json
reasoning_samples.json
metrics.json
decision.json
artifact_manifest.json
run_summary.json
```

The verifier checks file hashes and preserves scientific failure as a successful
scheduler run.

## 5. Operational integrity checks

Before interpreting metrics, verify:

- both entity-count panels exist and match the frozen hashes;
- only train/validation development worlds were used;
- both renderer views are present for every model/world;
- all branches use cloned KV reuse;
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
