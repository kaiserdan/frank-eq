# Stage M0 Olivia runbook

## Authority

Stage M0 completed as the development-only Olivia audit:

```text
job:       frank-eq-moment-compute-m0 / Slurm 1970800
state:     COMPLETED 0:0
decision:  OPERATION_CLOSED_EVENTS_NOT_READABLE
evidence:  evidence/real_stage_m_olivia_m0/
```

Stage-A v3-2 remains consumed and terminal. Do not rerun it, RC0, or Stage Q.
Do not rerun Stage M0 or treat these historical commands as execution authority.
There is no authorized cluster executable.

## Local validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
python scripts/validate_moment_compute.py
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
```

## Historical content-addressed dry run

```bash
export FRANK_EQ_IMAGE_SHA256=a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1
python olivia/cli.py submit \
  --job-name frank-eq-moment-compute-m0 \
  --config configs/moment_compute/real_olivia_m0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

The executed plan rejected a null or different runtime image hash and bound the
source archive, config, exact Qwen revisions, clean Git state, fresh remote
target, and checkpoint availability. The command is retained for provenance,
not resubmission.

Historically, reject a plan whose runtime image hash is null or differs from the exported
digest. Inspect the source SHA-256, config SHA-256, exact Qwen revisions, clean
Git state, fresh remote target, and checkpoint availability. Submit the
identical command without `--dry-run` only after that inspection.

## Completed retrieval and verification

```bash
python olivia/cli.py status --job-name frank-eq-moment-compute-m0 --json
python olivia/cli.py fetch  --job-name frank-eq-moment-compute-m0 --json
python olivia/cli.py verify --job-name frank-eq-moment-compute-m0 --json
python scripts/verify_moment_compute_run.py \
  --run .agents/state/olivia/frank-eq-moment-compute-m0/remote/runs
```

A scientific gate failure must still finish as a valid scheduler-level run. The
independent verifier checks artifact hashes, disjoint development roles, exact
public algebra, prediction recomputation, metric recomputation, the machine
decision, and closed protected authorizations.

For job `1970800`, Slurm and the repository verifier pass. The in-job and
NumPy-2.2.6 exact-runtime specialized verifiers reproduce all 1,824 predictions,
metrics, and the negative decision exactly. The fetched tree matches Olivia by
checksum. A newer NumPy changes only diagnostic projection-adjustment summaries
and changes no scientific field or decision.

## Expected artifact surface

```text
config.yaml
run_manifest.json
workflow_status.json
development_splits.json
panels/n4.json
panels/n6.json                 # explicit unused-role tombstone
event_registry.json
models.json
records_raw.jsonl
calibration.json
records_calibrated.jsonl
direct_protocol_selection.json
compiled_predictions.jsonl
metrics.json
decision.json
run_summary.json
artifact_manifest.json
independent_verification.json
```

Generated response records and operational state remain outside Git. The
compact, hash-verified package is adopted under
`evidence/real_stage_m_olivia_m0/`.
