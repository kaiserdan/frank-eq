# Olivia runbook

## Local environment

The operator CLI uses your SSH alias and project root from:

```bash
export FRANK_EQ_OLIVIA_HOST=olivia
export FRANK_EQ_OLIVIA_ROOT=/cluster/home/dakai5365/project/frank-eq
```

Defaults match these values. Authentication remains outside the repository.

## Checkpoint preflight

`configs/stage0/real_olivia.yaml` sets `local_files_only: true`. Before submission, verify that the configured revisions exist under the cluster `HF_HOME` and that the Llama license/token has already been accepted. The Slurm default is:

```text
/cluster/projects/nn12027k/hf-cache
```

Override with `FRANK_EQ_HF_HOME` at submission/runtime when required.

## Dry run

```bash
python olivia/cli.py submit \
  --job-name frank-eq-stagea-cache-v1 \
  --config configs/stage0/real_olivia.yaml \
  --profile full \
  --stages cache,validate \
  --dry-run --json
```

Inspect:

- source archive SHA-256 and file count;
- exact config path;
- remote source/job roots;
- Slurm script;
- stages.

## Submit and monitor

```bash
python olivia/cli.py submit \
  --job-name frank-eq-stagea-cache-v1 \
  --config configs/stage0/real_olivia.yaml \
  --profile full \
  --stages cache,validate --json
python olivia/cli.py status --job-name frank-eq-stagea-cache-v1 --json
```

The source package is extracted under:

```text
$FRANK_EQ_OLIVIA_ROOT/jobs/<job>/source
```

The immutable source archive is retained under `sources/<sha256>/source.tar.gz`.

## Fetch and verify

```bash
python olivia/cli.py fetch --job-name frank-eq-stagea-cache-v1 --json
python olivia/cli.py verify --job-name frank-eq-stagea-cache-v1 --json
```

Local state is under:

```text
.agents/state/olivia/<job>/
```

A cache-only verification requires `dataset.npz`, `metadata.json`, and `cache_validation.json`. A full verification additionally requires the training checkpoint and evaluation decision.

## Resume training/evaluation

The generic submitter creates a new source/job root per job name. To reuse a prior cache, copy it only through an explicit provenance-controlled job or run the full workflow in one allocation. The default first campaign should run all stages after cache compatibility has been established.

## Result interpretation

- Slurm `COMPLETED` plus workflow `completed` means engineering integrity.
- `eval/decision.json: status=fail` is a valid scientific failure and produces only a verifier warning.
- Missing or invalid cache hashes are engineering failures and prohibit training.
