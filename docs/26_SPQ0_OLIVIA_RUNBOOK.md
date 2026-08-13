# SPQ0 Olivia runbook

Status: prospective development-only procedure. Do not launch from the
implementation PR.

## Authority

The completed graph line is immutable and closed:

```text
Stage-A v1/v2       exact-pipeline negatives
Stage Q             development negatives
RC0                 development tomography pass; consumed
Stage-A v3-2        ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED; consumed
Stage M0            OPERATION_CLOSED_EVENTS_NOT_READABLE; consumed
```

SPQ0 is a fresh controlled-stochastic-system census. It is not a graph rerun or
retune. One development `audit` may be submitted only after this branch is
reviewed, merged or otherwise explicitly selected, all prerequisites below
pass from a clean commit, and an operator separately authorizes launch.

No test role, held sender, receiver, claim-bearing world, scientific claim, or
paper claim is authorized. OLMo2 and Granite are reserved unopened. A passing
decision authorizes only an SPQ1 protocol draft.

## Frozen Olivia resources

```text
account:       nn12027k
partition:     accel
nodes:         1
GPUs:          1 GH200
CPUs:          32
host memory:   128 GiB
wall time:     7-00:00:00
HF cache:      /cluster/projects/nn12027k/hf-cache
image:         /cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif
image SHA-256: a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1
```

The launcher fixes these values for SPQ0; environment overrides cannot replace
its image, image hash, or inspected plan. Runtime package installation and
network checkpoint resolution are forbidden.

## Frozen files

```text
config:        configs/spq0/real_olivia_spq0.yaml
protocol:      docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md
registration:  configs/spq0/registration.json
plan:          configs/spq0/inspected_plan.json
verifier:      scripts/verify_spq0_run.py
```

The two planning attachments at repository root record the prospective design
input. The executable config and protocol above are authoritative. The old
`agent/predictive-state-psr0` branch is a code donor only and must never be
merged or executed; its stochastic true/false response protocol is invalid.

## Pre-run local gate

Run from the repository root and stop on any failure:

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
python scripts/validate_moment_compute.py
python scripts/validate_spq0.py
git diff --check
git status --short
```

`git status --short` must be empty before any non-dry submission. Historical
evidence validation must pass without modifying any file under `evidence/`.

Regenerate the deterministic plan only while prospectively editing the
registration, never after reading model outcomes:

```bash
python -m frank_eq.shared_predictive_quotient.cli plan \
  --config configs/spq0/real_olivia_spq0.yaml \
  --out configs/spq0/inspected_plan.json
python scripts/validate_spq0.py
```

Independent plan generation must reproduce the committed file byte-for-byte.
The plan's public-basis digest uses the protocol's declared 10-decimal numeric
canonicalization and 14-decimal selection-score tie resolution so different
LAPACK backends cannot alter plan identity. Runtime arrays remain float64. Any
change requires review and a new clean commit before model access.

## Dry-run inspection

Use a fresh job name. This command packages source and prints the submission;
it does not contact the scheduler with `sbatch`:

```bash
python olivia/cli.py submit \
  --job-name frank-eq-spq0-olivia-20260813a \
  --config configs/spq0/real_olivia_spq0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

Run the identical dry-run twice and require exact equality. Inspect and record:

- `git.dirty` is false and `git.commit` is the intended commit;
- the content-addressed source archive SHA-256 is identical both times;
- config file SHA-256 and both internal/file plan SHA-256 values;
- active revision-registry SHA-256 and the two exact revisions;
- reserved-checkpoint non-access contract SHA-256 and the two reserved
  revisions;
- runtime image path and exact non-null SHA-256;
- `profile=full`, `stages=audit`, `partition=accel`, and seven-day wall time;
- a fresh immutable remote root with no prior submission state.

Do not use `--recover-from-job`: SPQ0 has no same-registration recovery path.

## Checkpoint preflight order

On an eventual run, `olivia/quickstart.sh` validates the config and inspected
plan before the workflow starts. The workflow then performs checkpoint
preflight before constructing any model adapter:

1. resolve Qwen3-4B locally at the exact 40-hex revision;
2. hash every resolved file and the complete file registry;
3. resolve Mistral-7B-v0.3 locally at its exact revision;
4. hash every resolved file and registry;
5. write zero access/load/inference counters for OLMo2 and Granite;
6. require the frozen user-prefix / assistant-ack / user-query turn shape and
   exact prefix continuity under each active tokenizer;
7. only then construct active model adapters and begin capture.

Missing active snapshots, broken files, revision mismatch, absent weights, a
network requirement, or any reserved access fails before task inference.

## Submission boundary

The implementation request explicitly forbids automatic launch. Do not remove
`--dry-run` as part of preparing or opening the PR. A later operator who has
separate launch authority may submit exactly the inspected command without the
dry-run flag. No other profile, stage list, config, image, recovery source, or
resource override is permitted.

W&B is fail-open telemetry only. Credentials are sourced from the protected
Olivia environment and must not enter Git, the source archive, Slurm export
arguments, logs, or manifests. Machine artifacts and the independent verifier
remain authoritative if telemetry is absent.

## Status, fetch, and verify after a separately authorized run

```bash
python olivia/cli.py status \
  --job-name frank-eq-spq0-olivia-20260813a --json

python olivia/cli.py fetch \
  --job-name frank-eq-spq0-olivia-20260813a --json

python olivia/cli.py verify \
  --job-name frank-eq-spq0-olivia-20260813a --json

python scripts/verify_spq0_run.py \
  --config configs/spq0/real_olivia_spq0.yaml \
  --run .agents/state/olivia/frank-eq-spq0-olivia-20260813a/remote/runs
```

Monitor scheduler state and both stdout/stderr. Lack of new log output during
long model capture is not by itself evidence of a stall; compare response
branch counters, artifact modification times, GPU process state, and Slurm
elapsed time. Do not mutate a running source tree or remote job root.

The fetch includes generated captures, fitted arrays, predictions, and hashes.
They remain outside Git. Do not commit `.agents/state/`, `.cluster-results/`,
raw runs, model caches, checkpoints, W&B state, credentials, or tokens.

## Result audit

Require all repository and specialized verifier checks before interpreting the
decision. Audit at minimum:

- exact config, protocol, registration, plan, source, image, and checkpoint
  identities;
- zero reserved-checkpoint access and absence from capture manifests;
- exact prefix continuity for every response branch;
- exclusive cloned-KV use and zero replay;
- disjoint calibration/selection/validation histories and zero test rows;
- validation-only system, symbolic renderer, and length-32 strata;
- all five capture surfaces and parameter-matched token-sequence control;
- refit/freeze order for complete source encoders and target-local readers;
- both ordered cross-family compositions and zero pair-specific mappers;
- full rank sweep, quantization, sender leakage, amortized rate frontier, and
  non-promotional residual census;
- grouped metrics, machine decision, closed claim fields, and independent
  recomputation.

Only a compact, hash-verified evidence package may be proposed for adoption,
and adopting it is a separate reviewed change. Preserve the complete fetched
run outside Git. Follow the machine diagnosis exactly; never tune the frozen
thresholds after validation is read.
