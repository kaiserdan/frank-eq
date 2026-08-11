# Olivia runbook

## Current authority

Stage R / RC0 completed on Olivia and is adopted as a development pass:

```text
capture:  frank-eq-rc0-rate-compute-olivia-20260811c  Slurm 1874736
recovery: frank-eq-rc0-rate-compute-olivia-20260811d-recovery  Slurm 1891471
result:   PUBLIC_BASIS_COMPOSITION_SUPPORTED
```

Stage-A v3-2 is now frozen and the user authorized one sequential representation
run after separate registration and implementation commits, full validation,
held-checkpoint staging, and an inspected content-addressed dry run. See
`docs/20_STAGEA_V3_PROTOCOL.md`.

This does not authorize an RC0 rerun, receiver execution, receiver-world access,
or a paper claim. V3 test panels must not be created before founder and held
compiler freeze manifests.

## Runtime contract

Olivia `accel` nodes are ARM64 NVIDIA Grace Hopper nodes. The inspected runtime
for RC0 is:

```text
image:  /cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif
sha256: a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1
cache:  /cluster/projects/nn12027k/hf-cache
work:   /cluster/work/projects/nn12027k/dakai5365/frank-eq
```

It exposes `python3`, CUDA-enabled PyTorch, Transformers, NumPy, PyYAML, and
W&B on `aarch64`. Do not use the historical AMD64 image or the existing CPython
x86_64 wheelhouse, and do not enable runtime package installation for RC0.

W&B remains fail-open telemetry. Its credential is sourced from
`$HOME/.config/codex-hpc/wandb.env` only when that file has mode `0600`; the
credential itself must never enter the repository, source archive, or Slurm
submission record.

## Checkpoint preflight

All exact revisions must be present in the shared offline cache:

```text
Qwen/Qwen3-4B   1cfa9a7208912126459214e8b04321603b3df60c
Qwen/Qwen3-8B   b968826d9c46dd6066d109eabc6255188de91218
Qwen/Qwen3-14B  40c069824f4251a91eefaf281ebe4c544efd3e18
```

The 14B snapshot may be downloaded, hashed, and loaded for a tokenizer/model
engineering check without any graph prompt. That does not expose the task. It
must not see a registered train prefix until the workflow has frozen both
founder checkpoints, and it must not see test data before its own freeze. No
unpinned head or network fallback is allowed during the outcome run.

## V3 implementation and plan freeze

Run the full local contract and commit the complete implementation. Only then
generate the outcome-blind plan; this command does not create a panel or load a
model:

```bash
python -m frank_eq.cli plan-stagea-v3 \
  --config configs/stagea_v3/real_olivia_v3.yaml \
  --out configs/stagea_v3/inspected_plan.json
```

Inspect its config hash, all implementation hashes, exact model revisions,
stage order, expected `1,824` prefix forwards and `213,408` logical source
queries, and the explicit false values for held-task opening and test-panel
instantiation. Commit this plan separately. Any later change to the bound code,
launcher, protocol, registration, or config invalidates it.

## V3 dry run and submission

The repository launcher creates a deterministic source archive under
`.agents/state/`, deploys it into a fresh immutable Olivia job root, and uses
`olivia/stagea_v3.slurm`. The job requests one GH200, 32 CPUs, 192 GiB host
memory, and Olivia's seven-day maximum walltime.

```bash
python olivia/cli.py submit \
  --job-name <fresh-v3-job-name> \
  --config configs/stagea_v3/real_olivia_v3.yaml \
  --profile full \
  --stages prepare,founder_fit,freeze,held_onboard,evaluate \
  --dry-run --json
```

Require a clean Git tree. Inspect the deterministic source archive SHA-256,
config and plan hashes, exact runtime image hash, resources, remote paths, and
stage sequence. Submit by removing only `--dry-run`; do not modify the tree in
between.

## Status, fetch, and verify

```bash
python olivia/cli.py status --job-name <job> --json
python olivia/cli.py fetch  --job-name <job> --json
python olivia/cli.py verify --job-name <job> --json
python scripts/verify_stagea_v3_run.py \
  --config configs/stagea_v3/real_olivia_v3.yaml \
  --run .agents/state/olivia/<job>/remote/runs
```

The fetch includes the large generated captures and checkpoints so the local
verifier can validate every artifact hash. Never commit them. Adopt only a
compact hash-verified evidence package after the machine decision, every
model/complexity/renderer/family stratum, rate/compute tables, access ledger,
and independent recomputation have been audited.

Slurm `COMPLETED` means engineering completion, not scientific success. A
well-formed negative machine decision is a valid terminal v3-2 result. An
engineering failure before test access may be repaired under a new immutable
source and job while preserving the failure. Once the ledger consumes test
access, v3-2 cannot be retried or recovered under the same registration.
