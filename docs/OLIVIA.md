# Olivia runbook

## Current authority

Stage R / RC0 completed on Olivia and is adopted as a development pass:

```text
capture:  frank-eq-rc0-rate-compute-olivia-20260811c  Slurm 1874736
recovery: frank-eq-rc0-rate-compute-olivia-20260811d-recovery  Slurm 1891471
result:   PUBLIC_BASIS_COMPOSITION_SUPPORTED
```

Stage-A v3-2 then completed on Olivia:

```text
job:       frank-eq-stagea-v3-2-olivia-20260812b  Slurm 1899057
result:    ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED
evidence:  evidence/real_stagea_v3_olivia/
```

The v3 registration is consumed. Do not submit, recover, or tune it again.
Stage M0 has also completed:

```text
job:       frank-eq-moment-compute-m0  Slurm 1970800
state:     COMPLETED 0:0
decision:  OPERATION_CLOSED_EVENTS_NOT_READABLE
evidence:  evidence/real_stage_m_olivia_m0/
```

Stage M0 has no held sender, test role, receiver access, or claim authority. Its
negative closes the current graph/source line. No RC0 or M0 rerun, successor
graph compiler, receiver protocol, receiver execution, receiver-world access,
scientific claim, or paper claim is authorized.

SPQ0 is a fresh prospective development census. Its implementation may be
validated and dry-run but not launched automatically. A separately authorized
future launch is restricted to `configs/spq0/real_olivia_spq0.yaml`, profile
`full`, and stage `audit`. See `docs/26_SPQ0_OLIVIA_RUNBOOK.md`; its rules
supersede this historical runbook for SPQ0.

## Runtime contract

Olivia `accel` nodes are ARM64 NVIDIA Grace Hopper nodes. The inspected Frank-EQ
runtime is:

```text
image:  /cluster/projects/nn12027k/frank/scratch_pytorch_gcc_updated.sif
sha256: a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1
cache:  /cluster/projects/nn12027k/hf-cache
work:   /cluster/work/projects/nn12027k/dakai5365/frank-eq
```

It exposes `python3`, CUDA-enabled PyTorch, Transformers, NumPy, PyYAML, and
W&B on `aarch64`. Do not use the historical AMD64 image or the existing CPython
x86_64 wheelhouse, and do not enable runtime package installation for Stage M0.

W&B remains fail-open telemetry. Its credential is sourced from
`$HOME/.config/codex-hpc/wandb.env` only when that file has mode `0600`; the
credential itself must never enter the repository, source archive, or Slurm
submission record.

## Historical Stage M0 execution record

The full local contract passed from clean commit `d4e64bb`. Static validation
reported 64 worlds, 32 operations, 318 event coordinates, the two exact founder
revisions, zero executor mismatches, and closed protected authorizations.

The historical dry-run command was:

```bash
export FRANK_EQ_IMAGE_SHA256=a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1
python olivia/cli.py submit \
  --job-name frank-eq-moment-compute-m0 \
  --config configs/moment_compute/real_olivia_m0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

The deterministic source/config hashes, exact revisions, non-null ARM64 image
hash, fresh remote root, and shared cache were inspected before the sole
submission. The job used `olivia/run.slurm`, one H200, 32 CPUs, 128 GiB, and a
12-hour limit. These details are provenance, not resubmission authority.

The completed job was fetched and verified with:

```bash
python olivia/cli.py status --job-name frank-eq-moment-compute-m0 --json
python olivia/cli.py fetch  --job-name frank-eq-moment-compute-m0 --json
python olivia/cli.py verify --job-name frank-eq-moment-compute-m0 --json
python scripts/verify_moment_compute_run.py \
  --run .agents/state/olivia/frank-eq-moment-compute-m0/remote/runs
```

Stage M0 finished scheduler-successfully despite its scientific gate miss. The
fetched tree, repository verifier, and NumPy-2.2.6 exact-runtime verifier pass;
compact evidence is adopted. Do not submit the historical command again.

## Preserved V3 checkpoint preflight

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

Stage the missing held snapshot through an immutable generic deployment and the
task-blind `olivia/stagea_v3_cache_held.slurm` job. The receipt must report the
exact revision, zero task prompts, no model inference, no broken files, and
content hashes for every resolved snapshot file. This cache-only job is not the
v3 outcome run.

Before the outcome dry run, use `olivia/stagea_v3_held_smoke.slurm` from a fresh
immutable generic deployment. It may load the exact held checkpoint and perform
one neutral prefix forward, KV clone, and tokenizer-only chat-prefix continuity
check. Its receipt must report zero registered worlds, operations, answers, and
test accesses. It must not generate text or score a task response.

## Preserved V3 implementation and plan freeze

The outcome was launched from a separately committed implementation and
outcome-blind plan. This historical command did not create a panel or load a
model:

```bash
python -m frank_eq.cli plan-stagea-v3 \
  --config configs/stagea_v3/real_olivia_v3.yaml \
  --out configs/stagea_v3/inspected_plan.json
```

The final plan binds config `92d7ede...f5b3`, internal plan
`7b509858...a113c`, plan file `9a728f19...0cbf8`, implementation tree
`bf8c87fa...812fc`, 1,824 prefix forwards, 213,408 logical source queries,
and explicit false values for held-task opening and test-panel instantiation.

## Frozen V3 submission record

The repository launcher created source archive `b6203d03...a44e0` under ignored
`.agents/state/`, deployed it into a fresh immutable Olivia job root, and used
`olivia/stagea_v3.slurm`. Job `1899057` requested one GH200, 32 CPUs, 192 GiB
host memory, and seven days. It ran on `gpu-1-85` for 12:31:24.

The command is intentionally omitted here now that the one registered launch is
consumed. Repository launcher support remains for provenance and tests, not as
authorization to submit v3-2 again.

## Preserved V3 status, fetch, and verify

```bash
python olivia/cli.py status --job-name frank-eq-stagea-v3-2-olivia-20260812b --json
python olivia/cli.py fetch  --job-name frank-eq-stagea-v3-2-olivia-20260812b --json
python olivia/cli.py verify --job-name frank-eq-stagea-v3-2-olivia-20260812b --json
python scripts/verify_stagea_v3_run.py \
  --config configs/stagea_v3/real_olivia_v3.yaml \
  --run .agents/state/olivia/frank-eq-stagea-v3-2-olivia-20260812b/remote/runs
```

The fetch includes the large generated captures and checkpoints so the local
verifier can validate every artifact hash. Never commit them. Adopt only a
compact hash-verified evidence package after the machine decision, every
model/complexity/renderer/family stratum, rate/compute tables, access ledger,
and independent recomputation have been audited.

Both verification commands return nonzero for this preserved outcome. The
runner reports terminal workflow state `failed`; the specialized verifier
reproduces the original order-sensitive metric refusal. The adopted audit
explains why the machine decision is nevertheless an integrity-valid negative.
Do not reinterpret either nonzero exit as permission to repair or rerun v3-2.
