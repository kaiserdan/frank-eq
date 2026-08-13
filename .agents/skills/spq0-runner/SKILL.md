---
name: spq0-runner
description: Validate, dry-run, submit only with separate authority, monitor, fetch, and independently verify the development-only Frank-EQ Shared Predictive Quotient census on Olivia.
---

# SPQ0 runner

Read `AGENTS.md`, `HANDOFF.md`,
`docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md`, and
`docs/26_SPQ0_OLIVIA_RUNBOOK.md` completely before acting.

## Authority

SPQ0 is prospective and development-only. Implementation, local validation,
and content-addressed dry runs are authorized. Do not submit automatically.
Submission requires separate operator authority and permits exactly:

```text
configs/spq0/real_olivia_spq0.yaml
--profile full --stages audit
```

There is no recovery path, test role, held sender, receiver, or claim-bearing
role. A pass authorizes only an SPQ1 protocol draft, never its execution.

## Non-access contract

Only exact Qwen3-4B and Mistral-7B-v0.3 founders are active. OLMo2 and Granite
are reserved unopened. Never resolve their snapshots, list/open files,
instantiate adapters, load weights, tokenize a task, or run inference. Require
the zero-access receipt and independent verification.

## Preflight

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

Require a clean commit, deterministic inspected-plan equality, exact runtime
image hash, exact active revisions and snapshot file hashes, and unchanged
historical evidence before model inference.

## Dry run

```bash
python olivia/cli.py submit \
  --job-name frank-eq-spq0-olivia-20260813a \
  --config configs/spq0/real_olivia_spq0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

Repeat and compare the complete JSON. Record source/config/plan/image hashes,
active revision-registry hash, and reserved non-access-contract hash. Do not
remove `--dry-run` without separate launch authority.

## Eventual monitoring and verification

```bash
python olivia/cli.py status --job-name <job> --json
python olivia/cli.py fetch  --job-name <job> --json
python olivia/cli.py verify --job-name <job> --json
python scripts/verify_spq0_run.py \
  --config configs/spq0/real_olivia_spq0.yaml \
  --run .agents/state/olivia/<job>/remote/runs
```

Require categorical forecasting, exact prefix continuity, exclusive cloned KV,
zero replay, separated roles, validation-only system/renderer/length, all
surface and token controls, frozen target readers, both ordered cross-family
pairs, zero pair mappers, the complete rank/rate census, non-promotional
residual reporting, grouped metrics, exact recomputation, and closed protected
authorizations.

Never commit generated runs, caches, checkpoints, `.agents/state/`, secrets, or
W&B credentials. Adopt evidence only through a separate compact,
hash-verified review.
