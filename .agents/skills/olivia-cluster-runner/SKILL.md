---
name: olivia-cluster-runner
description: Package, submit, monitor, fetch, and verify Frank-EQ jobs on Olivia, including the development-only RC0 audit.
---

# Olivia cluster runner

Use `olivia/cli.py`; do not create mutable remote worktrees or ad-hoc copy
procedures. Read `AGENTS.md`, `HANDOFF.md`, `docs/OLIVIA.md`, and the selected
protocol first.

## Current authorized run

```text
config: configs/stagea_v3/real_olivia_v3.yaml
protocol: stagea-v3-1
role: outcome-bearing representation qualification
```

The user authorized exactly one sequential representation run on 2026-08-12,
but only after the frozen registration and implementation are separately
committed, all validators pass, and the content-addressed dry run is inspected.
The repository launcher does not support v3 until that implementation lands.
Do not improvise an ad-hoc command. RC0 is complete and must not be rerun.

Local validation:

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
```

Submit only after checking the deterministic source SHA-256 and confirming all
three exact model revisions are available on Olivia. Qwen3-14B remains
task-unopened during staging and engineering smoke tests.

## Operator commands

```bash
python olivia/cli.py status --job-name frank-eq-rc0-rate-compute --json
python olivia/cli.py fetch  --job-name frank-eq-rc0-rate-compute --json
python olivia/cli.py verify --job-name frank-eq-rc0-rate-compute --json
python scripts/verify_rate_compute_run.py --run <fetched-run-root>
```

## Invariants

- Stages and their order must match the frozen v3 registration.
- No model revision substitution or network-resolved unpinned head.
- Corrected `chat_turn`, exclusive KV reuse, no replay fallback.
- Test panels do not exist before founder and held freeze manifests.
- The primary compiler makes zero post-capture source queries.
- Semantic and behavioral channels remain separate.
- Both complexity panels, renderer roles, and all registered baselines complete.
- Scheduler completion and scientific promotion remain separate.
- Generated caches, responses, source archives, and `.agents/state/` stay out of
  Git.
- RC0 development worlds and exposed models cannot become later confirmation or
  held roles.

Do not launch receiver execution, rerun RC0, or run another Stage-Q scale screen.
See `docs/20_STAGEA_V3_PROTOCOL.md` and `references/contract.md` for the current
scientific and failure contract.
