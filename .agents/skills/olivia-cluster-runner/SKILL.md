---
name: olivia-cluster-runner
description: Package, submit, monitor, fetch, and verify Frank-EQ jobs on Olivia, including the development-only RC0 audit.
---

# Olivia cluster runner

Use `olivia/cli.py`; do not create mutable remote worktrees or ad-hoc copy
procedures. Read `AGENTS.md`, `HANDOFF.md`, `docs/OLIVIA.md`, and the selected
protocol first.

## Current authority

```text
completed job: frank-eq-stagea-v3-2-olivia-20260812b / Slurm 1899057
decision: ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED
evidence: evidence/real_stagea_v3_olivia/
next authorized execution: none
```

The user's one-run authorization from 2026-08-12 is exhausted. The complete
workflow reached its frozen negative decision, then failed closed because the
independent verifier used a different bundle-reduction order and required
bit-exact metrics. The exact Olivia runtime diagnostic reproduced the stored
metrics in config order and found only 46 lexicographic-order roundoff changes,
bounded by `5.551115123125783e-17`, with no decision change.

Do not resubmit, repair, recover, or tune v3-2. RC0 must not be rerun. Receiver
work, new receiver-world access, and all claim fields remain unauthorized.

Local validation:

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
```

## Historical-run inspection commands

```bash
python olivia/cli.py status --job-name frank-eq-stagea-v3-2-olivia-20260812b --json
python olivia/cli.py fetch  --job-name frank-eq-stagea-v3-2-olivia-20260812b --json
python olivia/cli.py verify --job-name frank-eq-stagea-v3-2-olivia-20260812b --json
python scripts/verify_stagea_v3_run.py \
  --config configs/stagea_v3/real_olivia_v3.yaml \
  --run .agents/state/olivia/frank-eq-stagea-v3-2-olivia-20260812b/remote/runs
```

The last two verifiers are expected to return nonzero for the preserved
fail-closed workflow state. Diagnose from the adopted evidence; do not reinterpret
that status as permission to rerun.

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
- The v3-2 test role and held Qwen3-14B checkpoint are task-exposed and cannot be
  reused as unopened evidence.

Do not launch receiver execution, rerun RC0, run another Stage-Q scale screen,
or resubmit v3-2. See `AGENTS.md`, `HANDOFF.md`, and
`evidence/real_stagea_v3_olivia/AUDIT.md` for the current scientific and failure
contract; `docs/20_STAGEA_V3_PROTOCOL.md` remains the immutable historical
registration.
