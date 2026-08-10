---
name: olivia-cluster-runner
description: Package, submit, monitor, fetch, and verify Frank-EQ Stage-A jobs on the UiT Olivia cluster.
---

# Olivia cluster runner

Use this skill for Frank-EQ jobs on Olivia. The operator surface is `olivia/cli.py`; do not create ad-hoc `scp` or mutable remote worktrees.

## Contract

1. Read `AGENTS.md`, `HANDOFF.md`, `docs/11_REAL_STAGEA_IMPLEMENTATION.md`, and `docs/OLIVIA.md`.
2. Validate locally:

   ```bash
   python -m compileall -q src scripts olivia
   pytest -q
   python scripts/validate_repo.py
   python -m frank_eq.cli validate-real-config --config <config>
   ```

3. Dry-run the exact submission:

   ```bash
   python olivia/cli.py submit \
     --job-name <immutable-name> \
     --config <repo-relative-config> \
     --profile smoke \
     --stages cache,validate,train,eval \
     --dry-run --json
   ```

4. Submit only after the dry-run records a deterministic source SHA-256. The client packages the repository, excluding local state and generated artifacts, and deploys it under a content-addressed source identity.
5. Monitor with `status`, then `fetch`, then `verify`. A failed scientific gate is a valid completed workflow; engineering validity and scientific promotion are separate.
6. Preserve `.agents/state/olivia/<job>/submission.json`, `last_status.json`, `fetch.json`, and `verify.json`.

## Commands

```bash
python olivia/cli.py status --job-name <name> --json
python olivia/cli.py fetch --job-name <name> --json
python olivia/cli.py verify --job-name <name> --json
```

## Safety and access

- Never put Hugging Face or W&B credentials in source, configs, Slurm files, or logs.
- Use environment variables forwarded by `olivia/run.slurm`.
- Do not reinterpret a scheduler success as a scientific pass.
- Do not run receiver execution or locked data from a Stage-A representation job.
- Do not change the operation panel, model roles, layers, or gates after inspecting test outcomes.

See `references/contract.md` for expected artifacts and failure handling.
