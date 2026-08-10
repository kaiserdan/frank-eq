# Contributing

Changes must preserve the scientific invariants in `AGENTS.md`.

Before submitting changes:

```bash
python -m compileall -q src scripts
pytest -q
python scripts/validate_repo.py
```

Protocol or gate changes require an append-only entry in `docs/10_DECISION_LOG.md`. Never modify a frozen threshold after viewing an outcome without creating a new versioned protocol and explicitly labeling it post-outcome.

Do not commit checkpoints, generated caches, private prompts, model outputs, or credentials.
