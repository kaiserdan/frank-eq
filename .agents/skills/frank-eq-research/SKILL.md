---
name: frank-eq-research
description: Implement, audit, and extend Frank-EQ future-defined operational equivalence quotient experiments while preserving causal boundaries and fail-closed gates.
---

# Frank-EQ research skill

## Use this skill when

- implementing a new future-operation family;
- adding a real-model state-capture backend;
- training or evaluating model-local quotient charts;
- auditing split, temporal, packet, or held-sender boundaries;
- preparing a real-model canary or independent review package.

## Mandatory workflow

1. Read `AGENTS.md` and `HANDOFF.md`.
2. Identify the exact gate and information role affected.
3. Check `docs/10_DECISION_LOG.md` for frozen decisions.
4. Add tests before changing claim-bearing behavior.
5. Keep state formation before operation reveal.
6. Keep all views/branches of a world in one split.
7. Preserve the public decoder during held-sender onboarding.
8. Run:

```bash
python -m compileall -q src scripts
pytest -q
python scripts/validate_repo.py
```

9. Run the smoke configuration.
10. If metrics, objectives, packet fields, or gates changed, rerun the full reference and update the evidence copy.
11. Append the decision log; never rewrite a prior outcome.

## Scientific review checklist

- Is the cached state operation agnostic?
- Are public coordinates externally defined?
- Could a learned decoder absorb model-specific gauge?
- Are held-out operations genuinely absent from training labels?
- Are renderer variants grouped by world?
- Is facts-only compared at the same public decoder and rate?
- Is the held sender the only component updated during onboarding?
- Is target-private information absent from the strict path?
- Are confidence intervals grouped by world?
- Does the decision artifact fail closed on missing metrics?

## Output standard

Every completed experiment should produce:

```text
resolved_config.yaml or embedded config
split/operation manifest
training summary and history
held-out metrics
machine decision
artifact SHA-256 manifest
run record with code/source identity
```

Do not use W&B state or a training loss as an authorization artifact.
