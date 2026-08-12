# PSR0 Olivia runbook

Status: ready for one development-only Olivia audit after branch review. The run
contains no held sender, claim-bearing test split, or receiver stage.

Read first:

```text
docs/22_PREDICTIVE_STATE_PSR0.md
HANDOFF.md
AGENTS.md
```

## 1. Local contract validation

```bash
python -m compileall -q src scripts olivia
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
python scripts/validate_predictive_state.py
```

Verify that regenerating the plan changes nothing:

```bash
python scripts/predictive_state_cli.py plan \
  --config configs/predictive_state/real_olivia_psr0.yaml \
  --out /tmp/psr0-plan.json
cmp /tmp/psr0-plan.json configs/predictive_state/inspected_plan.json
```

## 2. Frozen checkpoint preflight

Olivia must already contain these exact snapshots:

```text
Qwen/Qwen3-4B
1cfa9a7208912126459214e8b04321603b3df60c

Qwen/Qwen3-8B
b968826d9c46dd6066d109eabc6255188de91218
```

`local_files_only: true` must remain set. Do not resolve a moving model head or
substitute a checkpoint.

## 3. Content-addressed dry run

```bash
python olivia/cli.py submit \
  --job-name frank-eq-psr0-olivia-20260812a \
  --config configs/predictive_state/real_olivia_psr0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

Inspect and record:

- source archive SHA-256;
- Git commit and clean-tree status;
- config path and config SHA-256;
- internal plan SHA-256
  `a25e80cc4a5b11adf869f11196394a93bf647915807803fea2df12434bd2c27f`;
- one-GPU full profile;
- output root;
- stages exactly `audit`.

Do not submit a source package different from the inspected dry run.

## 4. Submit

```bash
python olivia/cli.py submit \
  --job-name frank-eq-psr0-olivia-20260812a \
  --config configs/predictive_state/real_olivia_psr0.yaml \
  --profile full \
  --stages audit \
  --json
```

The quickstart validates the config, checks the committed inspected plan, runs
the audit, and independently recomputes the probes, predictions, metrics, and
decision.

## 5. Monitor, fetch, and verify

```bash
python olivia/cli.py status \
  --job-name frank-eq-psr0-olivia-20260812a \
  --json

python olivia/cli.py fetch \
  --job-name frank-eq-psr0-olivia-20260812a \
  --json

python olivia/cli.py verify \
  --job-name frank-eq-psr0-olivia-20260812a \
  --json
```

Run the PSR0 verifier against the fetched directory containing
`run_manifest.json`:

```bash
python scripts/predictive_state_cli.py verify \
  --config configs/predictive_state/real_olivia_psr0.yaml \
  --run <fetched-run-root>
```

A scientific gate failure is a valid completed workflow. Scheduler success,
engineering verification, and scientific promotion are separate.

## 6. Expected scale

Per model:

```text
query-blind prefixes:        1,088
future tests per prefix:        22
KV response branches:       23,936
```

The 4B and 8B checkpoints are loaded sequentially. Query branches are grouped by
token length and executed in batches of eight. Runtime basis interrogation is
recorded as development tomography and cannot be counted as a one-shot packet.

## 7. Required artifacts

```text
config.yaml
dry_run_plan.json
run_manifest.json
workflow_status.json
automaton.json
public_basis.json
panels/train.json
panels/validation.json
capture_manifest.json
captures/*.npz
captures/*.json
probe_training.json
predictions_manifest.json
predictions/*.npz
metrics.json
decision.json
run_summary.json
artifact_manifest.json
```

The verifier checks artifact hashes, frozen revisions, exact prefix counts,
train/validation-only roles, plan identity, regenerated panels, public-basis
factorization, refitted probes, prediction arrays, metrics, decision, and closed
authorization fields.

## 8. Reading the decision

```text
PREDICTIVE_BASIS_OR_EXECUTOR_INVALID
ACTIVATION_PREDICTIVE_STATE_NOT_READABLE
NO_ACTIVATION_SPECIFIC_PREDICTIVE_STATE_ADVANTAGE
PREDICTIVE_STATE_NOT_RENDERER_INVARIANT
PREDICTIVE_STATE_NOT_LENGTH_TRANSFERABLE
PUBLIC_PREDICTIVE_STATE_NOT_COMPOSITIONALLY_USEFUL
PUBLIC_PREDICTIVE_STATE_CANDIDATE_SUPPORTED
```

Only the last diagnosis permits drafting one fresh PSR Stage 1 registration. It
does not permit executing it.

## 9. Evidence adoption

Generated captures remain outside Git until reviewed. Adopt only a compact,
hash-verified package containing:

```text
frozen config and plan
run/workflow manifests
verification summary
metrics.json
decision.json
independent scientific audit
content hash manifest
```

Do not commit model snapshots, raw capture arrays, source archives, scheduler
state, W&B credentials, or `.agents/state/`.
