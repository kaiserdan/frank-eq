# Frank-EQ

**Future-defined operational equivalence quotients for cross-model LLM state interfaces.**

Frank-EQ asks whether a pre-operation LLM state can be compressed into a
public, model-independent description of the future computations it supports.
It does not treat hidden-coordinate similarity as interoperability.

For frozen model `M`, cached state `h`, and public operation family `K`:

```text
Sigma_M(h) = { p_M(y | h, k) : k in K }.
```

## Current status

### Synthetic Stage 0

The controlled implementation reference passes and authorizes only a real-model
canary. It is not evidence about LLM latent spaces.

### Real Stage A v1: valid negative

The first frozen real run, `frank-eq-stagea-devg-v2` on LUMI job `20942127`,
completed with full engineering integrity and returned:

```text
STOP_OR_REVISE_STAGE0
```

The exact failed pipeline was:

```text
raw non-chat world prefix
→ final-token residual at normalized depths 0.35/0.60/0.85
→ model-local chart
→ shared fact/residual heads
→ frozen graph interrogator
```

Six gates failed: fact accuracy, held-out signature Brier, cross-model
retrieval, wrong-world specificity, held-sender retention, and model-ID
leakage. The trained fact accuracy was `0.5296`, only `0.0019` above the
reconstructed global-majority baseline. Renderer cosine and quantization
passed, but renderer stability coexisted with code collapse/model identity and
does not rescue the result.

The decision is retained as a valid negative for Stage-A v1. The broader
diagnosis is **not yet identified**. V1 did not independently test whether the
source information was readable before applying its nonlinear, multi-objective,
shared-head quotient.

See:

- `evidence/real_stagea_devg_v2/`
- `docs/12_STAGEA_V1_AUDIT_AND_V2_PROTOCOL.md`
- `docs/10_DECISION_LOG.md`

## Immediate next action

Run the non-promotional train/validation-only localization on the fetched v1
cache:

```bash
frank-eq diagnose-real-cache \
  --cache <fetched-run>/runs/cache \
  --out <fetched-run>/runs/diagnostics
```

It separately measures:

- formal fact readability;
- oracle-signature readability;
- readability of each model's own future branch signature;
- renderer-transfer readability;
- native branch competence against operation-wise priors;
- individual-layer versus concatenated capture.

It consumes zero test worlds and cannot authorize another run. Its result
selects which Stage-A v2 hypothesis to freeze:

```text
native incompetence
  → repair task/prompt competence first

own future signature unreadable
  → expand capture to token-sequence / selected-KV state

own future signature readable but facts unreadable
  → separate operational state from semantic grounding

raw targets readable but quotient fails
  → complete model-local compilers and revise the joint objective
```

## Architecture

```text
frozen source state before operation reveal
        |
        v
model-local causal-state capture
        |
        v
complete model-local compiler
        |
        +---- self-future operational signature
        |
        +---- externally grounded facts / calibration
        |
        v
typed public state
        |
        v
frozen public interrogator
        |
        v
receiver-native execution (locked)
```

The repository preserves the v1 shared-head architecture by default. A dormant
v2 option, `model.public_head_scope: local`, gives every sender its own chart,
fact head, and residual head while keeping public semantics and the
interrogator shared. It must not be launched until a versioned v2 protocol and
fresh test role are frozen.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

frank-eq validate-config --config configs/stage0/synthetic_smoke.yaml
frank-eq run-stage0 \
  --config configs/stage0/synthetic_smoke.yaml \
  --out runs/synthetic-smoke
```

Real extraction additionally requires:

```bash
pip install -e '.[real,dev]'
frank-eq validate-real-config --config configs/stage0/real_smoke.yaml
```

Available real commands:

```text
frank-eq make-real-cache
frank-eq validate-real-cache
frank-eq diagnose-real-cache
frank-eq run-real-stagea
```

The workflow stage order is:

```text
cache,validate,diagnose,train,eval
```

`diagnose` is optional and non-promotional. Historical
`cache,validate,train,eval` commands remain valid.

## Olivia

```bash
python olivia/cli.py submit \
  --job-name frank-eq-stagea-localization \
  --config configs/stage0/real_olivia.yaml \
  --profile full \
  --stages cache,validate,diagnose \
  --dry-run --json
```

## LUMI

```bash
python lumi/cli.py submit \
  --job-name frank-eq-stagea-localization-lumi \
  --config configs/stage0/real_lumi.yaml \
  --profile full \
  --stages cache,validate,diagnose \
  --dry-run --json
```

For the existing v1 cache, prefer the standalone diagnostic rather than
recapturing data.

## Evidence hierarchy

```text
frozen config/source identity
cache validation and access audit
machine decision
held-out metrics
predictions
training history
non-promotional localization
W&B telemetry
prose
```

Operational `.agents/state/` caches and source archives are local-only and must
not be committed. Adopted evidence belongs under `evidence/` with a hash
manifest.

## Validation

```bash
python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
```

## Claim boundary

Frank-EQ currently establishes:

- a functioning synthetic quotient implementation;
- a causally ordered, reproducible real Stage-A workflow;
- a valid negative result for one narrow real architecture.

It does **not** establish:

- that future-defined quotients are absent from LLMs;
- that such a quotient exists across model families;
- receiver-native execution;
- held-sender establishment;
- superiority to text communication;
- safety or causal utility.

Receiver work remains locked.
