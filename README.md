# Frank-EQ

**Future-defined operational equivalence quotients for cross-model LLM state interfaces.**

Frank-EQ asks whether a state formed before an operation is revealed can be
compiled into a public description of the future computations it supports. It
does not equate hidden-coordinate similarity with interoperability.

For frozen model `M`, cached state `h`, and operation family `K`:

```text
Sigma_M(h) = { p_M(y | h, k) : k in K }.
```

## Current status

### Synthetic Stage 0

The controlled implementation reference passes. It establishes only that the
contracts, packet, frozen decoder, held-sender workflow, and gates are
executable.

### Real Stage A: two valid negatives

```text
v1  frank-eq-stagea-devg-v2   LUMI 20942127   STOP_OR_REVISE_STAGE0
v2  frank-eq-stagea-lumi-v2   LUMI 20952565   STOP_OR_REVISE_STAGE0
```

The v2-1 shared-head oracle quotient failed decisively:

```text
native competence gain vs prior:  -0.0521
held-out signature Brier:          0.2065 (upper95 0.2421)
fact accuracy:                     0.5509 (lower95 0.5120)
cross-model retrieval:             0.1528 (lower95 0.0972)
wrong-world margin:               -0.0607
held-sender retention:            -0.3445
model-ID leakage over chance:      0.6389
```

No receiver experiment or scientific claim is authorized.

### Important v2-1 interpretation correction

The exact v2-1 negative remains valid, but the original description as a
falsification of native chat prompting was too broad.

Historical `prompt_format: chat` cached a sequence ending at an
assistant-generation header. The operation was appended after that header, so
it entered as assistant content rather than as a new user turn. In addition,
v1 and v2 used different panel seeds, changing worlds, operations, and split
assignments. Their competence point estimates were not a paired prompt
comparison and had no interval for the difference.

The claim supported by v2-1 is therefore:

> The exact legacy chat-assistant-continuation, final-token, shared-head oracle
> quotient fails.

See:

```text
evidence/real_stagea_lumi_v2/REVIEW.md
docs/16_STAGEA_V2_INTERPRETATION_CORRECTION.md
docs/15_STAGEA_V2_REVIEW_AND_STAGEQ.md
```

## Current next step: Stage Q

Stage Q is a development-only source-competence qualification. It must run
before another quotient is trained or another claim-bearing test role is used.

Two frozen LUMI configs share the same models, worlds, renderers, operations,
split, and KV path:

```text
configs/stageq/real_lumi_legacy_chat.yaml
configs/stageq/real_lumi_chat_turn.yaml
```

The candidate `chat_turn` path caches a complete system/user/assistant
conversation and reveals the operation afterward as a new user message. The
backend fails closed unless the cached token IDs are an exact prefix of the full
branch conversation.

Build and validate both caches only:

```bash
python lumi/cli.py submit \
  --job-name frank-eq-stageq-legacy-chat \
  --config configs/stageq/real_lumi_legacy_chat.yaml \
  --profile full --stages cache,validate --json

python lumi/cli.py submit \
  --job-name frank-eq-stageq-chat-turn \
  --config configs/stageq/real_lumi_chat_turn.yaml \
  --profile full --stages cache,validate --json
```

Then qualify and compare them locally:

```bash
python scripts/qualify_real_cache.py \
  --cache <legacy>/runs/cache \
  --out runs/stageq/legacy-qualification

python scripts/qualify_real_cache.py \
  --cache <chat-turn>/runs/cache \
  --out runs/stageq/chat-turn-qualification

python scripts/compare_stageq_caches.py \
  --baseline-cache <legacy>/runs/cache \
  --candidate-cache <chat-turn>/runs/cache \
  --out runs/stageq/paired-comparison
```

The qualifier uses founder models and train/validation worlds only, averages
views within world, and reports a grouped 95% interval. It excludes test worlds
and the held sender. Every output keeps all scientific, receiver, test-access,
and fresh-outcome authorization fields false.

Stage Q requires:

```text
candidate competence lower95 >= 0
paired candidate-minus-legacy Brier improvement lower95 >= 0
```

A pass permits drafting one fresh Stage-A registration; it does not itself
establish a latent interface.

## Architecture direction after qualification

The current shared-head oracle quotient should not be rescued unchanged. A
future Stage-A registration, only after source competence passes, should
separate:

```text
behavioral operational channel
  predicts the frozen source model's own future response distribution

semantic grounding channel
  maps model-local state to externally defined facts and correctness
```

It should also use complete model-local compilers:

```yaml
model:
  public_head_scope: local
```

This makes the chart, fact head, and residual head local to each sender while
keeping public coordinate semantics and the interrogator frozen. The option is
implemented but dormant.

## Implemented commands

Synthetic and real workflow commands:

```text
frank-eq validate-config
frank-eq run-stage0
frank-eq validate-real-config
frank-eq make-real-cache
frank-eq validate-real-cache
frank-eq diagnose-real-cache
frank-eq run-real-stagea
```

Development qualification commands:

```text
python scripts/qualify_real_cache.py
python scripts/compare_stageq_caches.py
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[real,dev]'

python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
python scripts/validate_repo.py
```

## Evidence hierarchy

```text
frozen protocol and source identity
cache validation and access audit
machine decision
held-out metrics
predictions and training history
development-only diagnostics/qualification
W&B telemetry
prose
```

Adopted evidence belongs under `evidence/` with hashes. Generated caches,
checkpoints, scheduler state, and source archives remain local under `runs/` or
`.agents/state/` and must not be committed.

## Claim boundary

Frank-EQ currently establishes:

- a functioning synthetic implementation;
- causally ordered real-cache and cluster workflows;
- two valid negative results for narrow real architectures;
- a development-only path for correctly qualifying source competence.

It does not establish:

- the existence or non-existence of a universal operational quotient;
- native-chat incompetence in general;
- cross-family sender establishment;
- receiver-native execution;
- superiority to text communication;
- safety or causal utility.

Receiver work remains locked.
