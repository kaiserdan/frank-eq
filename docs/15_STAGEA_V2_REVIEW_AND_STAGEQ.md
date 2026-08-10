# Stage-A v2 review and Stage-Q development protocol

Status: implementation and development protocol only. No new Stage-A outcome
run, receiver experiment, or scientific claim is authorized.

## 1. Preserved v2-1 result

`frank-eq-stagea-lumi-v2` remains an engineering-valid negative for its exact
frozen pipeline. Its machine decision is `STOP_OR_REVISE_STAGE0`; all claim and
receiver authorization fields remain false.

Measured failures include:

```text
native competence gain vs prior:  -0.0521
held-out signature Brier:          0.2065 (upper95 0.2421)
fact accuracy:                     0.5509 (lower95 0.5120)
cross-model retrieval:             0.1528 (lower95 0.0972)
wrong-world margin:               -0.0607
held-sender retention:            -0.3445
model-ID leakage over chance:      0.6389
```

The shared-head, oracle-grounded quotient is therefore decisively negative.
Renderer cosine and quantization do not rescue it, and the positive residual
control remains confounded by density/reciprocity labels printed in the prefix.

## 2. Interpretation correction

The broader statement that native chat prompting, or the prompt surface in
general, was falsified is withdrawn.

### 2.1 The operation was not a new user turn

Historical `prompt_format: chat` called the model chat template with system and
user world messages and `add_generation_prompt=True`. The cached prefix ended at
the assistant-generation header. The operation string was then appended directly
to that token history, making it assistant content rather than a new user
message.

This is a well-defined and reproducible input contract, so the exact v2-1
negative is valid. It is not a normal native multi-turn chat competence test.

### 2.2 The v1/v2 comparison was not paired

Both world generation and operation generation derive from `panel.seed`, and the
split/operation holdout also uses that seed. Moving from seed 1729 to 20260810
therefore changed:

- world graphs;
- operation arguments and polarities;
- train/validation/test assignment;
- held-out operation instances.

The change from -0.0603 to -0.0521 cannot isolate prompt format and cannot be
called unchanged within noise. No paired interval was computed.

### 2.3 Competence did not function as a prerequisite

The native competence point estimate was computed during final evaluation,
after quotient training and test-world scoring. It had no grouped confidence
interval. It therefore did not stop the pipeline before test consumption, even
though the protocol described it as a prerequisite.

### 2.4 KV reuse and replay are different execution paths

The outcome cache is internally coherent because all branches used KV reuse.
The observed replay differences up to 0.1089 show that replay is not an
interchangeable fallback on the current bf16/ROCm stack. The amended 0.33
threshold is retained only as a stack alarm; it is not a parity claim.

The supplemental immutable review is under
`evidence/real_stagea_lumi_v2/REVIEW.md` and `review.json`.

## 3. Scientific reading

Two conclusions survive:

1. The current shared-head oracle quotient should not continue to receiver
   execution.
2. Some development probes can predict a model's own future branch behavior
   better than they recover external graph facts. This motivates separating:

```text
behavioral operational state
  what the frozen source will do under future operations

semantic grounding
  whether that state corresponds to the correct external world
```

Before any latent architecture is revised, the source competence prerequisite
must be tested correctly.

## 4. Stage Q: development-only native-competence qualification

Stage Q uses no claim-bearing test result. It compares two cache contracts on
exactly the same models, worlds, renderers, operation registry, split, and KV
branch path.

Frozen development configs:

```text
configs/stageq/real_lumi_legacy_chat.yaml
configs/stageq/real_lumi_chat_turn.yaml
```

Both use panel seed `20260811`. These worlds are development-only and may never
be promoted into a later Stage-A confirmation role.

### 4.1 Baseline

`legacy_chat` exactly reproduces the v2-1 conversational construction: the
cached prefix ends at an assistant-generation header and the operation suffix is
assistant content.

### 4.2 Candidate

`chat_turn` caches a complete conversation:

```text
system: fixed reasoning contract
user:   query-blind world statement
assistant: fixed acknowledgement
```

After capture, each operation is revealed as:

```text
user: operation question
assistant: <generation boundary>
```

The backend renders the entire candidate conversation, tokenizes it, and fails
closed unless the cached prefix token IDs are an exact prefix of the full
conversation. The operation can therefore enter only after state formation.

`chat_template_kwargs.enable_thinking: false` is frozen for both development
conditions so model-specific hidden reasoning modes do not differ between them.

### 4.3 Primary competence statistic

For founder models and validation worlds only, on held-out operation instances:

```text
gain = Brier(oracle, operation-wise training prior)
     - Brier(oracle, frozen source branch)
```

Views are averaged within world before a 2,000-replicate grouped bootstrap.
Stage Q requires:

```text
candidate competence lower95 >= 0
paired candidate-minus-baseline Brier improvement lower95 >= 0
```

Model- and operation-family intervals are mandatory diagnostics. The held sender
and every test world are excluded.

A Stage-Q pass permits drafting one fresh Stage-A registration. It does not
permit receiver execution or a scientific claim.

## 5. Execution

Build and validate the two caches only:

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

Fetch and verify both jobs, then run:

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

Do not run `train` or `eval` for these configs.

## 6. Decision after Stage Q

### Candidate fails competence

Do not modify the latent architecture. Screen stronger source checkpoints or a
simpler formal task using development-only competence caches. Freeze the first
combination whose lower confidence bound is non-negative before Stage-A
representation training.

### Candidate passes competence but not paired improvement

The models are competent, but the effect cannot be attributed to the corrected
chat turn. Use the competent condition as a source prerequisite only; make no
prompt-mechanism claim.

### Candidate passes both gates

Freeze one Stage-A v3 registration with:

- fresh worlds unavailable to Stage Q;
- complete model-local compilers (`public_head_scope: local`);
- self-future prediction as a separately named behavioral channel;
- oracle facts as a separately named semantic-grounding channel;
- a development-selected, frozen capture representation;
- no receiver execution until the representation gate passes.

## 7. Prohibited actions

- Reinterpreting v2-1 as a native-chat falsification.
- Comparing v1 and v2 point estimates as a paired prompt experiment.
- Running quotient training or test evaluation before Stage-Q qualification.
- Tuning thresholds, operations, checkpoints, or prompts on a fresh Stage-A
  test split.
- Resuming the shared-head oracle quotient unchanged.
- Starting receiver-native execution.
