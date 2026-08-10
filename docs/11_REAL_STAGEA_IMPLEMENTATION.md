# Real-checkpoint Stage-A implementation

## Scientific question

Does a query-blind hidden state from independently trained frozen LLMs contain a compact public state that preserves held-out future operations across renderer changes and held-sender onboarding?

Stage A is representation qualification only. It does not execute the quotient in a receiver.

## Panel

Each world is a six-entity directed closed graph. The source prefix lists positive edges, declares that omitted edges are false, and provides two public global tags: density and reciprocity class. Two renderer forms express the same world as natural-language statements or an adjacency table.

The registered operation families are:

```text
lookup
inverse
mutual
compose
compare_outdegree
counterfactual_add
density
reciprocity
```

There are two instances per family. The split builder holds out an instance inside every family, so the frozen graph interrogator is evaluated on unseen operation descriptors rather than only unseen worlds.

## Capture boundary

For every model, world, and renderer:

1. tokenize the world prefix with no operation text;
2. run the frozen checkpoint with hidden-state output;
3. capture the final prefix position at normalized depths 0.35, 0.60, and 0.85;
4. hash the literal prefix and unpadded float32 hidden capture;
5. reveal each operation only after the capture step;
6. map the next-token distribution to a tokenizer-specific single-token false/true pair;
7. write a complete `FutureSignatureRecord`.

`branch_mode=auto` first clones the prefix KV cache. If a model/cache implementation cannot be cloned safely, exact prefix token IDs are replayed and the fallback is counted. Prefix text is never re-tokenized jointly with the query, avoiding BPE boundary drift.

## Public and diagnostic signatures

The cache deliberately contains two matrices:

```text
signatures        formal smoothed oracle outcomes
model_signatures  frozen source-model post-reveal probabilities
```

The public quotient is trained against formal operational semantics. `model_signatures` measures whether overt source behavior agrees with those semantics and whether the learned quotient is closer to oracle state than the source's immediate response. This prevents model disagreement from silently redefining the public interface.

## Public quotient

The model-local chart receives only selected source hidden states. It predicts:

- one probability for each canonical directed edge;
- a density coordinate;
- a reciprocity coordinate.

The shared graph interrogator has no parameters. It computes differentiable operation probabilities directly from these coordinates, including noisy-OR two-hop composition and a counterfactual edge-addition operation. Held operations are therefore executable without fitting an operation decoder.

The private chart bottleneck remains model-local. Invariance, retrieval, leakage, quantization, and packet metrics use only the public facts-plus-residual code.

## Cache authority

Training is prohibited until `cache_validation.json` confirms:

- every operation is revealed strictly after capture;
- every state covers the entire frozen operation registry;
- all views of a world share one split;
- every model × world × renderer cell exists exactly once;
- hidden bytes match their record hashes;
- branch descriptors match the frozen registry hashes;
- branch probabilities match the serialized record;
- panel, dataset, and JSONL file hashes match metadata.

The validator may authorize training but always sets `authorizes_scientific_claim=false`.

## Workflow stages

```text
cache     build panel, capture models, write bundle and records
validate  reproduce the complete cache audit
train     founder charts, then frozen-executor held-sender chart
eval      held operations, test worlds, bootstrap, packet audit, decision
```

Stages can be split across cluster jobs. A later stage requires the earlier artifacts under the same run root. Do not copy a cache from a different source/config hash without writing a new manifest and explicit provenance decision.

## Primary outputs

In addition to the existing Stage-0 metrics, real evaluation reports:

```text
source_branch_brier_to_oracle
source_branch_accuracy_to_oracle
quotient_brier_to_source_branch
```

These are diagnostic. Promotion remains conjunctive on formal held-operation fidelity, facts, renderer invariance, retrieval/specificity, residual contribution, quantization, held-sender retention, and leakage.

## Stop rule

If the frozen real Stage-A gate fails, do not add a receiver, target-state decoder, pairwise translator, or a second operation-conditioned chart. Localize the failure from the existing artifacts. A revised Stage A requires a new versioned config and a new untouched test role.
