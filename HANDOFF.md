# Frank-EQ handoff

Snapshot: 2026-08-12

## Current authority

Synthetic Stage 0 passes as implementation evidence only.

Two real Stage-A runs are adopted exact-pipeline negatives:

```text
v1: frank-eq-stagea-devg-v2     LUMI 20942127   STOP_OR_REVISE_STAGE0
v2: frank-eq-stagea-lumi-v2     LUMI 20952565   STOP_OR_REVISE_STAGE0
```

Stage Q and the stronger-checkpoint screens are development-only negatives.
None authorizes a scientific claim, another latent training run, or receiver
execution.

Stage R / RC0 is now a completed development pass. The frozen scientific
capture and its artifact-only recovery were:

```text
capture:  frank-eq-rc0-rate-compute-olivia-20260811c  Slurm 1874736
recovery: frank-eq-rc0-rate-compute-olivia-20260811d-recovery  Slurm 1891471
```

Stage-A v3-2 is now frozen in `docs/20_STAGEA_V3_PROTOCOL.md` and
`configs/stagea_v3/real_olivia_v3.yaml`. The user explicitly authorized the
sequential next steps on 2026-08-12. Implementation and exactly one
outcome-bearing representation run may proceed only after separate protocol and
implementation commits, a green repository validation, and an inspected
content-addressed dry run.

Receiver execution, receiver-world access, and scientific-claim authorization
remain false. A v3 pass may authorize drafting a receiver protocol only.

## What the accumulated results establish

### The shared private-code path is not working

The v2 representation has high renderer cosine but negative wrong-world
specificity, low cross-model retrieval, and near-perfect model identity leakage.
The exact final-token, model-local-chart, shared-public-head architecture is
falsified.

### The source screens did not isolate state sufficiency

The Stage-Q response was the immediate probability of one A/B token after the
operation query. The branch used zero additional autoregressive scratchpad tokens.
Its Brier score therefore confounds:

- model-local answer-channel calibration;
- available post-query computation;
- information in the query-blind state.

The operation pattern is the key clue. At 8B, inverse and reciprocity pass while
mutual, composition, and out-degree comparison fail. This is consistent with a
compute/composition wall, not only absent state information.

### Earlier Frank branches point in the same direction

Across SuperFrank, strict realization, SSCT, BaryFrank, Frank-Sol, and CTSI:

- descriptive or predictable hidden structure is repeatedly positive;
- direct receiver realization and autonomous continuation repeatedly fail;
- model-local calibration is repeatedly load bearing;
- structured receiver-native computation is the strongest positive endpoint;
- learned rule/rendering layers are less reliable than grounded facts plus a
  deterministic compiler.

The next method must therefore fix the public gauge and expose downstream compute
rather than add another continuous translator.

## Current scientific hypothesis

A useful cross-model interface is a **public separating operational basis** plus
an explicit consumer-compute contract.

For an operation bank `K`, basis `B` is separating when equal basis responses
imply equality under every operation in `K`. A separating basis is a complete
invariant of operational equivalence, so every target operation factors through
it. In the controlled graph task, all non-diagonal directed edges are an exact
basis.

RC0 tested whether the frozen sources can expose that basis and whether a
parameter-free public executor can compose it into the hard structural
operations more reliably than direct model reasoning.

## RC0 experiment

Models:

```text
Qwen3-4B  revision 1cfa9a7208912126459214e8b04321603b3df60c
Qwen3-8B  revision b968826d9c46dd6066d109eabc6255188de91218
```

Development panels:

```text
entity counts:             4 and 6
worlds per complexity:     96
renderer views:            2
registered targets:        32
basis coordinates:         12 and 30 directed edges
claim-bearing test worlds: 0
held sender:               none
```

Response protocols:

```text
answer_token    historical immediate A/B readout
sequence        semantic false/true candidate likelihood
reason          32 generated scratchpad tokens + final cue
pause           32 fixed pause tokens + final cue
```

All semantic scores receive train-only Platt calibration. Basis calibration is
coordinate-specific within each model and complexity; direct target calibration
is family/protocol-specific. Negative calibration slopes are legal because a
stable answer-label inversion is local calibration, not missing information.

The public executor composes calibrated edge probabilities into lookup, inverse,
mutual, two-hop composition, out-degree comparison, and counterfactual
composition. Exact binary basis inputs are required to reproduce the formal
oracle.

## RC0 result

The original job completed both pinned models and all 89,856 raw and calibrated
response rows before a mechanical aggregation API regression failed the first
metric. The recovery restored the historical two-output helper contract, bound
every reused artifact by SHA-256, copied rather than modified the capture, and
executed no model inference. The original failed job remains immutable.

The machine diagnosis is `PUBLIC_BASIS_COMPOSITION_SUPPORTED`, independently
recomputed with no failures and zero hard-oracle executor mismatches. Across
3,712 hard-family validation predictions, compiled Brier is `0.0408`, versus
`0.2035` for the training-selected direct baseline and `0.2181` for the prior.
The lower-95 gains are `0.1542` and `0.1661`; every model/complexity and hard
family stratum is positive over both baselines.

Semantic sequence likelihood improves over the historical answer-token channel.
Generated reasoning is worse than the matched fixed-pause condition: the
reasoning-minus-pause interval is `[-0.00540, -0.00029]`. Preserve this negative
diagnostic.

## Promotion gate

RC0 supports drafting a Stage-A v3 protocol only if:

1. every model x complexity basis group has lower-95 Brier gain at least zero;
2. every basis group has balanced accuracy at least 0.60;
3. compiled hard operations beat the train-world operation prior;
4. compiled hard operations beat the training-selected direct response protocol
   for the aggregate and every model x complexity group.

Answer-channel and reasoning-over-pause effects are diagnostic only.

A pass authorizes protocol drafting, not execution.

## Next action

The frozen v3-2 implementation is complete without changing its scientific
fields. It includes fresh panels, typed coordinates, independent semantic and
behavioral compilers, process-locked one-time test access, all-token capture,
cloned-KV teachers, world-balanced fitting, every registered control, packet
and executor records, grouped gates, protected decisions, measured and
estimated rate/compute accounting, hash-bound founder/held freezes, an
independent recomputation verifier, CLI, and the Olivia entrypoint.

Run the full local contract and commit the green implementation. Then generate,
inspect, and separately commit `configs/stagea_v3/inspected_plan.json`, stage the
exact unopened Qwen3-14B revision without task queries, inspect the repository
launcher dry run, and submit exactly one full v3-2 workflow. Do not instantiate
the registered test panel locally.

Those pre-launch freezes are now complete: implementation commit `e75952a`,
archive-hygiene repair `29e2cc3`, Python 3.10 compatibility repair `f91f9d3`,
and superseding inspected-plan commit `a9a6b74`. The current plan has internal
hash `694408c6...505922`, plan-file hash `3f4f5740...af0b7`, and
implementation-tree hash `2fe48197...1014` across 64 bound files. It preserves
1,824 prefix forwards, 213,408 logical source queries, zero pre-run test access,
and `held_model_task_opened=false`.

Task-blind Qwen3-14B staging job `1895307` verified 18 hashed files, eight
weight shards, zero broken files, zero prompts, and no model inference. The
subsequent neutral-prefix runtime smoke `1895366` loaded the exact offline held
revision, produced a 5,120-wide hidden state, cloned its KV cache, and verified
chat-prefix continuity. Its receipt records zero registered worlds, operations,
answers, and test accesses.

The first v3-2 execution attempt, immutable Olivia Slurm job `1895410` under job
name `frank-eq-stagea-v3-2-olivia-20260812a`, failed during `founder_fit` after
computing all 160 Qwen3-4B train/n4 rows. An extreme semantic-sequence score
rounded to probability `1.0` when stored as float32, and the shard correctly
failed its existing open-interval validation before serialization. The run
contains no capture file, founder freeze, held-task exposure, test panel, or
test access; its ledger records `test_access_count=0` and no registered/opened
test files. The failed job and fetched logs remain immutable.

This is a pre-test engineering failure under Section 14 of the frozen protocol,
not a scientific v3-2 outcome. The immediate next action is a minimal
Stage-A-only probability-boundary repair that retains raw log-odds, a focused
regression, full validation, and a freshly generated and inspected plan before
one fresh immutable retry. Once test access is consumed, no retry is permitted.

The primary compiler must make zero post-capture source queries. The graph text
parser is an oracle-like ceiling because the controlled prefix states every
edge; a pass cannot support a hidden-over-text claim.

Full operator instructions:

```text
docs/18_RATE_COMPUTE_OPERATIONAL_BASIS.md
docs/19_STAGE_R_CLUSTER_RUNBOOK.md
docs/20_STAGEA_V3_PROTOCOL.md
```

## Decision after RC0

### Basis readout fails

Stop the current graph/source contract. Do not train a larger latent model.

### Basis passes, composition fails against the prior

Investigate structured calibration/dependence on development data only. Do not
change the private representation.

### Composition beats the prior but not direct computation

Retain the basis as a diagnostic; it is not yet the constructive paper result.

### Composition beats direct computation

This is the observed RC0 branch. Draft one Stage-A v3 registration using:

- fresh claim-bearing worlds;
- a new unopened held sender;
- complete model-local token/slot compilers into typed public coordinates;
- separate self-future behavioral and oracle-semantic channels;
- text/token/direct/continuous/oracle baselines;
- receiver execution still locked.

The runtime basis probing in RC0 is an upper-bound diagnostic. Stage-A v3 must
replace it with a one-shot source-local compiler before any communication claim.

## Prohibited actions

- Running another Stage-Q scale screen under the immediate A/B contract.
- Treating RC0 interactive basis queries as a latent interface.
- Reusing RC0 development worlds for Stage-A confirmation.
- Reusing any development-exposed model as the future held sender.
- Resuming the shared-head oracle quotient.
- Selecting a Stage-A architecture on a fresh claim-bearing test split.
- Starting receiver-native execution.

## Evidence policy

Generated RC0 responses and model caches remain outside Git. Adopt only a small,
hash-verified package containing the frozen config/source identity, workflow
verification, metrics, machine decision, and independent audit. The adopted
package is `evidence/real_stage_r_olivia_rc0/`.
