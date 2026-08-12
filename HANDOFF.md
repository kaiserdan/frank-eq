# Frank-EQ handoff

Snapshot: 2026-08-12

## Current authority

Synthetic Stage 0 passes as implementation evidence only.

Two historical real Stage-A runs are adopted exact-pipeline negatives:

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

Stage-A v3-2 has now completed as the third exact-pipeline negative:

```text
job:       frank-eq-stagea-v3-2-olivia-20260812b  Slurm 1899057
decision:  ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED
evidence:  evidence/real_stagea_v3_olivia/
```

All registered stages and integrity checks complete, and the single test grant
is consumed. Behavioral basis, public alignment, held-sender retention,
quantization, and oracle execution pass. Semantic calibration,
unseen-renderer transfer, activation specificity, and the conjunctive
composition gate fail.

The terminal Slurm failure is the fail-closed response to an
independent-verifier bundle-order difference of at most `5.55e-17`. The exact
runtime reproduces stored metrics bit-for-bit in workflow order and the same
decision in either order. The original failed audit remains immutable.

Stage-A v3-2 cannot be retried or tuned; receiver-protocol drafting, receiver
execution, receiver-world access, scientific claims, and paper claims all
remain false.

Merge commit `659c120` adds the separately frozen, development-only Stage M0
continuation. It is the sole newly authorized experiment:

```text
config:  configs/moment_compute/real_olivia_m0.yaml
cluster: Olivia
profile: full
stages:  audit
```

The immediate next executable is its content-addressed dry run. Stage M0 has no
held sender, claim-bearing test role, receiver access, or claim authority. A
pass permits drafting—not launching—one separately frozen successor compiler
protocol.

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

## Current Stage M0 direction

The v3 packet stores first-order edge marginals, while its nonlinear executor
implicitly treats uncertain edges as independent. In general,
`E[f(E)] != f(E[E])`: mutuality needs reciprocal conjunctions, two-hop
reachability needs path conjunctions and intersections, and degree comparison
needs a joint degree distribution. The v3 composition miss therefore does not
identify whether the missing object is source information or an
operation-incomplete public state.

Stage M0 isolates that distinction with interactive event tomography. Its
registry is generated from the complete four-entity operation grammar and
contains edge events, reciprocal conjunctions, two-hop path/intersection
events, load-bearing counterfactual-add events, and complete ordered-pair joint
degree tables. An exact affine executor consumes those typed events and is
compared against both the historical marginal/independence executor and a
direct response protocol selected on a disjoint development role.

The frozen static contract currently validates as:

```text
worlds:                 64 development-only
operations:             32
event coordinates:      318
models:                 Qwen3-4B, Qwen3-8B
event-registry SHA-256: 70ce5d31...a6d55
contract SHA-256:       769fbf65...8326
exact-executor errors:  0
```

Calibration, direct-protocol selection, and validation worlds are disjoint.
Event calibration is model-local but event-ID agnostic within event kind and
order. Corrected `chat_turn`, literal cloned-KV branching, no replay fallback,
world-grouped intervals, and closed protected authorizations remain mandatory.

The next executable is dry-run only:

```bash
python olivia/cli.py submit \
  --job-name frank-eq-moment-compute-m0 \
  --config configs/moment_compute/real_olivia_m0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

After inspecting the immutable plan, submit the identical command without
`--dry-run`, monitor to termination, fetch, run both verifiers, audit every
registered stratum, and adopt a compact evidence package. Only
`OPERATION_CLOSED_MOMENT_BASIS_SUPPORTED` permits a new protocol draft; it
never authorizes that successor run.

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

## Stage-A v3-2 execution record

The frozen implementation, preflight, immutable launch, evaluation, fail-closed
audit, fetch, exact-runtime diagnosis, and compact evidence adoption are
complete. The chronology below is retained as provenance, not as a launch
instruction.

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
not a scientific v3-2 outcome. The minimal repair retains raw log-odds, clamps
only the serialized float32 behavioral probability view to
`[1e-7, 1-1e-7]`, records the per-shard clamp count, and makes workflow and
independent verification require that storage contract. A focused endpoint
round-trip regression passes. The repair is committed as `d40cdf5`; the complete
local contract passes with 110 tests. The replacement inspected plan has
internal hash `7b509858...a113c`, plan-file hash `9a728f19...0cbf8`, and
implementation-tree hash `bf8c87fa...812fc` across the same 64 bound files. It
preserves the exact config, models, 1,824 prefix forwards, 213,408 logical
queries, and closed pre-run access flags.

The fresh immutable retry ran as Olivia Slurm job `1899057`, job name
`frank-eq-stagea-v3-2-olivia-20260812b`, bound to Git `4f93143`, source archive
`b6203d03...a44e0`, and the replacement plan above. The archive reproduced
byte-for-byte and independently passed all 64 bound hashes, 208 safe-file
checks, and generated-artifact/credential exclusions before submission. The
retry crossed the formerly failing first-shard boundary and froze both founder
models at `2026-08-12T06:14:41Z` with 50 hash-bound artifacts and no existing
test files. An independent hash of the 1.10-GB first capture matches the freeze
manifest. Its embedded summary records 1,115 float32 endpoint clamps, strict
open-interval targets, unchanged raw log-odds, zero replay, and zero primary
post-capture queries.

Only after that founder freeze did the ledger enter `held_onboard`. The held
Qwen3-14B has therefore now legitimately become task-open on registered
train/validation data. All four held train/validation captures and all 15 held
checkpoint units completed. The held freeze was written at `09:22:18Z` with 22
hash-bound artifacts, the exact founder-freeze hash, and no existing test file.

The one-way test boundary has now been crossed. At `09:22:32Z`, 14 seconds after
the held freeze, the ledger consumed its single grant, registered exactly the
21 expected test panel/capture/prediction paths, entered `evaluate`, and opened
the test manifest plus both panels under matching SHA-256 hashes. Qwen3-4B test
captures for both n=4 and n=6 have now completed all 96 worlds with 10,368 and
12,096 logical queries respectively and zero recorded errors. Independent
SHA-256 checks match `c44e661d...2e6c03` for the 780,205,761-byte n=4 capture
and `938ed418...7d758e0` for the 1,397,854,209-byte n=6 capture. Qwen3-8B test
captures have also completed all 96 worlds for both n=4 and n=6, with 10,368
and 12,096 logical queries and zero recorded errors. Independent hashes match
`e91dc820...19020a` for the 1,247,336,833-byte n=4 capture and
`5ba3ba0c...84e04f` for the 2,235,406,145-byte n=6 capture. The held Qwen3-14B
test captures have now completed all 96 worlds for both n=4 and n=6, with
10,368 and 12,096 logical queries and zero recorded errors. Independent hashes
match `edc92a81...d814e43` for the 1,558,724,915-byte n=4 capture and
`76d5ae33...295868` for the 2,793,722,099-byte n=6 capture. All six test
captures and all 12 registered prediction files now exist; the access ledger
records exactly one grant and all 21 registered opens. Reduction and audit are
complete. `decision.json` records `fail` with diagnosis
`ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED`: semantic-basis and unseen-renderer gates
fail, while behavioral basis, public alignment, held-sender retention,
quantization, oracle execution, and all eleven integrity checks pass. Composition
and activation-specificity gates also fail. Every authorization remains false.

The outcome job terminates Slurm `FAILED`/exit 1 only because its fail-closed
independent audit requires byte-exact metric recomputation. The audit verifies
118 artifact files, all registered test hashes, integrity, decision, and
rate/compute, but records `metrics_recomputed_exactly=false`. Exact-runtime
diagnostic job `1953471` proves the cause: workflow/config bundle order
reproduces stored metrics at SHA-256 `10dd9254...45e41` with zero differences,
whereas the verifier's lexicographic bundle order creates 46 floating-point
roundoff differences, maximum absolute delta `5.55e-17`, without changing the
decision. This is a verifier order-sensitivity refusal, not a scientific or
artifact ambiguity. The consumed result must not be rerun or altered. The
complete 26.45-GB run/log tree was fetched and matched Olivia under a
checksum-only mirror comparison. The compact hash-verified negative is adopted
under `evidence/real_stagea_v3_olivia/`.

The primary compiler must make zero post-capture source queries. The graph text
parser is an oracle-like ceiling because the controlled prefix states every
edge; a pass cannot support a hidden-over-text claim.

Full operator instructions:

```text
docs/18_RATE_COMPUTE_OPERATIONAL_BASIS.md
docs/19_STAGE_R_CLUSTER_RUNBOOK.md
docs/20_STAGEA_V3_PROTOCOL.md
docs/22_MAIN_RESULTS_AUDIT_AND_STAGE_M.md
docs/23_STAGE_M_OPERATION_CLOSED_BASIS.md
docs/24_STAGE_M_OLIVIA_RUNBOOK.md
```

## Historical decision after RC0

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
- Retrying, recovering, or tuning Stage-A v3-2 after its consumed test result.
- Drafting a receiver protocol from the negative v3-2 decision.
- Running Stage M0 with any stage other than `audit`.
- Tuning the Stage M registry, split roles, controls, thresholds, or gates after
  validation outcomes are available.
- Treating Stage M event tomography as a one-shot or communication result.
- Launching a successor compiler, held sender, confirmation role, or receiver
  merely because Stage M0 passes.

## Evidence policy

Generated responses, captures, checkpoints, prediction arrays, model caches,
and operational state remain outside Git. The adopted compact packages are:

```text
evidence/real_stage_r_olivia_rc0/
evidence/real_stagea_v3_olivia/
```

No Stage M evidence package exists yet. Create one only after a terminal run is
fetched and independently verified.
