# Implementation status

Snapshot: 2026-08-13

## Completed evidence

- Synthetic Stage 0: implementation evidence only.
- Real Stage-A v1 and v2: completed exact-pipeline negatives.
- Corrected Stage-Q prompt comparison: completed negative.
- Stronger-checkpoint Stage-Q screens through Qwen3-8B: completed negative.
- Stage R / RC0: completed development pass; protocol drafting only.
- Stage-A v3-2: completed exact-pipeline negative;
  `ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED`.
- Stage M0: completed, verified development negative;
  `OPERATION_CLOSED_EVENTS_NOT_READABLE`.
- Model-local public-head option: implemented but dormant.

The v2 shared public code failed with held-out Brier `0.2065`, fact accuracy
`0.5509`, cross-model retrieval `0.1528`, wrong-world margin `-0.0607`,
held-sender retention `-0.3445`, and model-ID leakage over chance `0.6389`.
The final-token, private-chart, shared-head architecture must not be resumed.

At 8B, the immediate answer-token source screen passed inverse and reciprocity
but failed mutual, lookup, composition, and out-degree comparison. The completed
screen therefore exposes a structured wall, but it does not identify whether the
wall is answer calibration, post-query computation, or missing state information.
The branch allowed no additional autoregressive scratchpad tokens.

## Completed implementation: Stage R / RC0

RC0 ran as a development-only audit on Olivia. It evaluated:

```text
answer_token    historical immediate A/B probability
sequence        semantic false/true sequence likelihood
reason          32 generated scratchpad tokens
pause           32 fixed matched-control tokens
```

It also probes a public separating basis containing every directed edge: 12
slots for four entities and 30 for six. A parameter-free executor composes the
calibrated basis into lookup, inverse, mutual, two-hop composition, out-degree
comparison, and counterfactual composition. Exact binary basis inputs reproduce
the formal oracle and are covered by tests.

Implemented surfaces:

```text
src/frank_eq/rate_compute/
configs/rate_compute/real_lumi_rc0.yaml
configs/rate_compute/real_olivia_rc0.yaml
scripts/verify_rate_compute_run.py
scripts/audit_rate_compute_result.py
scripts/validate_rate_compute.py
docs/18_RATE_COMPUTE_OPERATIONAL_BASIS.md
docs/19_STAGE_R_CLUSTER_RUNBOOK.md
evidence/real_stage_r_olivia_rc0/
```

Both cluster configs use pinned Qwen3-4B and Qwen3-8B revisions, corrected
`chat_turn`, exclusive KV reuse, no replay fallback, world-grouped intervals,
and no held or claim-bearing test role. Quickstart scripts accept only the
`audit` stage for these configs.

The original capture (`frank-eq-rc0-rate-compute-olivia-20260811c`, Slurm
`1874736`) completed all 89,856 response rows and then hit a mechanical grouped
aggregation API regression before producing any outcome. A fail-closed,
hash-bound recovery (`frank-eq-rc0-rate-compute-olivia-20260811d-recovery`,
Slurm `1891471`) copied the exact capture into a fresh run and executed no model
inference.

Both verifiers and an independent recomputation pass. The machine diagnosis is
`PUBLIC_BASIS_COMPOSITION_SUPPORTED`: hard-family compiled Brier is `0.0408`,
versus direct `0.2035` and prior `0.2181`, with lower-95 gains `0.1542` and
`0.1661`. Every frozen basis and composition stratum passes, and the independent
executor has zero hard-oracle mismatches. Generated reasoning is worse than the
matched pause control, so no reasoning-token benefit is established.

## RC0 promotion boundary

RC0 supports drafting one Stage-A v3 protocol only if every model/complexity
basis group passes prior-relative Brier and balanced-accuracy gates, and compiled
hard operations beat both the prior and the training-selected direct protocol.
Answer-channel and reasoning-versus-pause findings are diagnostic only.

A pass does not authorize a Stage-A run, hidden-state compiler training,
claim-bearing test access, receiver execution, or a scientific claim.

## Completed implementation: Stage-A v3-2

Stage-A v3-2 is frozen and its causal core is now implemented. The first
implementation checkpoint adds:

- hash-bound config loading for the exact registration;
- deterministic role-fresh panels with one complexity-specific operation
  registry shared across train, validation, and test;
- a canonical six-entity coordinate registry whose four-entity condition uses
  the correct 12-coordinate induced subgraph;
- model-local all-token cross-attention compilers with independent semantic and
  behavioral parameter sets; and
- a process-locked access ledger that refuses test creation until founder and
  held manifests both exist and verify, then consumes the registered access
  exactly once.

The second implementation checkpoint adds causally ordered all-token capture,
exclusive cloned-KV semantic and direct teachers, exact prefix bytes/token
offsets/layer hashes, world-balanced fitting, validation-only early stopping,
and hash-bound checkpoint reload. The token-ID resampler has exactly the primary
parameter count; the final-token MLP is constructed and checked within the
frozen 5% tolerance. A deterministic parser recovers every frozen renderer.

Focused tests cover deterministic panels, renderer completeness, padding
isolation, coordinate masking, channel disjointness, manifest hash drift, early
or repeated test access, capture/query accounting, config mutation, text-parser
exactness, baseline parameter matching, and a fit/save/reload/predict cycle.
The third implementation checkpoint adds the historical continuous quotient,
train-only Platt calibration and direct-protocol selection, typed packet
serialization, deterministic execution, all registered negative/text/oracle
conditions, hash-bound prediction bundles, world-grouped gates, public retrieval
and identity probes, held-sender retention, rate/compute accounting, and the
seven-way machine reducer. Payload, framing, checksum, source queries, generated
tokens, and consumer work remain separate quantities.

The final implementation checkpoint adds the complete ordered workflow,
hash-bound founder and held freeze manifests, an exact one-access test-file
registry, immutable plan comparison, separate generated-reasoning/pause-token
accounting, compiler FLOP estimates and measured inference/executor wall time,
an independent hash and metric recomputation audit, CLI commands, and a
seven-day one-GH200 Olivia entrypoint. The repository launcher rejects partial
stage sequences, stale plans, dirty source, wrong image provenance, and any
same-registration recovery after test access.

The one authorized representation workflow has now terminated. Olivia job
`1899057` completes every registered stage, passes all eleven integrity checks,
consumes one test grant, and emits the negative diagnosis
`ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED`. Behavioral basis, public alignment,
held-sender retention, quantization, and oracle execution pass; semantic basis,
unseen-renderer transfer, activation specificity, and the conjunctive
composition gate fail.

The workflow then fails closed because the independent verifier reduces bundle
keys lexicographically and demands byte-exact metrics. Same-runtime diagnostic
job `1953471` proves that config/workflow order reproduces stored metrics
exactly, while verifier order creates 46 numeric-only differences no larger
than `5.55e-17`; the decision is identical. The complete fetched tree matches
Olivia byte-for-byte, and compact evidence is adopted under
`evidence/real_stagea_v3_olivia/`.

The miss is terminal for v3-2. No rerun, tuning, receiver protocol, receiver
execution, receiver-world access, or claim is authorized.

## Tested Stage-A v3 architecture

```text
frozen query-blind token/layer state
        -> complete model-local token/slot compiler
        -> public typed operational basis
        -> frozen deterministic or receiver-native executor
```

The frozen implementation used fresh worlds, a new held sender, separate
behavioral and oracle-semantic channels, and the complete registered control
set. It did not qualify the one-shot public interface. Runtime basis probing in
RC0 remains an upper-bound diagnostic, not a latent interface.

## Completed Stage M0

Stage M0 is a separately frozen development-only question, not a v3 retry. It
tested whether the nonlinear composition gap was caused by an undercomplete
first-order marginal packet. The implementation adds:

```text
src/frank_eq/moment_compute/
configs/moment_compute/real_olivia_m0.yaml
scripts/validate_moment_compute.py
scripts/verify_moment_compute_run.py
docs/22_MAIN_RESULTS_AUDIT_AND_STAGE_M.md
docs/23_STAGE_M_OPERATION_CLOSED_BASIS.md
docs/24_STAGE_M_OLIVIA_RUNBOOK.md
```

The full-grammar four-entity registry contains 318 typed events for edges,
reciprocal conjunctions, paths and intersections, load-bearing
counterfactual-add events, and ordered-pair joint degree tables. The panel has
64 worlds, 32 operations, two renderers, and disjoint calibration,
direct-selection, and validation roles. There is no held or test role.

Olivia job `1970800` completed `0:0` in `01:37:52`, writing 105,984 records and
1,824 predictions. Static and runtime validation preserve event-registry
SHA-256 `70ce5d31...a6d55`, contract SHA-256 `769fbf65...8326`, and zero exact
executor mismatches.

The machine diagnosis is `OPERATION_CLOSED_EVENTS_NOT_READABLE`. Joint
out-degree and two-path-intersection groups fail balanced accuracy for both
models. The moment executor beats cross-fitted direct responses but is worse
than the marginal/independence control in aggregate and for each model. Both
verifiers pass in the NumPy 2.2.6 run environment; a newer NumPy changes only
non-scientific projection-adjustment summaries at sub-ULP scale.

Compact evidence is adopted under `evidence/real_stage_m_olivia_m0/`. The
negative closes the current graph/source line and authorizes no successor draft
or run, held/receiver/test role, or claim.

## Prospective implementation: SPQ0

SPQ0 implements a fresh development-only Shared Predictive Quotient census over
controlled stochastic systems. It does not reuse graph data. The implementation
adds:

```text
src/frank_eq/shared_predictive_quotient/
configs/spq0/real_olivia_spq0.yaml
configs/spq0/inspected_plan.json
configs/spq0/registration.json
scripts/validate_spq0.py
scripts/verify_spq0_run.py
docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md
docs/26_SPQ0_OLIVIA_RUNBOOK.md
.agents/skills/spq0-runner/SKILL.md
```

It freezes categorical probability-bin forecasting, an exact rank-four typed
future-test core, five query-blind surface families, parameter-matched token
controls, complete model-local encoders, frozen target-local readers, both
ordered Qwen/Mistral compositions, a rank and quantization census, a
non-promotional residual census, and a rate-aware amortized direct comparison.

The local plan is model-free. Olivia support is content addressed and rejects
dirty source, image/hash overrides, recovery, non-`full` profiles, and any stage
other than `audit`. The implementation PR performs deterministic dry runs only;
it does not launch a model job.

## Not authorized

- another scale-only Stage-Q screen under immediate A/B readout;
- treating interactive basis probing as communication;
- reusing RC0 worlds or exposed models for held/confirmation roles;
- restarting the shared-head quotient;
- retrying or tuning Stage-A v3-2;
- drafting or executing receiver work;
- using the exposed v3-2 test outcome to select a successor protocol;
- rerunning Stage M0 under any stage, scale, registry, or calibration;
- changing its event registry, development roles, baselines, thresholds, or
  gates after validation outcomes; or
- treating interactive Stage M tomography as one-shot communication.
- launching SPQ0 automatically from the implementation PR;
- using any SPQ0 stage other than `audit` or accessing its reserved checkpoints;
- allowing validation-only SPQ0 rows to select any protocol component; or
- treating an SPQ0 plan or future development outcome as a scientific claim.
