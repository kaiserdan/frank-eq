# Implementation status

Snapshot: 2026-08-12

## Completed evidence

- Synthetic Stage 0: implementation evidence only.
- Real Stage-A v1 and v2: completed exact-pipeline negatives.
- Corrected Stage-Q prompt comparison: completed negative.
- Stronger-checkpoint Stage-Q screens through Qwen3-8B: completed negative.
- Stage R / RC0: completed development pass; protocol drafting only.
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

## Current implementation boundary

Stage-A v3-1 is frozen and its causal core is now implemented. The first
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

Workflow orchestration, freeze-manifest construction, independent artifact
verification/recomputation, CLI and Olivia entrypoints remain to be completed
before the implementation freeze and content-addressed dry run.

The user authorized implementation and one representation run after the
registration and implementation are separately committed and validated. No
receiver execution is authorized. A v3 pass may authorize drafting one receiver
protocol; a miss is terminal for v3-1.

## Intended architecture after a pass

```text
frozen query-blind token/layer state
        -> complete model-local token/slot compiler
        -> public typed operational basis
        -> frozen deterministic or receiver-native executor
```

Stage-A v3 must use fresh worlds, a new unopened held sender, separate behavioral
and oracle-semantic channels, and strong token/text/direct/continuous/oracle
baselines. Runtime basis probing in RC0 is an upper-bound diagnostic, not a
latent interface.

## Not authorized

- another scale-only Stage-Q screen under immediate A/B readout;
- treating interactive basis probing as communication;
- reusing RC0 worlds or exposed models for held/confirmation roles;
- restarting the shared-head quotient;
- receiver execution.
