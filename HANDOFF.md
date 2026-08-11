# Frank-EQ handoff

Snapshot: 2026-08-11

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

The only authorized next execution is **Stage R / RC0**, using one of:

```text
configs/rate_compute/real_lumi_rc0.yaml
configs/rate_compute/real_olivia_rc0.yaml
```

Run exactly `--stages audit`.

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

RC0 tests whether the frozen sources can expose that basis and whether a
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

All semantic scores receive train-only Platt calibration. Negative calibration
slopes are legal because a stable answer-label inversion is local calibration,
not missing information.

The public executor composes calibrated edge probabilities into lookup, inverse,
mutual, two-hop composition, out-degree comparison, and counterfactual
composition. Exact binary basis inputs are required to reproduce the formal
oracle.

## Promotion gate

RC0 supports drafting a Stage-A v3 protocol only if:

1. every model x complexity basis group has lower-95 Brier gain at least zero;
2. every basis group has balanced accuracy at least 0.60;
3. compiled hard operations beat the train-world operation prior;
4. compiled hard operations beat the training-selected direct response protocol
   for the aggregate and every model x complexity group.

Answer-channel and reasoning-over-pause effects are diagnostic only.

A pass authorizes protocol drafting, not execution.

## Next command

LUMI dry run:

```bash
python lumi/cli.py submit \
  --job-name frank-eq-rc0-rate-compute \
  --config configs/rate_compute/real_lumi_rc0.yaml \
  --profile full \
  --stages audit \
  --dry-run --json
```

Then remove `--dry-run` after checking the content-addressed source package.
Olivia uses the same command pattern and `real_olivia_rc0.yaml`.

Full operator instructions:

```text
docs/18_RATE_COMPUTE_OPERATIONAL_BASIS.md
docs/19_STAGE_R_CLUSTER_RUNBOOK.md
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

Draft one Stage-A v3 registration using:

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
verification, metrics, machine decision, and independent audit.
