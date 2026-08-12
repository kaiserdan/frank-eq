# Independent audit of Stage M0

Run: `frank-eq-moment-compute-m0` on Olivia (`accel`, Slurm `1970800`).
Protocol: `stage-m0-operation-closed-basis`. Source archive SHA-256:
`352215a6...27668`. Runtime image SHA-256: `a3ca46f0...aa3b1`.

## Preserved outcome

Stage M0 is a scheduler-valid and integrity-valid development negative. The
machine decision is `fail` with diagnosis
`OPERATION_CLOSED_EVENTS_NOT_READABLE`.

The exact public algebra and atomic-retention checks pass. The moment executor
also beats the train-selected direct response by a wide margin. It does not
beat the historical marginal/independence executor, however, and not every
registered high-order public event clears the frozen balanced-accuracy gate.

This result authorizes no successor compiler draft or run, held sender, test
role, receiver work, scientific claim, or paper claim. Under the frozen stop
rule, stop the current graph/source line rather than enlarging its basis.

## Execution and causal integrity

- Slurm job `1970800` completed `0:0` in `01:37:52` on one H200, 32 CPUs, and
  128 GiB. It consumed 1.6 GPU hours and stayed well inside the 12-hour limit.
- Qwen3-4B and Qwen3-8B loaded at the exact registered revisions. Each model
  completed 128 prefixes and 52,992 records, for 105,984 records total.
- Both models used exclusive cloned-KV batches, maximum batch size eight, with
  11,520 response batches and zero exact-replay branches per model.
- The development split contains 32 calibration, 13 protocol-selection, and
  19 validation worlds. The roles are disjoint and `test_world_ids` is empty.
- The generated registry contains 318 typed coordinates. Exact event truth
  reproduces all 32 formal operations with zero executor mismatch.
- The run writes 1,824 validation predictions. All 16 manifest-bound artifacts
  retain their SHA-256 values, and the fetched `runs/` and `logs/` trees match
  Olivia under checksum-only dry-run comparison.

## Event-readout gate

Most event families are strongly readable. Edges, reciprocal conjunctions,
two-hop paths, counterfactual paths, and counterfactual path intersections pass
for both models. Two high-order groups fail the conjunctive gate:

| Event group | Qwen3-4B balanced-accuracy lower 95% | Qwen3-8B balanced-accuracy lower 95% | Frozen minimum |
|---|---:|---:|---:|
| joint out-degree, order 6 | 0.5000 | 0.5000 | 0.55 |
| two-path intersection, order 4 | 0.5000 | 0.5294 | 0.55 |

Their Brier gains over their calibration-world priors remain positive. The
failure is specifically the frozen balanced-accuracy requirement, not absence
of every high-order signal. Because every model/event-kind/order group must
pass, `operation_closed_events_readable=false`.

## Composition gate

Across 1,216 hard-family validation rows, the projected moment executor has
Brier `0.05310`. The train-selected direct response has Brier `0.19445`, so the
moment basis improves by `0.14135`, lower 95% `0.11980`. The historical
marginal/independence executor is substantially better at Brier `0.02744`; the
moment-minus-marginal gain is `-0.02565`, interval
`[-0.02999, -0.02197]`.

| Model | Moment Brier | Marginal Brier | Direct Brier | Lower 95% over marginal | Lower 95% over direct |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 0.07818 | 0.05042 | 0.19676 | -0.03318 | 0.09259 |
| Qwen3-8B | 0.02801 | 0.00447 | 0.19214 | -0.02975 | 0.14421 |

The marginal comparison fails in every hard family: out-degree comparison is
decisively worse, while compose, counterfactual-add, and mutuality have lower
bounds at or below zero. Atomic lookup/inverse behavior is unchanged: both
executors have Brier `0.022746` and the retention interval is exactly `[0, 0]`.

## Verification and runtime diagnostic

The in-job independent verifier passes all nine checks: required artifacts,
hashes, role separation, registry identity, prediction/metric/decision exact
recomputation, completed workflow, and closed authorizations. The repository
launcher verifier also passes after fetch.

The workstation's default Python 3.14 / NumPy 2.5.2 verifier initially refused
byte-exact prediction equality. Localization found 96 of 1,824 rows differing
only in the repeated `projection_mean_absolute_adjustment` or
`projection_max_absolute_adjustment` diagnostics, with maximum absolute
difference `4.440892098500626e-16`. No moment, marginal, direct, prior, truth,
operation, or protocol field differed; metrics and the decision reproduced
exactly.

A fresh temporary Python 3.10 / NumPy 2.2.6 environment, matching the run's
NumPy version, then reproduces all predictions, metrics, and the decision
byte-for-byte and passes the specialized verifier. This classifies the initial
refusal as cross-version diagnostic reduction sensitivity, not a gate or
scientific discrepancy. The original artifacts and in-job verification remain
immutable.

## Stop decision

Stage M answers its frozen development question negatively. Querying the
operation-closed high-order events does not improve on the simpler marginal
executor for these exposed models and graph worlds, and two required event
groups do not qualify. Do not train a one-shot Stage M compiler or launch
another graph-scale screen. Any continuation needs a new task-level scientific
question and fresh registration; there is currently no authorized executable.
