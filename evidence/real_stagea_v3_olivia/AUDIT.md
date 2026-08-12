# Independent audit of Stage-A v3-2

Run: `frank-eq-stagea-v3-2-olivia-20260812b` on Olivia (`accel`, Slurm
`1899057`). Protocol: `stagea-v3-2`. Source archive:
`b6203d03...a44e0`. Runtime image: `a3ca46f0...aa3b1`.

## Preserved outcome

Stage-A v3-2 is a consumed, integrity-valid negative for the exact registered
one-shot typed-basis compiler. The machine decision is `fail` with diagnosis
`ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED`.

The run does not qualify receiver-protocol drafting, receiver execution, new
receiver-world access, a scientific claim, or a paper claim. It cannot be
retried, tuned, or repaired under the same registration.

This negative is narrow. It applies to the frozen all-token residual capture,
model-local compiler, renderer split, public directed-edge coordinates,
executor, baselines, and conjunctive gates. It does not show that the full
KV/runtime state lacks useful information, nor that every possible public
operational interface fails.

## Causal and artifact integrity

- The founder freeze precedes held-task onboarding. The held freeze precedes
  test creation by 14 seconds, and both freeze manifests report no existing test
  file.
- The access ledger records exactly one test grant, exactly 21 registered test
  paths, and exactly 21 hash-verified opens.
- All six test captures complete: 96 worlds per model and complexity, with
  10,368 logical queries at four entities and 12,096 at six entities.
- Every one of the eleven named integrity checks passes, including exact revisions,
  exclusive cloned-KV branching, prefix continuity, zero replay fallback, zero
  primary post-capture source queries, checkpoint-seed completeness, both
  freezes, required baselines, rate/compute declarations, and closed protected
  authorizations.
- A checksum-only remote/local comparison covers the complete 26.45-GB run and
  logs and reports no difference, missing file, or extra local file.

The original artifact manifest has 118 entries. All 118 exist and 117 retain
their bound hash. The sole terminal mismatch is `workflow_status.json`: the
manifest binds the completed pre-audit status (`f664a9a0...2c113`), while the
exception handler subsequently records the fail-closed audit refusal
(`dd67c7c3...cf5ba`). `independent_audit.json` is also necessarily outside the
pre-audit manifest because the audit failed before the workflow could add it.
No config, model, panel, capture, checkpoint, prediction, metric, decision,
rate, or run-summary artifact differs.

## Representation gates

All confidence intervals use the frozen 2,000 world-grouped bootstrap
replicates.

### Semantic and behavioral basis

Every semantic compiler improves over its training-world coordinate prior and
has balanced-accuracy lower bounds above 0.78. Nevertheless, every semantic
Brier score exceeds the conjunctive maximum of 0.10.

| Model | Entities | Semantic Brier | Gain lower 95% | Balanced-accuracy lower 95% | Semantic pass |
|---|---:|---:|---:|---:|---|
| Qwen3-4B | 4 | 0.1671 | 0.0638 | 0.8133 | no |
| Qwen3-4B | 6 | 0.1719 | 0.0633 | 0.7835 | no |
| Qwen3-8B | 4 | 0.1644 | 0.0649 | 0.8118 | no |
| Qwen3-8B | 6 | 0.1547 | 0.0821 | 0.8173 | no |
| Qwen3-14B held | 4 | 0.1689 | 0.0589 | 0.8178 | no |
| Qwen3-14B held | 6 | 0.1608 | 0.0733 | 0.8065 | no |

The behavioral channel passes in every group. Its Brier-gain lower bounds range
from 0.0559 to 0.1168. This is useful evidence that the captured state predicts
the frozen model's own post-reveal behavior, but it is not semantic basis
qualification.

### Renderer transfer

Natural and adjacency renderers have strongly positive semantic gains in every
group. The unseen renderer reverses the result in every group: all six gain
intervals are entirely negative, with lower/upper bounds spanning roughly
`[-0.3133, -0.1480]`. The miss is therefore a consistent renderer-transfer
failure across founders and the held sender, not absence of signal on the
development renderers.

### Composition and controls

Aggregate compiled hard-operation predictions beat both baselines:

| Baseline | Brier gain | Lower 95% | Aggregate pass |
|---|---:|---:|---|
| Training-world prior | 0.0365 | 0.0144 | yes |
| Training-selected direct protocol | 0.0352 | 0.0196 | yes |

The registered composition gate is conjunctive across all model/complexity and
family strata. It fails because held n=4 is negative against both baselines,
held n=6 fails against direct, Qwen3-8B n=4 fails against direct, and the
`mutual` family fails against both. Aggregate improvement cannot promote the
run.

Activation specificity also fails. The all-token compiler beats the continuous
and final-token controls, but its gain over the parameter-matched token-ID
control has interval `[-0.0165, 0.0221]`.

The remaining diagnostics pass:

- cross-model same-world retrieval is 1.0 with lower bound 1.0;
- wrong-world margin lower bound is 0.1043;
- model-ID accuracy is 0.2969 versus one-third chance;
- held semantic retention is 0.963/1.030 and behavioral retention is
  0.791/0.762 at four/six entities;
- four-bit compiled-gain retention is 0.9961;
- the independent oracle executor has zero hard-operation mismatches.

These positives identify useful structure but do not override the frozen
semantic, unseen-renderer, activation, and composition misses.

## Rate and compute

The primary four-bit semantic packet carries 48 payload bits at four entities
and 120 payload bits at six entities. Framing is reported separately at
1,744/1,752 bits. Primary compilation makes zero post-capture source queries;
consumer execution remains part of the interface.

At 16 amortized operations, payload rate is 3.0 bits per operation for four
entities and 7.5 for six. Interactive basis tomography would require 0.75 and
1.875 source queries per operation at the same boundary; direct querying uses
one source query per operation. These are accounting diagnostics, not a
communication claim.

## Fail-closed verifier diagnosis

All ordered workflow stages complete and the metrics, decision, rate,
predictions, run summary, and artifact manifest are written. The original
independent audit verifies 118 artifact files, all 21 test artifacts, every
integrity check, the exact decision, and exact rate/compute. It records only
`metrics_recomputed_exactly=false`, then the workflow intentionally raises
`independent Stage-A v3 audit failed`; Slurm exits `1:0` after 12:31:24.

Exact-runtime diagnostic job `1953471` reruns only the deterministic reducer in
the same ARM64 image and NumPy 2.2.6. It performs no model inference, training,
test access, or artifact mutation:

- workflow/config bundle order reproduces stored canonical metric SHA-256
  `10dd9254...45e41` exactly with zero differences;
- the verifier's lexicographic manifest-key order produces 46 numeric-only
  differences, maximum absolute magnitude `5.55e-17`;
- both orders reproduce `decision.json` exactly, with no gate difference.

The Slurm failure is therefore a deliberate fail-closed response to
floating-point reduction-order sensitivity, not congestion, OOM, CUDA failure,
capture failure, access leakage, metric ambiguity, or scientific disagreement.
The original failed audit remains immutable and is not relabeled as a pass.

## Stop decision

Stop the Stage-A v3-2 source/task/compiler contract. Do not select another seed,
renderer, checkpoint, threshold, baseline, or same-registration retry. Do not
draft or execute receiver work, and do not make a scientific or paper claim.

Any future research direction requires a separately justified and freshly
registered question. The present result suggests that renderer-invariant
semantic calibration and model-local compilation—not scale alone—would be the
central unresolved issues, but this exposed test result cannot be used to tune
such a protocol.
