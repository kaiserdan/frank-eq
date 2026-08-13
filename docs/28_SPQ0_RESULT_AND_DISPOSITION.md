# SPQ0 result and disposition

Status: completed and consumed development negative, 2026-08-13.

The frozen Shared Predictive Quotient census ran on Olivia as
`frank-eq-spq0-olivia-20260813c`, Slurm `2006680`. It completed successfully at
the scheduler and workflow levels. The machine outcome is:

```text
status:     fail
diagnosis:  SOURCE_PROBABILITY_PROTOCOL_NOT_QUALIFIED
```

The compact adopted record is
`evidence/real_spq0_olivia/`. The pre-execution protocol and runbook in
`docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md` and
`docs/26_SPQ0_OLIVIA_RUNBOOK.md` remain immutable consumed inputs; this document
records the outcome without changing them.

## What ran correctly

Both exact founders completed the full audit: Qwen3-4B at revision
`1cfa9a72...df60c` and Mistral-7B-Instruct-v0.3 at revision
`c170c708...bab71`. The run used literal cloned KV, exact prefix continuity,
zero replay, three disjoint development roles, no test histories, all five
registered surfaces, both ordered cross-family compositions, and no
pair-specific mapper.

The public predictive construction is mathematically sound: exact linear rank
four, normalization-aware affine dimension three, and executor error below
`8.5e-16`. OLMo2 and Granite remained unopened under zero resolution/open/load/
inference counters. The in-job independent verifier recomputed the complete
pipeline with zero numeric difference.

## What failed scientifically

The categorical source readout is severely miscalibrated. On seen histories,
Mistral and Qwen Brier scores are `0.21985` and `0.35346`, compared with the
same `0.03061` prior. Both models fail every source-protocol stratum.

Semantic core decoding is positive on seen renderers—Brier `0.02169` for
Mistral and `0.01862` for Qwen, both better than the `0.03611` history prior—but
is not stable. Both fail unseen-renderer and joint-OOD gates; Qwen is especially
brittle (`0.22679` and `0.19286`). Neither model establishes activation
specificity over the parameter-matched token sequence, and both fail history
specificity.

Cross-family transfer is directional rather than shared. Mistral-to-Qwen passes
with rank-four Brier `0.007503` versus target prior `0.008057` and oracle-reader
retention `1.109`. Qwen-to-Mistral improves slightly on its prior but retains
only `0.498` of oracle-reader gain and fails the rank gate. Sender identity is
still recoverable at `0.71846` accuracy, `0.21846` above chance. The residual
census selects rank zero.

## Interpretation

SPQ0 rules out the exact frozen combination of probability-bin elicitation,
systems/renderers, surface selection, linear encoders, and bidirectional
cross-family gate. It does not show that LLM activations contain no predictive
state, nor that typed public predictive coordinates can never transfer. The
seen semantic signal and the Mistral-to-Qwen positive are real but
non-promotional because the registered source, OOD, specificity, identity, and
two-direction gates fail.

The positive amortized utility table is also non-promotional: it is measured
against the failed direct categorical protocol. Four-bit retention shows packet
compressibility conditional on a fitted packet; it does not rescue the failed
interface.

## Verification qualification

Olivia's Python 3.10.12 / NumPy 2.2.6 verifier passes exact recomputation. The
workstation Python 3.14.6 / NumPy 2.5.2 verifier refuses strict `1e-12`
portability, with maximum difference `2.04e-9` in Qwen fitted weights/compiled
targets and sub-femtoscale panel posterior differences. It nevertheless
recomputes the same decision and closed authorization vector. Preserve both
records; do not repair or rewrite the consumed run.

## Current authority

All protected fields are false, including `spq1_protocol_draft_authorized`.
Do not rerun or retune SPQ0, open the reserved checkpoints, draft or execute
SPQ1 from this outcome, introduce a pair-specific mapper, or promote the
residual/rate diagnostics. There is no current experiment or cluster
executable. A continuation must begin with a new scientific question, fresh
roles, and a separately frozen registration.
