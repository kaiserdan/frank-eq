# Independent implementation review of SPQ0

Status: reviewed prospective development protocol; no model experiment launched.

This review was performed after PR #6 implemented the Shared Predictive Quotient
census. The original implementation had a sound stochastic task, an exact
rank-four public predictive basis, cross-family founders, query-blind capture,
disjoint calibration/selection/validation roles, independently frozen target
readers, and no pair-specific source-target map. Four aspects could nevertheless
have supported an over-strong positive interpretation and were corrected before
execution.

## Corrections

1. The transcript control is now described precisely as a fixed causal token
   sketch with an independently selected, parameter-matched linear readout. It
   is not called a learned recurrent or Transformer sequence encoder.
2. Every ordered cross-family path evaluates both the activation-derived packet
   and a packet compiled from the source token sketch through the same frozen
   target reader. Promotion requires a positive paired activation-over-token
   lower confidence bound in both directions.
3. Predictive rank is evaluated at the cross-family target-reader endpoint.
   Rank four must strictly beat ranks one through three and be noninferior to
   ranks six and eight within the registered 0.002 Brier margin.
4. The Brier-equivalent rate scalarization is explicitly non-promotional because
   its compute/bit exchange rates are conventional rather than empirically
   identified. The rate frontier remains descriptive.

The implementation names were also corrected: coefficient-SVD truncation is
`truncated_ridge`, and the exploratory behavioral remainder is
`pooled_residual_pca`, not reduced-rank regression or MAXVAR-GCCA.

## Preserved scientific contract

The corrections do not alter the active or reserved model roster, stochastic
systems, history panels, public test registry, probability-bin protocol, causal
capture boundary, reserved-checkpoint non-access contract, or protected
authorizations. SPQ0 remains development-only. A machine pass may authorize only
drafting a separately frozen SPQ1 protocol; it never authorizes SPQ1 execution,
held-model access, receiver execution, a scientific claim, or a paper claim.

## Validation receipt

The corrected tree passed Python compilation, Ruff, 126 tests, shell syntax,
historical repository/evidence validation, the rate-compute validator, the
moment-compute validator, SPQ0 plan regeneration and static validation, reserved
checkpoint zero-access checks, and `git diff --check`. The temporary correction
workflow and helper removed themselves after committing the reviewed tree.

No Olivia job has been submitted. Before any launch, regenerate a
content-addressed dry run from the final reviewed commit and inspect the new
source archive hash, config hash, plan identity, runtime image, active checkpoint
registries, and zero-access reserved-model ledger.
