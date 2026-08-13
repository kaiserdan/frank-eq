# Independent audit of SPQ0

Run: `frank-eq-spq0-olivia-20260813c` on Olivia (`accel`, Slurm
`2006680`). Protocol: `spq0-development-census`. Source archive SHA-256:
`607ef102...528579`. Runtime image SHA-256: `a3ca46f0...aa3b1`.

## Preserved outcome

SPQ0 is a scheduler-valid, causal-integrity-valid development negative. The
machine decision is `fail` with diagnosis
`SOURCE_PROBABILITY_PROTOCOL_NOT_QUALIFIED`.

The result is stronger than an infrastructure failure and narrower than a
general impossibility claim. The exact predictive system and executor work;
both active checkpoints complete the frozen categorical forecast census; and
one ordered transfer direction qualifies. The registered source probability
protocol, semantic stability, activation/history specificity, sender-identity,
bidirectional transfer, and predictive-dimension gates do not jointly qualify.

Every protected authorization is false. In particular, SPQ0 authorizes no SPQ1
protocol draft or execution, reserved-checkpoint access, held sender, test role,
receiver execution, scientific claim, or paper claim.

## Execution and access integrity

- Slurm job `2006680` completed `0:0` in `00:51:22` on one GH200, 32 CPUs,
  and 128 GiB. Workflow time was 2,715.90 seconds, about 0.9 GPU-hours.
- Qwen3-4B revision `1cfa9a72...df60c` and Mistral-7B-Instruct-v0.3 revision
  `c170c708...bab71` loaded from exact offline snapshots.
- Each founder produced 2,880 query-blind capture rows, 92,160 post-reveal
  response branches, 92,160 exact prefix-continuity checks, and 46,080 exact
  event-boundary checks. Literal cloned-KV reuse and exclusive batching are
  true; replay branches are zero.
- Calibration, architecture selection, and validation contain 384, 192, and
  576 histories. There is no test role. The third transition/emission system,
  symbolic renderer, and length 32 occur only in validation.
- Final-token, event-boundary, all-token-summary, embedding, and
  parameter-matched token-sequence surfaces are present. Both selected rank-four
  semantic encoders use a final-token residual surface.
- OLMo2 and Granite remain genuinely unopened. Snapshot resolution, file open,
  tokenizer/model load, adapter construction, and inference counters are all
  zero.
- Both ordered cross-family pairs run with target-local readers frozen before
  source evaluation and with zero pair-specific mappers.

The deterministic public construction is exact: homogeneous linear rank is
four, normalization-aware affine dimension is three, maximum executor errors
are `5.55e-16` and `8.47e-16`, and the rank grid is `1,2,3,4,6,8`.

## Source forecasting fails first

The categorical likelihood protocol is not merely noisy; it is badly
miscalibrated relative to the future-test probabilities. Temperature selection
hits the largest registered value, `2.0`, for both founders. Even on the seen
condition, its Brier score is far worse than the target prior:

| Founder | Seen categorical Brier | Seen prior | Lower 95% gain over prior | Joint-OOD categorical Brier | Joint-OOD prior |
|---|---:|---:|---:|---:|---:|
| Mistral-7B-v0.3 | 0.21985 | 0.03061 | -0.19728 | 0.16399 | 0.01730 |
| Qwen3-4B | 0.35346 | 0.03061 | -0.33312 | 0.18193 | 0.01730 |

Every registered source-protocol stratum fails for both models. This is the
first frozen stop diagnosis. It means the ten-bin candidate likelihoods are not
a qualified behavioral probability readout for these exact models and prompts;
it does not establish that the query-blind activations contain no predictive
information.

## Semantic state is readable in-distribution but brittle

The model-local rank-four encoders recover useful semantic core information on
seen renderers:

| Founder | Seen semantic-core Brier | Seen history prior | Lower 95% gain | Unseen-renderer Brier | Joint-OOD Brier |
|---|---:|---:|---:|---:|---:|
| Mistral-7B-v0.3 | 0.02169 | 0.03611 | +0.01296 | 0.07592 | 0.06111 |
| Qwen3-4B | 0.01862 | 0.03611 | +0.01572 | 0.22679 | 0.19286 |

That signal does not survive the registered renderer/system/length conjunction.
Both models fail unseen-renderer and joint-OOD semantic gates; Qwen also fails
the unseen-system and length-32 strata. The activation packet does not beat the
parameter-matched token-sequence control robustly. On seen data, activation
minus token lower bounds are `-0.00316` for Mistral and `-0.00645` for Qwen;
they deteriorate further in joint OOD. History specificity also fails for both.

The frozen semantic compiler substantially beats the poor direct categorical
response on most strata, but that comparison is not sufficient. In joint OOD,
compiled target Brier is `0.08377` for Mistral and `0.06552` for Qwen, versus
priors `0.01730` for both. The unseen-renderer Qwen compiler is even worse than
its direct response (`0.13942` versus `0.12191`).

## Cross-family transfer is asymmetric

One direction is a genuine registered positive:

| Source to target | Rank-4 transferred Brier | Target prior | Lower 95% gain | Lower 95% activation gain over token | Oracle-reader retention | Direction passes |
|---|---:|---:|---:|---:|---:|---|
| Mistral to Qwen | 0.007503 | 0.008057 | +0.000421 | +0.000461 | 1.109 | yes |
| Qwen to Mistral | 0.000902 | 0.000971 | +0.000061 | +0.000045 | 0.498 | no |

The Qwen-to-Mistral direction fails because it retains only `0.498` of the
oracle-reader gain, below the frozen `0.70` requirement. Its rank-four packet
also does not localize the predictive dimension: normalization-aware affine
ranks one and two have slightly lower Brier than rank four. By contrast,
Mistral-to-Qwen passes the directional rank gate. Four-bit gain retention passes
in both directions, approximately `1.0020` and `1.0026`.

The directional positive cannot override the conjunctive two-direction gate.
It remains useful development evidence that a typed, mapper-free cross-family
composition can work in one orientation.

## Rate, residual, and identity diagnostics

The primary robust packet is 16 payload bits; the affine rank-three control is
12 bits. The registered 1/4/16/32-query utility scalarization is positive, but
only relative to a catastrophically poor direct categorical baseline and is
explicitly non-promotional. It is not evidence for a deployable communication
advantage.

After conditioning on the semantic core, the shared behavioral-residual census
selects rank zero with incremental gain zero. Sender identity remains visible:
validation accuracy is `0.71846`, or `0.21846` above chance, exceeding the
frozen `0.15` ceiling. The learned packets therefore do not establish a
model-invariant public quotient.

## Verification and portability diagnostic

The in-job independent verifier passes every registered check. It live-rehashes
the active checkpoints, validates all 32 artifact entries, regenerates systems
and panels, refits probes and target readers, recomputes predictions, metrics,
rate/compute, and the exact same decision, and reports maximum numeric delta
`0.0` at absolute/relative tolerance `1e-12`.

The workstation Python 3.14.6 / NumPy 2.5.2 verifier refuses that same strict
tolerance. Its maximum difference is `2.04e-9`, localized to Qwen linear-algebra
weights and compiled targets, with tied rank-selection method labels and a few
derived diagnostic fields. Panel actions and observations are unchanged; panel
posteriors differ by at most `4.44e-16`, with an additional tuple/list JSON
shape difference. Artifact hashes, basis, access contract, and machine decision
remain exact.

This is recorded as a cross-runtime portability refusal, not silently rounded
away. The exact Olivia-runtime recomputation is authoritative, the consumed run
is not modified, and the workstation discrepancy changes no gate or
authorization.

## Provenance and stop decision

Job `2006621` failed before model inference on a Python-3.10 startup
incompatibility. Job `2006673` used the repaired source but was operator-cancelled
before a complete outcome. Neither contributes a scientific result. Job
`2006680` is the sole completed outcome, from clean commit `2a0b2044...ab156`.

SPQ0 answers its frozen question negatively. Do not retune the categorical
elicitation, renderer handling, rank sweep, thresholds, or model pair on these
exposed roles; do not access the reserved checkpoints; and do not rerun SPQ0.
Any continuation requires a fresh question, fresh roles, and a separately
frozen protocol. There is currently no authorized cluster executable.

This compact package retains the run's config, registration, plan, systems,
public basis, causal/access manifests, fitted summaries, grouped metrics,
decision, and both verifier records. Generated NPZ captures, checkpoints,
predictions, panels, W&B state, logs, credentials, and `.agents/state/` remain
outside Git.
