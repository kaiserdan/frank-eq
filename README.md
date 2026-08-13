# Frank-EQ

**Public operational interfaces for future-defined LLM state.**

Frank-EQ asks whether a state formed before an operation is revealed can be
compiled into a public description of the future computations it supports. It
does not treat hidden-coordinate similarity as interoperability.

For frozen model `M`, query-blind state `h`, future operation `k`, and explicit
post-reveal compute contract `c`:

```text
Sigma_M(h; k, c) = p_M(y | h, k, c).
```

## Current evidence

### Synthetic Stage 0

The synthetic reference passes as implementation evidence only. It establishes
that the original contracts, packet, held-sender workflow, and reducer execute.

### Real Stage A v1/v2: two exact-pipeline negatives

```text
v1  frank-eq-stagea-devg-v2   LUMI 20942127   STOP_OR_REVISE_STAGE0
v2  frank-eq-stagea-lumi-v2   LUMI 20952565   STOP_OR_REVISE_STAGE0
```

The v2 shared-head oracle quotient failed:

```text
native competence gain vs prior:  -0.0521
held-out signature Brier:          0.2065 (upper95 0.2421)
fact accuracy:                     0.5509 (lower95 0.5120)
cross-model retrieval:             0.1528 (lower95 0.0972)
wrong-world margin:               -0.0607
held-sender retention:            -0.3445
model-ID leakage over chance:      0.6389
```

This rules out the exact final-token, private-chart, shared-head pipeline. It
does not rule out an operational interface over the full runtime state.

### Stage Q: prompt correction and scale screens

A paired development experiment compared the historical assistant-continuation
construction with a proper new-user-turn reveal. Both failed source competence,
and the prompt effect was not identified:

```text
legacy     -0.098  [-0.152, -0.040]
chat_turn  -0.118  [-0.162, -0.068]
paired improvement -0.020 [-0.107, 0.071]
```

Subsequent development screens through Qwen3-8B also failed the aggregate and
per-founder gate. The operation pattern was highly structured:

```text
8B inverse lower95:             +0.136
8B reciprocity lower95:         +0.147
8B mutual lower95:              -0.758
8B lookup lower95:              -0.482
8B compose lower95:             -0.299
8B compare-outdegree lower95:   -0.252
```

The screens used an immediate single-token A/B readout. They therefore do not
separate answer-token calibration, post-query computation, and information
stored in the query-blind state.

## Stage R / RC0: completed development pass

RC0 was a development-only rate--compute and public-basis audit. The original
Olivia capture completed all 89,856 response rows, then failed before its first
metric because a grouped-aggregation helper no longer matched its callers. A
fresh, hash-bound artifact-only recovery restored the historical helper API and
executed no model inference:

```text
capture   frank-eq-rc0-rate-compute-olivia-20260811c  Slurm 1874736
recovery  frank-eq-rc0-rate-compute-olivia-20260811d-recovery  Slurm 1891471
```

Both repository verifiers and a separate recomputation audit pass. The machine
diagnosis is `PUBLIC_BASIS_COMPOSITION_SUPPORTED`:

```text
compiled hard-family Brier:                0.0408
training-selected direct Brier:            0.2035
training-world prior Brier:                0.2181
lower95 gain over direct / prior:          0.1542 / 0.1661
basis balanced accuracy, weakest group:    0.9246
hard-oracle executor mismatches:            0
```

Semantic sequence likelihood improved over the historical answer-token channel.
Generated reasoning did not: the reasoning-minus-pause Brier-gain interval was
`[-0.00540, -0.00029]`. The positive result is therefore public-basis recovery
and deterministic composition, not a reasoning-token effect.

RC0 compared:

1. historical immediate A/B token probability;
2. semantic false/true sequence likelihood;
3. 32 generated reasoning tokens;
4. 32 matched fixed pause tokens;
5. a public separating basis containing every directed edge;
6. a parameter-free executor that composes the basis into complex operations.

The key distinction was:

```text
answer-channel calibration
vs post-reveal computation
vs state/basis sufficiency
```

The public basis is gauge fixed by semantics: slot `(i,j)` always means the
same directed edge. For a closed directed graph, all `n(n-1)` edge slots form a
separating basis, so every registered structural operation factors through it.

The adopted, hash-verified evidence is in
`evidence/real_stage_r_olivia_rc0/`. Runtime basis interrogation remains
interactive tomography, not a one-shot hidden-state compiler or communication
result.

Read:

```text
docs/18_RATE_COMPUTE_OPERATIONAL_BASIS.md
docs/19_STAGE_R_CLUSTER_RUNBOOK.md
HANDOFF.md
AGENTS.md
```

Frozen configs:

```text
configs/rate_compute/real_lumi_rc0.yaml
configs/rate_compute/real_olivia_rc0.yaml
```

No RC0 rerun is authorized.

## Stage-A v3-2: completed exact-pipeline negative

The sole registered one-shot compiler workflow ran on Olivia as job `1899057`.
It preserved founder/held/test causal order, consumed its test grant exactly
once, completed all six test captures, and passed every integrity check. The
machine decision is:

```text
status:     fail
diagnosis:  ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED
```

Behavioral-basis prediction, public alignment, held-sender retention,
quantization, and oracle execution pass. The semantic basis misses its frozen
Brier ceiling, all six unseen-renderer gains are negative, and the conjunctive
composition and activation-specificity gates fail. Aggregate composition is
positive, but cannot override the failed registered strata.

The job exits 1 because the independent audit fails closed on 46
reduction-order differences no larger than `5.55e-17`. Exact-runtime diagnostic
job `1953471` reproduces stored metrics bit-for-bit in workflow/config order and
reproduces the same decision in both orders. This is a verifier-order refusal,
not an alternate scientific outcome; the failed audit remains immutable.

The compact adopted package is in `evidence/real_stagea_v3_olivia/`. Stage-A
v3-2 is consumed and terminal. No rerun, receiver protocol, receiver execution,
new receiver-world access, or claim is authorized.

## Stage M0: completed development negative

Stage M0 tested whether the nonlinear executor's public state was
undercomplete. Its full-grammar registry contains 318 typed edge, conjunction,
path, counterfactual, and joint-degree events. Olivia job `1970800` completed
all 105,984 response rows and 1,824 validation predictions with exact revisions,
exclusive cloned-KV branching, zero replay, and no held or test role.

The machine decision is:

```text
status:     fail
diagnosis:  OPERATION_CLOSED_EVENTS_NOT_READABLE
```

Exact public algebra and atomic retention pass. The moment executor beats the
cross-fitted direct response (`0.05310` versus `0.19445` Brier), but the simpler
marginal/independence executor is better (`0.02744`). Joint out-degree and
two-path-intersection events fail the frozen balanced-accuracy gate for both
models. The same conclusion holds model by model.

The scheduler, repository verifier, artifact hashes, and exact-runtime
specialized verifier pass. A newer workstation NumPy version changes only two
projection-adjustment diagnostic summaries by at most `4.44e-16`; no scientific
prediction field, metric, gate, or decision changes. The compact adopted
package is in `evidence/real_stage_m_olivia_m0/`.

This is interactive event tomography, not a one-shot latent interface. The
negative result authorizes no successor graph compiler draft or run, held
sender, receiver access, or claim. The current graph/source line stops.

## SPQ0: completed development negative

The fresh Shared Predictive Quotient census completed on Olivia as Slurm job
`2006680`. It used exact Qwen3-4B and Mistral-7B-Instruct-v0.3 founders,
categorical probability-bin likelihoods, an exact rank-four future-test core,
and the normalization-aware affine rank-three control. OLMo2 and Granite stayed
reserved and unopened with every access counter at zero.

The machine decision is:

```text
status:     fail
diagnosis:  SOURCE_PROBABILITY_PROTOCOL_NOT_QUALIFIED
```

Both founders fail every registered categorical source-protocol stratum. On
seen histories, Brier is `0.21985` for Mistral and `0.35346` for Qwen, versus a
`0.03061` target prior. Model-local semantic encoders recover useful seen-state
information, but both fail unseen-renderer and joint-OOD gates and neither
qualifies activation specificity over the parameter-matched token sequence.

Cross-family composition is asymmetric. Mistral-to-Qwen passes its directional
gate (`0.007503` Brier versus `0.008057` target prior; oracle-reader retention
`1.109`), while Qwen-to-Mistral retains only `0.498` of oracle-reader gain and
fails predictive-dimension localization. Sender identity remains recoverable at
`0.71846` accuracy, and the non-promotional residual census selects rank zero.
Four-bit retention passes in both directions but cannot rescue the failed
source, OOD, specificity, identity, and two-direction gates.

Slurm and workflow completion, exact causal branching, the reserved-checkpoint
non-access contract, and the Olivia-runtime independent verifier all pass. A
newer workstation NumPy stack changes Qwen fitted weights/compiled targets by
at most `2.04e-9` and refuses strict portability, but recomputes the same
decision and authorization vector. The compact adopted package is in
`evidence/real_spq0_olivia/`.

SPQ0 is consumed. It authorizes no SPQ1 draft or execution, reserved access,
held sender, test role, receiver work, or claim. There is no current executable.

Read:

```text
evidence/real_spq0_olivia/AUDIT.md
docs/28_SPQ0_RESULT_AND_DISPOSITION.md
docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md
docs/26_SPQ0_OLIVIA_RUNBOOK.md
docs/27_SPQ0_INDEPENDENT_REVIEW.md
```

## Historical RC0 decision tree

### Basis readout fails

Stop the current graph/source contract. Do not train another cross-model latent.

### Basis passes but public composition fails

Revise only structured calibration or executor assumptions on development data.
Do not enlarge a private latent code.

### Public composition beats the prior but not direct computation

The basis is a valid diagnostic but not yet a constructive paper result.

### Public composition beats the training-selected direct baseline

This is the observed branch. Draft exactly one Stage-A v3 registration with:

- fresh claim-bearing worlds;
- a new unopened held sender;
- complete model-local token/slot compilers into typed public coordinates;
- separate behavioral self-future and oracle-semantic channels;
- token-only, text, direct-operation, continuous-latent, and oracle-basis
  baselines;
- receiver execution still locked until the representation gate passes.

## Tested architecture and unresolved direction

Stage-A v3-2 tested this alternative to a shared private vector:

```text
query-blind frozen source state
        |
        v
model-local token/slot compiler
        |
        v
public separating operational basis
        |
        v
frozen deterministic or receiver-native executor
```

The frozen compiler did not qualify this interface. It predicts the sources'
behavioral channel and transfers across models on seen renderers, but semantic
calibration and unseen-renderer transfer fail. Stage M0 subsequently showed
that interactively querying a larger operation-closed graph basis does not beat
the simpler marginal executor. SPQ0 then found seen predictive-state decoding
and one directional transfer positive, but failed categorical source readout,
OOD stability, specificity, identity closure, and bidirectional transfer. Any
future direction must change the task-level scientific question and use fresh
roles; it cannot be selected on the exposed v3, Stage M, or SPQ0 outcomes.

## Commands

Synthetic and historical Stage-A commands:

```text
frank-eq validate-config
frank-eq run-stage0
frank-eq validate-real-config
frank-eq make-real-cache
frank-eq validate-real-cache
frank-eq diagnose-real-cache
frank-eq run-real-stagea
```

RC0 commands:

```text
frank-eq validate-rate-compute-config
frank-eq run-rate-compute-audit
frank-eq recover-rate-compute-audit
python scripts/verify_rate_compute_run.py
python scripts/audit_rate_compute_result.py
```

Stage-A v3 commands:

```text
frank-eq validate-stagea-v3-config
frank-eq plan-stagea-v3
frank-eq run-stagea-v3
frank-eq verify-stagea-v3
python scripts/verify_stagea_v3_run.py
```

Historical Stage M0 validation and verification commands:

```text
python scripts/validate_moment_compute.py
python scripts/verify_moment_compute_run.py --run <fetched-run-root>
```

Historical SPQ0 validation and verification commands:

```text
python scripts/validate_spq0.py
python olivia/cli.py status --job-name frank-eq-spq0-olivia-20260813c --json
python olivia/cli.py fetch  --job-name frank-eq-spq0-olivia-20260813c --json
python scripts/verify_spq0_run.py \
  --config configs/spq0/real_olivia_spq0.yaml \
  --run .agents/state/olivia/frank-eq-spq0-olivia-20260813c/remote/runs
```

These are provenance commands, not rerun or resubmission authority.

## Local validation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[real,dev]'

python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
bash -n olivia/*.sh olivia/*.slurm lumi/*.sh lumi/*.slurm
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
python scripts/validate_moment_compute.py
python scripts/validate_spq0.py
```

## Evidence hierarchy

```text
frozen protocol and source identity
causal cache/branch validation
machine decision
world-grouped metrics
prediction/response artifacts
training history or calibration state
W&B telemetry
prose
```

Generated runs, checkpoints, source archives, and `.agents/state/` remain outside
Git. Adopted evidence belongs under `evidence/` with a content hash manifest.

## Claim boundary

Frank-EQ currently establishes only:

- functioning synthetic and cluster infrastructure;
- three valid exact-pipeline negatives across the historical shared-code and
  one-shot typed-basis compiler architectures;
- a corrected source-qualification methodology;
- development evidence that an interactively queried, typed edge basis is
  recoverable from the tested sources and composes better than their
  training-selected direct protocols under the frozen RC0 contract;
- v3 evidence that the all-token compiler predicts the frozen sources'
  behavioral edge responses, while its semantic basis fails the registered
  calibration and unseen-renderer gates;
- Stage M development evidence that the registered high-order graph events are
  not uniformly readable and their projected executor is worse than the
  first-order marginal control, despite beating direct responses; and
- SPQ0 development evidence that typed semantic predictive state is decodable
  on seen renderers and one mapper-free cross-family direction works, while the
  frozen categorical readout, OOD stability, activation/history specificity,
  sender-identity closure, and bidirectional quotient gates fail.

It does not yet establish a cross-model public interface, a hidden-state
advantage over text/tokens, a qualified one-shot compiler, receiver execution,
or a positive ICLR claim. SPQ0 is completed negative evidence, not authority to
retune its exposed roles or draft SPQ1. No experiment or cluster executable is
currently authorized.
