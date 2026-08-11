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

### Real Stage A: two exact-pipeline negatives

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

## Current next step: one Stage-A v3 registration draft

The RC0 pass authorizes drafting exactly one fresh protocol. It does not
authorize launching it. The registration must freeze fresh claim-bearing worlds,
a new unopened held sender, complete model-local token/slot compilers into the
typed public basis, separate behavioral and oracle-semantic channels, and the
registered token/text/direct/continuous/oracle baselines. Receiver execution
remains locked.

## Frozen RC0 decision tree

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

## Architecture direction

The intended constructive architecture is no longer a shared private vector:

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

This synthesizes the accumulated Frank evidence: descriptive representation
prediction is common, direct hidden realization is unreliable, and the strongest
positive endpoint is structured information followed by native downstream
computation.

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

## Local validation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[real,dev]'

python -m compileall -q src scripts olivia lumi
ruff check src scripts tests
pytest -q
python scripts/validate_repo.py
python scripts/validate_rate_compute.py
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
- two valid negative results for narrow shared-code architectures;
- a corrected source-qualification methodology;
- development evidence that an interactively queried, typed edge basis is
  recoverable from the tested sources and composes better than their
  training-selected direct protocols under the frozen RC0 contract.

It does not yet establish a cross-model public interface, a hidden-state
advantage over text/tokens, a one-shot compiler, held-sender establishment,
receiver execution, or a positive ICLR claim.
