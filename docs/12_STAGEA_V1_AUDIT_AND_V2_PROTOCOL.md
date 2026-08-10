# Stage-A v1 audit and Stage-A v2 prerequisite protocol

Status: frozen implementation protocol, not an outcome-bearing Stage-A v2 registration  
Date: 2026-08-10

## 1. Decision retained

`frank-eq-stagea-devg-v2` is retained as a valid failure of the exact v1
pipeline. Nothing in this document changes its gates, metrics, or decision.

The failed object was narrower than the project hypothesis:

```text
raw non-chat prefix
→ final-token residual at depths 0.35/0.60/0.85
→ model-local chart
→ shared fact/residual heads
→ formal graph interrogator
```

Stage A v1 did not separately identify source-state readability, compiler
capacity, private-gauge compatibility, objective interference, or native model
competence.

## 2. Correction to the scientific interpretation

The primary Frank-EQ definition is future-based:

```text
Sigma_M(h, k) = the frozen model M's response distribution
                after operation k is revealed to cached state h.
```

The v1 cache contains `model_signatures`, which instantiate this object, and
`signatures`, which contain externally correct oracle outcomes. V1 trains the
primary quotient against the oracle. It therefore tests an externally grounded
world-state compiler, not purely whether the model's own future-defined quotient
is readable.

Stage A v2 must separate:

1. **self-future readability** — predict `model_signatures` from pre-operation
   capture;
2. **semantic grounding** — predict externally defined facts/oracle outcomes;
3. **native competence** — compare `model_signatures` with oracle outcomes;
4. **cross-model public execution** — attempted only after the first three are
   identified.

## 3. Mandatory existing-cache localization

The next executable step uses the existing cache and only its train/validation
worlds. It is implemented by:

```bash
frank-eq diagnose-real-cache \
  --cache <run-root>/cache \
  --out <run-root>/diagnostics
```

The diagnostic fits fixed-ridge model-local probes for:

- facts;
- oracle future signatures;
- each model's own future signatures;
- declared residual coordinates.

It evaluates individual capture layers, their concatenation, both renderer
transfer directions, native branch competence, and prior baselines.

### Access rule

- Training worlds may fit probes.
- Validation worlds may score and choose a descriptive best capture.
- Test worlds and test labels may not be selected or scored.
- Diagnostic output cannot promote any architecture or authorize a new run.
- Any design choice informed by validation requires a new world seed and
  untouched Stage-A v2 test role.

## 4. Decision tree for the next frozen hypothesis

### A. Native competence fails

If source branches do not beat operation-wise priors on validation worlds,
freeze a model/task competence prerequisite before latent analysis. The next
cache may use a native chat template or stronger checkpoints, but state capture
must still precede operation reveal and exact prefix continuity must be audited.

### B. Native competence passes; self-future readability fails

The current capture is insufficient. The next state census must compare, under
fixed source-side validation:

- final-token residual;
- sequence mean plus final-token residual;
- typed selected-K/V summaries;
- a bounded token-sequence resampler.

The complete KV cache is the actual branched causal runtime state. A negative
result from three final-token residuals cannot be generalized to that state.

### C. Self-future readability passes; semantic grounding fails

Preserve a public behavior-signature channel and treat fact grounding as a
separate source-local calibration problem. Do not force one code to identify
both the model's future behavior and an oracle world state.

### D. Raw targets are readable; the quotient still fails

The bottleneck is the compiler/objective. Stage A v2 should use one complete
model-local compiler per sender:

```text
local capture chart
→ local fact/self-signature/residual heads
→ shared typed public coordinates
→ frozen public interrogator
```

Only the public coordinate semantics and interrogator are shared. Frank-EQ now
supports this architecture behind `model.public_head_scope: local`; it is
dormant until a versioned v2 config is frozen.

## 5. Requirements before an outcome-bearing Stage-A v2 run

A Stage-A v2 registration must include:

1. fresh world seed and untouched test role;
2. exact checkpoint revision pins in every model entry;
3. a native-competence gate defined before test execution;
4. exact capture stream definitions and dimensionality;
5. KV-reuse versus replay parity on a frozen audit sample;
6. local versus shared compiler scope declared in advance;
7. self-future and oracle-semantic metrics in separate namespaces;
8. prior-relative fact and signature metrics;
9. renderer invariance conditioned on non-collapse/specificity;
10. a real-specific reducer schema and authorization boundary.

Receiver-native execution remains locked until a prospective Stage-A v2
representation gate passes.
