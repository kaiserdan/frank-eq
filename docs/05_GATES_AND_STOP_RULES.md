# Gates and stop rules

## Synthetic full gate

The synthetic Stage-0 decision is conjunctive:

| Metric | Gate |
|---|---:|
| Held-out signature Brier | upper 95% bound <= 0.16 |
| Grounded-fact accuracy | lower 95% bound >= 0.82 |
| Renderer cosine | mean >= 0.90 |
| Cross-model retrieval top-1 | lower 95% bound >= 0.65 |
| Correct-minus-hardest-wrong margin | lower 95% bound >= 0.08 |
| Full-minus-facts-only residual Brier gain | lower 95% bound >= 0.015 |
| 8-bit quantization retention | >= 0.90 |
| Held-sender retention | >= 0.70 |
| Model-ID leakage over chance | <= 0.18 |

A pass authorizes only a real-model canary.

## Real Stage-A v1 decision

The first real v1 decision is frozen and failed. Its gates are not changed by
the later audit.

The result falsifies only the executed v1 pipeline. Its passing renderer,
residual, and quantization checks cannot be interpreted independently because:

- renderer cosine was not conditioned on non-collapse;
- residual coordinates were explicitly rendered in the prefix;
- quantization retained a failed code.

## Existing-cache localization gate

The next action is a **diagnostic**, not a promotion gate.

Allowed data:

```text
fit: training worlds
score: validation worlds
test worlds: zero
```

Required diagnostic families:

1. facts from each layer and concatenated capture;
2. oracle future signatures;
3. each model's own future signatures;
4. residual/global coordinates;
5. renderer 0→1 and 1→0 transfer;
6. native model branches versus operation-wise priors.

The diagnostic writes a recommendation but every authorization field remains
false. Any selected design requires a new world seed and untouched test role.

## Stage-A v2 prerequisites

No outcome-bearing v2 run may be frozen until the diagnostic chooses exactly
one hypothesis. The registration must include:

- exact model revision pins;
- a native-competence gate;
- self-future and oracle-semantic targets in separate namespaces;
- explicit residual/token/KV capture definitions;
- KV reuse versus replay parity;
- local versus shared compiler scope;
- prior-relative fact/signature metrics;
- renderer invariance conditioned on world specificity;
- fresh train/validation/test worlds;
- real-specific decision metadata.

At minimum, a future real representation gate must require:

- positive held-out self-future prediction over an operation-wise prior;
- positive external semantic/fact prediction over an appropriate prior;
- held-out renderer transfer;
- correct-world specificity;
- low sender identity leakage conditioned on non-collapse;
- held-sender retention with the public interrogator frozen;
- explicit facts-only and behavior-signature-only comparisons;
- complete world-grouped uncertainty.

## Receiver gate

Receiver execution remains locked until a prospective real representation gate
passes. A receiver experiment must additionally beat:

- no communication;
- rate-matched transcript;
- rate-matched summary;
- token-only source chart;
- continuous hidden-state/soft-prefix baseline;
- matched wrong source;
- shuffled and zero packets.

## Stop rules

Stop or redesign the active hypothesis when:

1. native operation competence does not beat registered priors;
2. the complete registered capture cannot predict the model's own future
   signature;
3. fact grounding adds no information beyond priors;
4. held sender requires updating public execution;
5. renderer invariance and correct-world specificity cannot pass jointly;
6. raw text matches activation-derived state at the same rate;
7. receiver utility fails despite a passing representation gate;
8. a valid gate miss is followed only by unregistered rescue variants.

A gate change after outcomes creates a new explicitly post-outcome protocol and
cannot reinterpret the previous decision.
