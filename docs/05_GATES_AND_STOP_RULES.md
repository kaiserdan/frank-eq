# Gates and stop rules

## Synthetic full gate

The full Stage-0 decision is conjunctive:

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

A pass writes `PROMOTE_REAL_MODEL_CANARY` and authorizes only implementation of the real-model representation canary.

## Real-model representation gate

Freeze exact thresholds in a new versioned protocol before the first outcome-bearing run. At minimum require:

- operation-agnostic code beats operation-local and raw-hidden baselines under matched rate;
- held-out operation and renderer generalization;
- correct-source versus matched-wrong-source specificity;
- low sender identity leakage conditioned on world;
- held-sender retention;
- explicit facts-only comparison;
- no operation available at state formation;
- complete world-grouped uncertainty.

## Receiver gate

Receiver execution remains locked until the real representation gate passes. A receiver experiment must additionally beat:

- no communication;
- rate-matched transcript;
- rate-matched summary;
- token-only source chart;
- continuous hidden-state/soft-prefix baseline;
- matched wrong source;
- shuffled and zero packets.

## Stop rules

Stop or redesign the quotient if any of the following persists under the frozen canary:

1. held-out operations fail while seen operations pass;
2. model identity remains strongly decodable after conditioning on world;
3. facts-only matches the full code;
4. held sender requires updating the public decoder;
5. renderer invariance and wrong-world separation cannot pass jointly;
6. raw text matches activation-derived state at the same rate;
7. receiver utility fails despite a passing representation gate;
8. a valid gate miss is followed only by unregistered rescue variants.

A gate change after outcomes creates a new explicitly post-outcome protocol and cannot reinterpret the previous decision.
