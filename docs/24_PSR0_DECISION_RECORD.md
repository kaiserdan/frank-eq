# PSR0 decision record

Date: 2026-08-12

This record is append-only with respect to the adopted RC0 and Stage-A v3-2
outcomes. It does not change either machine decision.

## Preserve Stage-A v3-2 as terminal

Preserved:

- one registered v3-2 test access was consumed;
- all integrity checks passed;
- the machine diagnosis is `ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED`;
- no receiver or claim authorization exists;
- the exact graph/compiler contract cannot be retried, tuned, or recovered.

Interpretation:

- strong seen-renderer semantic decoding shows that source information was not
  simply absent;
- universal unseen-renderer failure shows grammar-bound compilation;
- failure to beat token-ID controls shows that visible graph facts do not
  isolate activation-specific state;
- positive behavioral prediction motivates future-defined predictive state;
- graph-edge semantics are the wrong final positive task because they are
  explicitly available to parsers and token controls.

## Freeze PSR0

Decision: implement one fresh development-only predictive-state census before
opening any new claim-bearing role.

Registered object:

```text
noisy action-observation history
        -> query-blind frozen LLM state
        -> train-only linear public core-test readout
        -> exact public predictive executor
        -> held-out future-test probabilities
```

Registered models:

```text
Qwen3-4B  1cfa9a7208912126459214e8b04321603b3df60c
Qwen3-8B  b968826d9c46dd6066d109eabc6255188de91218
```

Registered data:

```text
train lengths:       8, 16
validation lengths:  8, 16, 32
fit grammars:        narrative, table
unseen grammar:      symbolic
train histories:     256
validation histories:192
held sender:         none
claim-bearing test:  none
```

Registered public state:

```text
predictive rank:       4
core future tests:     4
target future tests:  18
condition number:      1.8311192670
maximum executor L1:   2.4214182125
```

Registered primary gate:

1. every model and transfer condition beats the training-history prior;
2. every model beats the token-only control on joint OOD;
3. renderer transfer passes with Brier gap at most `0.02`;
4. validation-only length-32 transfer passes;
5. compiled target tests beat both prior and direct source queries on joint OOD;
6. oracle rank, conditioning, and exact execution pass.

Authorization:

```text
PSR0 development audit:             authorized
PSR Stage 1 protocol draft on pass: authorized
PSR Stage 1 execution:              not authorized
held-sender onboarding:             not authorized
claim-bearing test access:          not authorized
receiver execution:                 not authorized
scientific claim:                   not authorized
paper claim:                        not authorized
```

PSR0 validation histories may not be used to adapt the registered task and rerun
the same protocol. A positive result permits only a fresh prospective Stage 1
registration.
