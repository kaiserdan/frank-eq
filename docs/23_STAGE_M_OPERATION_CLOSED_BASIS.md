# Stage M0 protocol: operation-closed public event basis

Protocol: `stage-m0-operation-closed-basis`

Role: development-only kill canary. No held sender, test role, receiver, or claim
is authorized.

## Frozen object

Let `mu_h` be the operational uncertainty induced by a query-blind source state.
For Boolean graph variables `E`, a future operation has a multilinear expansion

```text
f_k(E) = sum_S a[k,S] product_{j in S} E_j.
```

Therefore

```text
E_mu[f_k(E)] = sum_S a[k,S] m_S(h),
m_S(h) = E_mu[product_{j in S} E_j].
```

Stage M uses the sparse event moments required by the registered operation
algebra. It does not assume that first-order edge marginals are independent.

## Event registry

The four-entity registry is generated from the complete operation-family grammar,
not from the sampled target instances. It contains:

- every directed edge event;
- every reciprocal-edge conjunction;
- every two-hop path conjunction;
- intersections of the two alternative paths;
- intervention-specific path events for the counterfactual-add grammar; and
- complete joint out-degree tables for every ordered entity pair.

The registry is public, typed, deterministic, hash-bound, and independent of
model identity.

## Exact executor

Lookup and inverse select edge events. Mutual selects the reciprocal conjunction.
Two-hop composition uses inclusion-exclusion over its two paths. Counterfactual
composition uses the corresponding intervention event coordinates. Out-degree
comparison sums the joint degree cells with `degree(source) > degree(target)`.

Exact binary event truth must reproduce every registered formal operation with
zero mismatch before model inference is permitted.

## Coherence projection

Calibration is model-local but event-ID agnostic within event kind and order.
After calibration, the public projection uses no labels:

- conjunction coordinates are clipped to Fréchet bounds;
- path intersections cannot exceed either parent path event; and
- every joint degree table is normalized to a probability simplex.

The historical marginal/independence executor remains unchanged as the primary
mechanistic control.

## Development roles

Worlds are deterministically split into three disjoint roles:

```text
calibration  fit affine answer-channel maps and item priors
selection    select the direct response protocol by family
validation   frozen scoring only
```

All model, renderer, event, and operation rows for one world inherit its role.
There is no test role.

## Frozen machine gate

`OPERATION_CLOSED_MOMENT_BASIS_SUPPORTED` requires all of:

1. exact public event algebra;
2. positive lower-95 event gain over item priors for every model/event-kind/order
   group;
3. lower-95 balanced accuracy at least 0.55 for every event group;
4. positive lower-95 gain over the marginal executor;
5. positive lower-95 gain over the cross-fitted direct protocol;
6. both gains for every model, not only in aggregate; and
7. no degradation on atomic lookup/inverse operations.

A pass authorizes only drafting a successor one-shot compiler protocol. Every
receiver, held-sender, test, scientific-claim, and paper-claim field remains
false.

## Stop rules

- Event readout fails: stop the current graph/source line.
- Events read out but do not beat marginals: the independence approximation was
  not the binding problem; pivot task rather than enlarge the basis.
- Moments beat marginals but not direct computation: retain a mechanistic result,
  not a constructive paper claim.
- Robust pass: draft one new one-shot compiler protocol on fresh development and
  confirmation roles; do not launch it automatically.
