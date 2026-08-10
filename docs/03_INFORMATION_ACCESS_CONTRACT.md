# Information-access contract

## Roles

| Role | May access | Must not access |
|---|---|---|
| State capture | source prefix, source activations, source token/span metadata | future operation, query, target, receiver tensors |
| Future branch executor | cached state, one revealed operation, frozen source model | alternative branch outcomes, test labels from other worlds |
| Source chart trainer | source captures, source-side public state/signature labels on train worlds | receiver gradients, receiver identity, target hidden/logits |
| Held-sender onboarding | held sender train worlds and source-side labels | founder/receiver updates, confirmation/locked outcomes |
| Packet selector | frozen public state and revealed query | hidden target state, answer label |
| Receiver | packet, public query, receiver-local frozen model | source activations, sender identity, source process |
| Reducer | complete registered artifacts | unregistered rescue outputs |

## Temporal boundary

The state must be captured at step `t_capture`. The operation is revealed only at `t_reveal > t_capture`. Every real cache record must satisfy this relation.

The validation implementation is in `src/frank_eq/contracts.py`.

## Split boundary

World identity owns the split. A world includes:

- all renderer variants;
- all source models;
- all operation branches;
- all random seeds derived from that world;
- all packet rates and controls.

No member of one world may cross train, validation, test, confirmation, or locked roles.

## Public versus private objects

Public:

- operation registry;
- outcome space;
- fact ontology;
- quotient coordinate semantics;
- quantizer and packet schema;
- receiver rendering/execution contract;
- gates and reducer.

Private/model-local:

- hidden-state coordinates;
- source chart parameters;
- receiver-native internal realization;
- private workspace features.

## Assisted namespaces

Any condition using target hidden states, target logits, native target labels, pair-specific training, or oracle packet repair must be labeled `ASSISTED` or `ORACLE`. It cannot satisfy the strict gate.
