# Artifact schemas

## Synthetic bundle

```text
data/dataset.npz
  world_ids
  model_ids
  renderer_ids
  hidden
  hidden_mask
  facts
  residual
  signatures
  operation_descriptors

data/metadata.json
  schema = frank_eq_synthetic_v1
  model_hidden_dims
  n_layers
  operation registry
  split manifest
```

## Training artifacts

```text
train/founders.pt
train/final.pt
train/training_history.jsonl
train/training_summary.json
```

The final checkpoint embeds the resolved configuration and shape contract.

## Evaluation artifacts

```text
eval/metrics.json
  schema = frank_eq_stage0_metrics_v1

eval/decision.json
  schema = frank_eq_stage0_decision_v1

eval/predictions.npz

eval/artifact_manifest.json
  checkpoint and result SHA-256 digests
```

## Operational packet

```text
schema: FRANK-EQ/OPERATIONAL-PACKET/1
task_family
query_operation_id
fact_values
probe_ids
probe_values
quantization_bits
uncertainty_value
checksum
```

Serialization is canonical JSON with sorted keys and compact separators. The checksum covers the packet body excluding the checksum field.

## Real cache contract

`StateCaptureRecord` contains source/world/renderer identity, split, prefix and hidden hashes, capture step, and an explicit pre-operation flag.

`FutureBranchRecord` contains the originating state, frozen operation descriptor hash, normalized outcome distribution, branch seed, and operation-reveal step.

`FutureSignatureRecord` binds one capture to all branches and rejects temporal, coverage, or duplicate-operation violations.
