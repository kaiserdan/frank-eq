# Olivia execution contract

## Required inputs

- repository-relative real Stage-A YAML;
- immutable job name;
- content-addressed source archive;
- model checkpoints already present in the shared Hugging Face cache when `local_files_only: true`.

Default environment overrides:

```text
FRANK_EQ_OLIVIA_HOST
FRANK_EQ_OLIVIA_ROOT
FRANK_EQ_OLIVIA_IMAGE
FRANK_EQ_HF_HOME
FRANK_EQ_ALLOW_PIP_INSTALL
```

## Required remote outputs

```text
runs/run_manifest.json
runs/workflow_status.json
runs/cache/dataset.npz
runs/cache/metadata.json
runs/cache/cache_validation.json
runs/train/final.pt                 # when train requested
runs/train/training_summary.json    # when train requested
runs/eval/metrics.json              # when eval requested
runs/eval/decision.json             # when eval requested
runs/eval/artifact_manifest.json    # when eval requested
logs/slurm-<job>.out
logs/slurm-<job>.err
```

## Valid negative result

A job is operationally valid when the scheduler completes, `workflow_status.json` says `completed`, required artifacts exist, and hash/causal-boundary validation passes. `eval/decision.json` may still return a scientific failure. Do not relabel that as an engineering failure or launch a rescue automatically.

## Engineering failure

Missing manifests, missing cache validation, import/model-cache errors, OOM, timeout before artifacts, checksum mismatch, or causal-order violation are engineering failures. Diagnose from fetched logs and artifacts, fix the minimal cause, use a new job name, and preserve the failed run.
