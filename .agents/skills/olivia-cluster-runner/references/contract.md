# Olivia execution contract

## Required inputs

- repository-relative frozen Stage-A v3-2 YAML;
- immutable job name;
- content-addressed source archive;
- a committed, implementation-matching `configs/stagea_v3/inspected_plan.json`;
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
runs/access_ledger.json
runs/freeze_manifest.json
runs/held_onboarding_manifest.json
runs/test_panel_manifest.json
runs/models.json
runs/capture_validation.json
runs/compiler_checkpoints_manifest.json
runs/training_summary.json
runs/baseline_manifest.json
runs/predictions_manifest.json
runs/rate_compute.json
runs/metrics.json
runs/decision.json
runs/artifact_manifest.json
runs/independent_audit.json
logs/slurm-<job>.out
logs/slurm-<job>.err
```

## Valid negative result

A job is operationally valid when the scheduler completes,
`workflow_status.json` says `completed`, required artifacts exist, and both
hash/causal-boundary validation and independent recomputation pass.
`decision.json` may still return a scientific failure. Do not relabel that as
an engineering failure or launch a rescue automatically.

## Engineering failure

Missing manifests, import/model-cache errors, OOM, timeout before artifacts,
checksum mismatch, or causal-order violation are engineering failures. Before
test access, diagnose the minimal cause under a new immutable source/job and
preserve the failed run. After test access, the outcome is consumed and v3-2
cannot be retried.
