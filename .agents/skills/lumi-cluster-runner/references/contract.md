# LUMI execution contract

Default environment overrides:

```text
FRANK_EQ_LUMI_HOST
FRANK_EQ_LUMI_ROOT
FRANK_EQ_LUMI_IMAGE
FRANK_EQ_HF_HOME
FRANK_EQ_ALLOW_PIP_INSTALL
```

The content-addressed source SHA, real config SHA, checkpoint revisions, operation descriptor hashes, cache hash, and final checkpoint hash must survive in machine-readable manifests.

A valid run contains `runs/run_manifest.json`, `runs/workflow_status.json`, and every artifact required by the requested stage list. `lumi/cli.py verify` checks this distinction:

- **workflow passed, scientific gate passed:** eligible only for the next phase named by the decision;
- **workflow passed, scientific gate failed:** terminal negative evidence under the frozen design;
- **workflow failed:** no scientific interpretation until the engineering defect is repaired.

Do not fetch multi-gigabyte caches unless diagnosis requires them; small decision, metric, manifest, and log artifacts are the first inspection surface. The current generic fetch command mirrors the run directory, so operators should use the smoke panel first and may add a selective fetch mode before full-scale caches.
