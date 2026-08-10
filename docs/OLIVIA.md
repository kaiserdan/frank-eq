# Olivia execution notes

The synthetic Stage-0 workflow is CPU-capable and should be run locally or on one lightweight node. Do not consume H200/GH200 allocation for it unless validating the cluster wrapper.

Example:

```bash
sbatch cluster/olivia_stage0.slurm configs/stage0/synthetic_full.yaml
```

The wrapper:

- creates a job-local virtual environment under `$TMPDIR`;
- installs the repository without persisting credentials;
- writes outputs under the supplied run root;
- runs compile, tests, repository validation, and Stage 0;
- fails if the machine decision fails.

Real-model jobs must be added only after a versioned real-cache protocol exists. They must pin model revisions, container/image hashes, operation registry hashes, split manifests, and the exact source archive.
