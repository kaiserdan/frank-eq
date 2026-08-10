# Cluster wrappers

`olivia_stage0.slurm` runs the CPU-capable synthetic workflow and repository checks. It is a deployment sanity check, not a real-model job.

```bash
sbatch cluster/olivia_stage0.slurm configs/stage0/synthetic_full.yaml
```

Add no real-model wrapper until `docs/06_REAL_MODEL_PLAN.md` has been converted into a frozen versioned protocol with pinned model and data identities.
