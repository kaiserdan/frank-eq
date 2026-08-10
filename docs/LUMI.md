# LUMI runbook

## Local environment

```bash
export FRANK_EQ_LUMI_HOST=lumi
export FRANK_EQ_LUMI_ROOT=/scratch/project_465002861/kaiserda/frank-eq
```

The Slurm script uses the `small-g` partition with one MI250X GCD, 32 CPUs, 128 GB memory, and a 12-hour limit. Change resources only by a versioned operational update; do not change the scientific config.

## Runtime

The default container is:

```text
/scratch/project_465002861/kaiserda/frank/build_env/usae-deps.sif
```

The default Hugging Face cache is:

```text
/scratch/project_465002861/kaiserda/frank/hf-cache
```

Override with `FRANK_EQ_LUMI_IMAGE` and `FRANK_EQ_HF_HOME`. The job loads the current LUMI, partition/G, ROCm, and Singularity binding modules and exports source/cache paths into the container.

## Dry run and submit

```bash
python lumi/cli.py submit \
  --job-name frank-eq-stagea-cache-lumi-v1 \
  --config configs/stage0/real_lumi.yaml \
  --profile full \
  --stages cache,validate \
  --dry-run --json

python lumi/cli.py submit \
  --job-name frank-eq-stagea-cache-lumi-v1 \
  --config configs/stage0/real_lumi.yaml \
  --profile full \
  --stages cache,validate --json
```

## Status, fetch, verify

```bash
python lumi/cli.py status --job-name frank-eq-stagea-cache-lumi-v1 --json
python lumi/cli.py fetch --job-name frank-eq-stagea-cache-lumi-v1 --json
python lumi/cli.py verify --job-name frank-eq-stagea-cache-lumi-v1 --json
```

Local state is stored under `.agents/state/lumi/<job>/`.

## ROCm cautions

- The Transformers backend receives `device: cuda`; ROCm PyTorch intentionally exposes the CUDA API.
- Keep extraction sequential by checkpoint. Do not replicate all three models across ranks for the canary.
- A missing model snapshot or unsupported checkpoint kernel is an engineering failure, not evidence against the quotient.
- Exact-prefix replay fallback counts must be reported because cache classes can differ across Transformers/ROCm builds.
