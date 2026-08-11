# Independent audit of Stage R / RC0

Run: `frank-eq-rc0-rate-compute-olivia-20260811d-recovery` on Olivia
(`accel`, Slurm `1891471`). The model responses came from the preserved failed
capture `frank-eq-rc0-rate-compute-olivia-20260811c` (Slurm `1874736`).

## Preserved outcome

RC0 is an engineering-valid development pass with machine diagnosis
`PUBLIC_BASIS_COMPOSITION_SUPPORTED`. Both repository verifiers and a separate
recomputation audit pass with no failures. The public directed-edge basis clears
every frozen basis and composition gate.

This result authorizes drafting exactly one Stage-A v3 registration. It does not
authorize a v3 outcome run, claim-bearing test access, receiver execution, or a
scientific claim. Runtime basis interrogation remains interactive tomography,
not a one-shot hidden-state compiler or a communication result.

## Recovery integrity

The original job completed both pinned models and wrote all 89,856 raw and
calibrated rows before failing in its first grouped metric. The failure was a
mechanical API regression: `aggregate_by_world` had been changed from returning
`(world_ids, world_means)` to returning only `world_means`, while every caller
retained the frozen two-output contract.

The repair restored that historical contract and its row-count guard, added a
regression with more than two worlds, and passed compile, Ruff, 91 tests, shell
validation, and both repository validators. The recovery used a fresh source
archive and job root. It verified SHA-256 for every reused artifact, required
that no compiled prediction, metric, decision, summary, or artifact manifest
already existed, copied rather than modified the original capture, and executed
no model inference. The original failed run remains immutable.

Key lineage hashes:

- frozen config: `f5edfb7f...1268`;
- original capture source: `c24ae1e...de48`;
- repaired recovery source: `84ea4112...c0b7`;
- recovery input manifest: `c7caddb4...2ecb`;
- raw responses: `7374f94f...4782`;
- calibrated responses: `932f561a...3cc`.

## Basis gate

All intervals use 2,000 world-grouped bootstrap replicates over the 29 validation
worlds per complexity.

| Model | Entities | Calibrated Brier | Balanced accuracy | Brier gain over prior, lower 95% | Pass |
|---|---:|---:|---:|---:|---|
| Qwen3-4B | 4 | 0.0424 | 0.9389 | 0.1923 | yes |
| Qwen3-4B | 6 | 0.0468 | 0.9246 | 0.1770 | yes |
| Qwen3-8B | 4 | 0.0041 | 0.9957 | 0.2309 | yes |
| Qwen3-8B | 6 | 0.0145 | 0.9779 | 0.2112 | yes |

All 84 typed basis calibrators have positive slopes. Across all 180 calibration
groups, every fit converged; ten target-response maps have legal negative slopes.

## Composition gate

Across 3,712 hard-family validation predictions, compiled Brier is 0.0408,
versus 0.2035 for the training-selected direct protocol and 0.2181 for the
training-world operation prior. The lower-95 gains are 0.1542 over direct and
0.1661 over prior.

| Model | Entities | Compiled Brier | Lower 95% over direct | Lower 95% over prior |
|---|---:|---:|---:|---:|
| Qwen3-4B | 4 | 0.0623 | 0.1266 | 0.1414 |
| Qwen3-4B | 6 | 0.0721 | 0.1215 | 0.1262 |
| Qwen3-8B | 4 | 0.0049 | 0.1794 | 0.1987 |
| Qwen3-8B | 6 | 0.0240 | 0.1687 | 0.1751 |

Every hard family is positive over both baselines. The weakest family lower
bound over direct is still 0.1005 (`mutual`); over prior it is 0.1224. The
independent executor check has zero hard-oracle mismatches.

The one-bit basis also remains positive: 12 bits for four-entity worlds and 30
bits for six-entity worlds, compiled Brier 0.0599, with lower-95 gains 0.1321
over direct and 0.1435 over prior. This is a rate diagnostic only: it requires
12 or 30 source queries, compared with one direct target query.

## Response-channel and compute diagnostics

Semantic sequence likelihood improves over the historical answer-token channel:
Brier gain estimate 0.0469, 95% interval `[0.0414, 0.0520]`.

Generated reasoning does not explain the positive composition result. Against
the matched 32-token pause control, its Brier gain is -0.00294 with interval
`[-0.00540, -0.00029]`; against immediate semantic sequence scoring it is
-0.01396 with interval `[-0.01846, -0.00949]`. The fixed pause and generated
conditions both record exactly 32 tokens. This negative diagnostic is preserved.

## Authorized continuation

Draft one fresh Stage-A v3 protocol using new claim-bearing worlds and a new
unopened held sender. The protocol must replace runtime basis interrogation with
complete model-local token/slot compilers into the same typed public edge
coordinates, keep behavioral and oracle-semantic channels separate, include the
registered text/token/direct/continuous/oracle baselines, and leave receiver
execution locked. Do not launch that protocol from this result alone.
