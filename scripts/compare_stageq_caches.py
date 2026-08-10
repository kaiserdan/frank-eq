#!/usr/bin/env python3
"""Compare two paired Stage-Q cache qualification results."""

from __future__ import annotations

import argparse
import json

from frank_eq.stageq import compare_native_competence_caches


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a paired development-cache comparison")
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-paired-improvement-lower95", type=float, default=0.0)
    args = parser.parse_args()
    result = compare_native_competence_caches(
        args.baseline_cache,
        args.candidate_cache,
        args.out,
        min_paired_improvement_lower95=args.min_paired_improvement_lower95,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
