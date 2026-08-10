#!/usr/bin/env python3
"""Run the development-only native-competence prerequisite on a real cache."""

from __future__ import annotations

import argparse
import json

from frank_eq.qualification import qualify_real_cache


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qualify frozen source competence without touching test worlds"
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-brier-gain-lower95", type=float, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    args = parser.parse_args()
    result = qualify_real_cache(
        args.cache,
        args.out,
        min_brier_gain_lower95=args.min_brier_gain_lower95,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
