#!/usr/bin/env python3
"""Verify a fetched SPQ0 run by independently refitting and reducing it."""

from __future__ import annotations

import argparse
import json

from frank_eq.shared_predictive_quotient.verify import verify_spq0_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--config",
        default="configs/spq0/real_olivia_spq0.yaml",
    )
    args = parser.parse_args()
    result = verify_spq0_run(args.run, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
