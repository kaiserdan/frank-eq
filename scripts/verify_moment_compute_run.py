#!/usr/bin/env python3
"""Independent verifier for fetched Stage-M0 Olivia artifacts."""

from __future__ import annotations

import argparse
import json

from frank_eq.moment_compute.verify import verify_moment_compute_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--config",
        default="configs/moment_compute/real_olivia_m0.yaml",
    )
    args = parser.parse_args()
    result = verify_moment_compute_run(
        args.run,
        config_path=args.config,
        write_verification=False,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
