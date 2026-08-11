#!/usr/bin/env python3
"""Verify and independently recompute one fetched Stage-A v3 run."""

from __future__ import annotations

import argparse
import json

from frank_eq.stagea_v3.verify import verify_stagea_v3_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    result = verify_stagea_v3_run(
        args.run,
        config_path=args.config,
        write_audit=False,
        require_existing_audit=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
