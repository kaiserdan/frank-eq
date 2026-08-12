#!/usr/bin/env python3
"""Validate the frozen Stage-M0 config and exact event algebra without model inference."""

from __future__ import annotations

import argparse
import json

from frank_eq.moment_compute.workflow import static_contract_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/moment_compute/real_olivia_m0.yaml",
    )
    args = parser.parse_args()
    print(json.dumps(static_contract_summary(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
