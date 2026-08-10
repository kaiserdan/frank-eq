#!/usr/bin/env python3
"""Run the frozen development-only native-competence prerequisite."""

from __future__ import annotations

import argparse
import json

from frank_eq.qualification import qualify_real_cache


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify frozen source competence from cache metadata without "
            "touching test worlds or overriding registered thresholds"
        )
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = qualify_real_cache(args.cache, args.out)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
