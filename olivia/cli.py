#!/usr/bin/env python3
"""Olivia operator CLI for Frank-EQ real Stage A."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frank_eq.cluster import cluster_cli_main


if __name__ == "__main__":
    raise SystemExit(cluster_cli_main("olivia"))
