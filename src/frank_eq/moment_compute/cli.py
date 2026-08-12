"""Standalone command surface for the Stage-M Olivia canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_moment_compute_config
from .verify import verify_moment_compute_run
from .workflow import run_moment_compute_audit, static_contract_summary


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="frank-eq-moment-compute")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--stages", default="audit")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--config", required=True)
    verify.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            _print(static_contract_summary(args.config))
            return 0
        if args.command == "run":
            config = load_moment_compute_config(args.config)
            _print(
                run_moment_compute_audit(
                    config,
                    config_path=args.config,
                    output_dir=Path(args.out),
                    stages=args.stages,
                )
            )
            return 0
        result = verify_moment_compute_run(
            args.run,
            config_path=args.config,
            write_verification=False,
        )
        _print(result)
        return 0 if result["passed"] else 1
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:
        print(f"frank-eq-moment-compute: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
