#!/usr/bin/env python3
"""Operator CLI for the development-only PSR0 predictive-state census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frank_eq.predictive_state import (
    load_predictive_state_config,
    run_predictive_state_audit,
)
from frank_eq.predictive_state.verify import verify_predictive_state_run
from frank_eq.predictive_state.workflow import (
    build_predictive_state_plan,
    write_predictive_state_plan,
)


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", required=True)
    plan.add_argument("--out", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--plan", required=True)
    run.add_argument("--inspected-plan-sha256", required=True)
    run.add_argument("--stages", default="audit")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--config", required=True)
    verify.add_argument("--run", required=True)

    args = parser.parse_args()
    if args.command == "validate":
        config = load_predictive_state_config(args.config)
        _print({"status": "passed", "config": config.as_dict()})
        return 0
    if args.command == "plan":
        config = load_predictive_state_config(args.config)
        payload = build_predictive_state_plan(config, config_path=args.config)
        write_predictive_state_plan(args.out, payload)
        _print(payload)
        return 0
    if args.command == "run":
        config = load_predictive_state_config(args.config)
        inspected = json.loads(Path(args.plan).read_text())
        result = run_predictive_state_audit(
            config,
            config_path=args.config,
            output_dir=args.out,
            stages=args.stages,
            inspected_plan=inspected,
            inspected_plan_sha256=args.inspected_plan_sha256,
        )
        _print(result)
        return 0
    if args.command == "verify":
        result = verify_predictive_state_run(
            args.run,
            config_path=args.config,
        )
        _print(result)
        return 0 if result["passed"] else 1
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
