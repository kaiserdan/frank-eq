"""Operator CLI for the development-only SPQ0 census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_spq0_config
from .verify import verify_spq0_run
from .workflow import build_spq0_plan, run_spq0_audit, write_spq0_plan


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--config", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--config", required=True)
    plan.add_argument("--out", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--plan", required=True)
    run.add_argument("--inspected-plan-sha256", required=True)
    run.add_argument("--stages", default="audit")
    verify = commands.add_parser("verify")
    verify.add_argument("--config", required=True)
    verify.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        config = load_spq0_config(args.config)
        systems, basis = config.build_systems_and_basis()
        _print(
            {
                "status": "passed",
                "development_only": True,
                "models": [model.model_id for model in config.models],
                "reserved_unopened": [model.model_id for model in config.reserved_unopened_models],
                "systems": len(systems),
                "validation_only_systems": sum(
                    system.role == "validation_only" for system in systems
                ),
                "exact_rank": basis.exact_rank,
                "worst_core_condition_number": max(basis.core_condition_numbers.values()),
                "maximum_target_l1": basis.maximum_target_l1,
                "maximum_exact_executor_error": basis.maximum_exact_executor_error,
            }
        )
        return 0
    if args.command == "plan":
        config = load_spq0_config(args.config)
        payload = build_spq0_plan(config, config_path=args.config)
        write_spq0_plan(args.out, payload)
        _print(payload)
        return 0
    if args.command == "run":
        config = load_spq0_config(args.config)
        inspected = json.loads(Path(args.plan).read_text())
        result = run_spq0_audit(
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
        result = verify_spq0_run(args.run, config_path=args.config)
        _print(result)
        return 0 if result["passed"] else 1
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
