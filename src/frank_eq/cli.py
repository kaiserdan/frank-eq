"""Command-line entry point for Frank-EQ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from frank_eq.config import load_config
from frank_eq.data.real import build_real_cache, validate_real_cache
from frank_eq.data.synthetic import SyntheticBundle, generate_synthetic_bundle
from frank_eq.diagnostics import diagnose_real_cache
from frank_eq.evaluation import Stage0Evaluator
from frank_eq.rate_compute import load_rate_compute_config, run_rate_compute_audit
from frank_eq.real_config import load_real_config
from frank_eq.training import Stage0Trainer
from frank_eq.utils import atomic_write_json
from frank_eq.workflow import REAL_STAGE_ORDER, run_real_stagea


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frank-eq",
        description="Future-defined operational equivalence quotient experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate a synthetic YAML contract")
    validate.add_argument("--config", required=True)

    make_data = subparsers.add_parser("make-synthetic", help="build the controlled Stage-0 bundle")
    make_data.add_argument("--config", required=True)
    make_data.add_argument("--out", required=True)

    train = subparsers.add_parser("train-stage0", help="train founder charts and onboard held sender")
    train.add_argument("--config", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--out", required=True)

    evaluate = subparsers.add_parser("eval-stage0", help="evaluate and reduce synthetic Stage 0")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--out", required=True)

    run = subparsers.add_parser("run-stage0", help="run the complete synthetic workflow")
    run.add_argument("--config", required=True)
    run.add_argument("--out", default=None)

    validate_real = subparsers.add_parser(
        "validate-real-config", help="validate a real-checkpoint Stage-A contract"
    )
    validate_real.add_argument("--config", required=True)

    make_real = subparsers.add_parser(
        "make-real-cache", help="capture frozen checkpoints and all post-reveal branches"
    )
    make_real.add_argument("--config", required=True)
    make_real.add_argument("--out", required=True)

    validate_cache = subparsers.add_parser(
        "validate-real-cache", help="audit a real Stage-A cache without training"
    )
    validate_cache.add_argument("--cache", required=True)

    diagnose_cache = subparsers.add_parser(
        "diagnose-real-cache",
        help="localize a failed Stage-A run using train/validation worlds only",
    )
    diagnose_cache.add_argument("--cache", required=True)
    diagnose_cache.add_argument("--out", required=True)
    diagnose_cache.add_argument("--ridge", type=float, default=10.0)

    run_real = subparsers.add_parser(
        "run-real-stagea", help="run selected real cache/diagnose/train/eval stages"
    )
    run_real.add_argument("--config", required=True)
    run_real.add_argument("--out", default=None)
    run_real.add_argument("--stages", default=",".join(REAL_STAGE_ORDER))

    validate_rate_compute = subparsers.add_parser(
        "validate-rate-compute-config",
        help="validate the development-only rate--compute operational-basis audit",
    )
    validate_rate_compute.add_argument("--config", required=True)

    run_rate_compute = subparsers.add_parser(
        "run-rate-compute-audit",
        help="run the paired response-channel, compute, and public-basis audit",
    )
    run_rate_compute.add_argument("--config", required=True)
    run_rate_compute.add_argument("--out", default=None)
    return parser


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-rate-compute-config":
            config = load_rate_compute_config(args.config)
            _print({"status": "passed", "config": config.as_dict()})
            return 0
        if args.command == "run-rate-compute-audit":
            config = load_rate_compute_config(args.config)
            root = Path(args.out or config.output_dir)
            _print(
                run_rate_compute_audit(
                    config,
                    config_path=args.config,
                    output_dir=root,
                )
            )
            return 0
        if args.command == "validate-real-config":
            config = load_real_config(args.config)
            _print({"status": "passed", "config": config.as_dict()})
            return 0
        if args.command == "make-real-cache":
            config = load_real_config(args.config)
            _print(build_real_cache(config, args.out))
            return 0
        if args.command == "validate-real-cache":
            _print(validate_real_cache(args.cache))
            return 0
        if args.command == "diagnose-real-cache":
            _print(diagnose_real_cache(args.cache, args.out, ridge=args.ridge))
            return 0
        if args.command == "run-real-stagea":
            config = load_real_config(args.config)
            root = Path(args.out or config.output_dir)
            _print(
                run_real_stagea(
                    config,
                    config_path=args.config,
                    output_dir=root,
                    stages=args.stages,
                )
            )
            return 0

        config = load_config(args.config)
        if args.command == "validate-config":
            _print({"status": "passed", "config": config.as_dict()})
            return 0
        if args.command == "make-synthetic":
            bundle = generate_synthetic_bundle(config.data)
            bundle.save(args.out)
            summary = {
                "status": "passed",
                "schema": "frank_eq_synthetic_build_v1",
                "views": bundle.n_views,
                "worlds": config.data.n_worlds,
                "models": config.data.n_models,
                "renderers": config.data.n_renderers,
                "operations": config.data.n_operations,
                "out": str(Path(args.out)),
            }
            atomic_write_json(Path(args.out) / "build_summary.json", summary)
            _print(summary)
            return 0
        if args.command == "train-stage0":
            bundle = SyntheticBundle.load(args.data)
            trainer = Stage0Trainer(config, bundle, args.out)
            _print(trainer.train())
            return 0
        if args.command == "eval-stage0":
            bundle = SyntheticBundle.load(args.data)
            evaluator = Stage0Evaluator(
                config,
                bundle,
                checkpoint_path=args.checkpoint,
                output_dir=args.out,
            )
            metrics, decision = evaluator.evaluate()
            _print({"metrics": metrics, "decision": decision})
            return 0 if decision["status"] == "pass" else 2
        if args.command == "run-stage0":
            root = Path(args.out or config.output_dir)
            data_dir = root / "data"
            train_dir = root / "train"
            eval_dir = root / "eval"
            bundle = generate_synthetic_bundle(config.data)
            bundle.save(data_dir)
            trainer = Stage0Trainer(config, bundle, train_dir)
            training_summary = trainer.train()
            evaluator = Stage0Evaluator(
                config,
                bundle,
                checkpoint_path=train_dir / "final.pt",
                output_dir=eval_dir,
            )
            metrics, decision = evaluator.evaluate()
            run_summary = {
                "schema": "frank_eq_stage0_run_v1",
                "run_name": config.run_name,
                "root": str(root),
                "training": training_summary,
                "metrics": metrics,
                "decision": decision,
            }
            atomic_write_json(root / "run_summary.json", run_summary)
            _print(run_summary)
            return 0 if decision["status"] == "pass" else 2
        parser.error(f"unsupported command: {args.command}")
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:
        print(f"frank-eq: {error}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
