#!/usr/bin/env python3
"""Validate the frozen development-only PSR0 contract without loading a model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from frank_eq.predictive_state.config import load_predictive_state_config  # noqa: E402
from frank_eq.predictive_state.panel import generate_predictive_panel  # noqa: E402
from frank_eq.predictive_state.workflow import (  # noqa: E402
    build_predictive_state_plan,
)
from frank_eq.utils import canonical_json_bytes, sha256_bytes  # noqa: E402

CONFIG = ROOT / "configs/predictive_state/real_olivia_psr0.yaml"
PLAN = ROOT / "configs/predictive_state/inspected_plan.json"
REQUIRED = (
    "docs/22_PREDICTIVE_STATE_PSR0.md",
    "docs/23_PSR0_OLIVIA_RUNBOOK.md",
    "configs/predictive_state/real_olivia_psr0.yaml",
    "configs/predictive_state/inspected_plan.json",
    "src/frank_eq/predictive_state/automaton.py",
    "src/frank_eq/predictive_state/config.py",
    "src/frank_eq/predictive_state/panel.py",
    "src/frank_eq/predictive_state/probes.py",
    "src/frank_eq/predictive_state/workflow.py",
    "src/frank_eq/predictive_state/verify.py",
    "tests/test_predictive_state.py",
)


def main() -> int:
    missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
    if missing:
        raise SystemExit(f"missing PSR0 files: {missing}")

    config = load_predictive_state_config(CONFIG)
    expected_plan = build_predictive_state_plan(config, config_path=CONFIG.relative_to(ROOT))
    stored_plan = json.loads(PLAN.read_text())
    if stored_plan != expected_plan:
        raise SystemExit("inspected PSR0 plan differs from the frozen config")
    without_hash = dict(stored_plan)
    observed_hash = without_hash.pop("plan_sha256")
    if observed_hash != sha256_bytes(canonical_json_bytes(without_hash)):
        raise SystemExit("inspected PSR0 plan has an invalid internal hash")

    automaton = config.build_automaton()
    basis = automaton.build_basis(
        horizons=config.automaton.candidate_horizons,
        n_target_tests=config.automaton.n_target_tests,
        target_seed=config.automaton.target_seed,
        max_condition_number=config.automaton.max_core_condition_number,
        max_target_l1=config.automaton.max_target_executor_l1,
    )
    train = generate_predictive_panel(
        automaton,
        basis,
        role="train",
        lengths=config.panel.train.lengths,
        histories_per_length=config.panel.train.histories_per_length,
        seed=config.panel.train.seed,
        min_entropy=config.panel.min_belief_entropy,
        max_entropy=config.panel.max_belief_entropy,
        min_core_variance=config.panel.min_core_variance,
        max_attempt_multiplier=config.panel.max_generation_attempt_multiplier,
    )
    validation = generate_predictive_panel(
        automaton,
        basis,
        role="validation",
        lengths=config.panel.validation.lengths,
        histories_per_length=config.panel.validation.histories_per_length,
        seed=config.panel.validation.seed,
        min_entropy=config.panel.min_belief_entropy,
        max_entropy=config.panel.max_belief_entropy,
        min_core_variance=config.panel.min_core_variance,
        max_attempt_multiplier=config.panel.max_generation_attempt_multiplier,
    )
    train_ids = {history.history_id for history in train.histories}
    validation_ids = {history.history_id for history in validation.histories}
    if train_ids & validation_ids:
        raise SystemExit("PSR0 train and validation history IDs overlap")
    if not np.allclose(
        basis.core_matrix @ basis.executor,
        basis.target_matrix,
        atol=config.gates.max_oracle_executor_abs_error,
        rtol=0.0,
    ):
        raise SystemExit("PSR0 public executor does not exactly factor target tests")

    quickstart = (ROOT / "olivia/quickstart.sh").read_text()
    for marker in (
        "configs/predictive_state/",
        "validate-predictive-state-config",
        "run-predictive-state-audit",
        "verify-predictive-state",
    ):
        if marker not in quickstart:
            raise SystemExit(f"Olivia quickstart lacks PSR0 dispatch marker {marker!r}")

    print(
        json.dumps(
            {
                "status": "passed",
                "development_only": True,
                "models": [model.model_id for model in config.models],
                "predictive_rank": basis.rank,
                "core_condition_number": basis.condition_number,
                "maximum_target_l1": basis.maximum_target_l1,
                "train_histories": len(train.histories),
                "validation_histories": len(validation.histories),
                "response_branches_per_model": expected_plan["compute"][
                    "response_branches_per_model"
                ],
                "plan_sha256": expected_plan["plan_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
