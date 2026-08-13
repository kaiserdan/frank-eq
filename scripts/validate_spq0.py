#!/usr/bin/env python3
"""Validate the prospective SPQ0 config, basis, executor, plan, and access contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from frank_eq.shared_predictive_quotient.automaton import (
    build_shared_predictive_basis,
)
from frank_eq.shared_predictive_quotient.config import load_spq0_config
from frank_eq.shared_predictive_quotient.panel import (
    build_panels,
    render_probability_query,
)
from frank_eq.shared_predictive_quotient.workflow import (
    SPQ0_RUNTIME_IMAGE,
    SPQ0_RUNTIME_IMAGE_SHA256,
    build_spq0_plan,
)
from frank_eq.utils import canonical_json_bytes, sha256_bytes, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/spq0/real_olivia_spq0.yaml"
PLAN = ROOT / "configs/spq0/inspected_plan.json"
REGISTRATION = ROOT / "configs/spq0/registration.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_spq0_config(config_path)
    systems, basis = config.build_systems_and_basis()

    _require(config_path == DEFAULT_CONFIG, "SPQ0 validator accepts only the frozen config")
    _require(config.systems.system_seed == 2026084101, "SPQ0 system seed changed")
    _require(len(systems) == 3, "SPQ0 system count changed")
    _require(
        [system.role for system in systems] == ["fit", "fit", "validation_only"],
        "SPQ0 system-role order changed",
    )
    _require(
        basis.exact_rank == 4
        and basis.normalization_aware_dimension == 3
        and basis.maximum_rank == 8,
        "SPQ0 linear or normalization-aware dimension changed",
    )
    _require(len(basis.target_tests) == 24, "SPQ0 target-test registry changed")
    _require(
        max(basis.core_condition_numbers.values()) <= 5.0,
        "SPQ0 public core exceeds its condition-number bound",
    )
    _require(basis.maximum_target_l1 <= 4.0, "SPQ0 executor exceeds its L1 bound")
    _require(
        max(basis.normalization_aware_condition_numbers.values()) <= 25.0,
        "SPQ0 normalization-aware core exceeds its condition-number bound",
    )
    _require(
        basis.maximum_normalization_aware_target_l1 <= 14.0,
        "SPQ0 normalization-aware executor exceeds its L1 bound",
    )
    _require(
        basis.maximum_exact_executor_error <= 1e-10,
        "SPQ0 exact executor exceeds tolerance",
    )
    _require(
        basis.maximum_normalization_aware_executor_error <= 1e-10,
        "SPQ0 normalization-aware executor exceeds tolerance",
    )
    for system in systems:
        public = basis.public_matrices[system.system_id]
        target = basis.target_matrices[system.system_id]
        for rank in range(1, basis.maximum_rank + 1):
            observed_rank = int(np.linalg.matrix_rank(public[:, :rank], tol=1e-11))
            _require(
                observed_rank == min(rank, basis.exact_rank),
                f"SPQ0 rank-conditioned public matrix failed for {system.system_id}/{rank}",
            )
            if rank >= basis.exact_rank:
                error = float(
                    np.max(
                        np.abs(
                            public[:, :rank] @ basis.executors[rank][system.system_id]
                            - target
                        )
                    )
                )
                _require(error <= 1e-10, "SPQ0 target executor is not exact")
        for rank in range(1, basis.normalization_aware_dimension + 1):
            augmented = np.column_stack((public[:, :rank], np.ones(basis.exact_rank)))
            expected_rank = rank + 1
            _require(
                int(np.linalg.matrix_rank(augmented, tol=1e-11)) == expected_rank,
                "SPQ0 normalization-aware dimension census changed",
            )
        affine_rank = basis.normalization_aware_dimension
        affine_public = np.column_stack(
            (public[:, :affine_rank], np.ones(basis.exact_rank))
        )
        affine_error = float(
            np.max(
                np.abs(
                    affine_public
                    @ basis.normalization_aware_executors[affine_rank][system.system_id]
                    - target
                )
            )
        )
        _require(affine_error <= 1e-10, "SPQ0 affine-complete executor is not exact")

    held = systems[-1]
    altered_held = replace(
        held,
        transitions=systems[1].transitions.copy(),
        emissions=systems[1].emissions.copy(),
    )
    altered_basis = build_shared_predictive_basis(
        (*systems[:-1], altered_held),
        horizons=config.systems.future_horizons,
        exact_rank=config.systems.predictive_rank,
        maximum_rank=max(config.semantic_encoder.rank_grid),
        n_target_tests=config.systems.target_tests,
        target_seed=config.systems.core_selection_seed,
        max_core_condition_number=config.systems.core_condition_number_max,
        max_target_l1=config.systems.target_executor_l1_max,
        normalization_aware_dimension=config.systems.normalization_aware_dimension,
        max_normalization_aware_condition_number=(
            config.systems.normalization_aware_condition_number_max
        ),
        max_normalization_aware_target_l1=(
            config.systems.normalization_aware_target_executor_l1_max
        ),
    )
    _require(
        altered_basis.public_tests == basis.public_tests
        and altered_basis.target_tests == basis.target_tests,
        "validation-only system can influence the future-test registries",
    )
    _require(
        not np.array_equal(held.transitions, systems[0].transitions)
        and not np.array_equal(held.emissions, systems[0].emissions),
        "validation-only transition/emission law is not distinct",
    )

    panels = build_panels(config, systems, basis)
    expected_counts = {"calibration": 384, "selection": 192, "validation": 576}
    _require(
        {role: len(panel.histories) for role, panel in panels.items()} == expected_counts,
        "SPQ0 panel counts changed",
    )
    role_ids = [
        {history.history_id for history in panels[role].histories}
        for role in ("calibration", "selection", "validation")
    ]
    _require(
        not role_ids[0] & role_ids[1]
        and not role_ids[0] & role_ids[2]
        and not role_ids[1] & role_ids[2],
        "SPQ0 history roles overlap",
    )
    _require(
        all(
            history.system_role == "fit"
            for role in ("calibration", "selection")
            for history in panels[role].histories
        )
        and any(
            history.system_role == "validation_only"
            for history in panels["validation"].histories
        ),
        "SPQ0 held system leaked into fit roles",
    )
    query = render_probability_query(
        systems[0],
        basis.target_tests[0],
        bins=config.probability_protocol.bins,
        candidate_labels=config.probability_protocol.candidate_labels,
    )
    _require(
        "probability bin" in query.lower()
        and "true or false" not in query.lower(),
        "SPQ0 categorical response protocol changed",
    )
    _require(
        config.capture.chat_turn_shape
        == "user_prefix_fixed_assistant_ack_user_query",
        "SPQ0 cross-family chat-turn shape changed",
    )

    expected_plan = build_spq0_plan(config, config_path=config_path)
    inspected_plan = _load_json(PLAN)
    _require(
        canonical_json_bytes(inspected_plan) == canonical_json_bytes(expected_plan),
        "committed SPQ0 plan differs from deterministic recomputation",
    )
    plan_without_hash = dict(inspected_plan)
    internal_hash = plan_without_hash.pop("plan_sha256", None)
    _require(
        internal_hash == sha256_bytes(canonical_json_bytes(plan_without_hash)),
        "SPQ0 inspected plan has an invalid internal hash",
    )
    _require(
        inspected_plan["runtime"]["image"] == SPQ0_RUNTIME_IMAGE
        and inspected_plan["runtime"]["image_sha256"]
        == SPQ0_RUNTIME_IMAGE_SHA256
        and inspected_plan["runtime"]["profile"] == "full"
        and inspected_plan["runtime"]["stages"] == ["audit"],
        "SPQ0 runtime registration changed",
    )
    _require(
        inspected_plan["public_basis"]["numeric_canonicalization"]
        == {
            "basis_registry_decimal_places": 10,
            "runtime_arrays_retain_float64": True,
            "selection_score_decimal_places": 14,
            "tie_break": "candidate_registry_order",
        }
        and inspected_plan["public_basis"]["maximum_exact_executor_error_bound"]
        == config.gates.max_oracle_executor_abs_error,
        "SPQ0 cross-platform basis canonicalization changed",
    )
    _require(
        inspected_plan["public_basis"]["exact_rank"] == 4
        and inspected_plan["public_basis"]["normalization_aware_dimension"] == 3
        and inspected_plan["public_basis"]["implicit_zero_bit_coordinate"]
        == "null_test_probability_1"
        and inspected_plan["rate_compute"]["primary_payload_bits"] == 16
        and inspected_plan["rate_compute"]["normalization_aware_payload_bits"] == 12
        and inspected_plan["rate_compute"]["interpretation"]
        == "robust_linear_packet_versus_rate_minimal_affine_packet"
        and inspected_plan["systems"]["validation_scope"] == "local_law_perturbation",
        "SPQ0 predictive-dimension or held-system interpretation changed",
    )
    held_shift = inspected_plan["systems"]["validation_shifts"]["system-02"]
    _require(
        held_shift["parent_system_id"] == "system-00"
        and np.isclose(held_shift["parent_weight"], 0.90)
        and np.isclose(held_shift["transition_mean_row_total_variation"], 0.084)
        and np.isclose(
            held_shift["emission_mean_row_total_variation"],
            0.05916106270987659,
        ),
        "SPQ0 local validation-law shift changed",
    )
    _require(
        inspected_plan["access"]
        == {
            "future_test_revealed_before_capture": False,
            "held_sender": False,
            "receiver": False,
            "reserved_checkpoint_files_opened": 0,
            "reserved_checkpoint_model_loads": 0,
            "reserved_checkpoint_snapshot_resolution_attempts": 0,
            "test_role": False,
        },
        "SPQ0 plan opens a protected access path",
    )

    registration = _load_json(REGISTRATION)
    _require(
        registration.get("schema") == "frank_eq_spq0_registration_manifest_v1",
        "SPQ0 registration schema changed",
    )
    _require(
        registration.get("status") == "prospective_development_only"
        and registration.get("implementation_pr_launch_authorized") is False
        and not any(registration.get("access", {}).values()),
        "SPQ0 registration opens execution or protected access",
    )
    expected_registered_files = {
        "configs/spq0/inspected_plan.json",
        "configs/spq0/real_olivia_spq0.yaml",
        "docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md",
        "docs/26_SPQ0_OLIVIA_RUNBOOK.md",
        "frank_eq_spq0_config_skeleton.yaml",
        "frank_eq_spq0_research_and_implementation_plan.md",
    }
    registered_files = registration.get("files", {})
    _require(
        set(registered_files) == expected_registered_files,
        "SPQ0 registered file set changed",
    )
    for relative, expected_hash in registered_files.items():
        _require(
            sha256_file(ROOT / relative) == expected_hash,
            f"SPQ0 registered file hash changed: {relative}",
        )
    _require(
        registration.get("inspected_plan_sha256") == inspected_plan["plan_sha256"]
        and registration.get("active_checkpoint_revision_registry_sha256")
        == inspected_plan["active_checkpoint_revision_registry_sha256"]
        and registration.get("reserved_checkpoint_non_access_contract_sha256")
        == inspected_plan["reserved_checkpoint_non_access_contract_sha256"],
        "SPQ0 registration and inspected plan hashes differ",
    )

    cluster_source = (ROOT / "src/frank_eq/cluster.py").read_text()
    quickstart = (ROOT / "olivia/quickstart.sh").read_text()
    capture_source = (
        ROOT / "src/frank_eq/shared_predictive_quotient/capture.py"
    ).read_text()
    _require(
        "SPQ0 requires exactly --profile full --stages audit" in cluster_source
        and "configs/spq0/*" in quickstart
        and "frank_eq.shared_predictive_quotient.cli" in quickstart,
        "SPQ0 Olivia fail-closed routing is incomplete",
    )
    _require(
        "reserved_unopened_models" not in capture_source,
        "SPQ0 capture code references the reserved checkpoint registry",
    )

    summary = {
        "schema": "frank_eq_spq0_static_validation_v1",
        "status": "passed",
        "development_only": True,
        "systems": len(systems),
        "validation_only_systems": 1,
        "panel_histories": expected_counts,
        "test_histories": 0,
        "exact_rank": basis.exact_rank,
        "rank_grid": config.semantic_encoder.rank_grid,
        "core_tests": [test.to_dict() for test in basis.core_tests],
        "worst_core_condition_number": max(basis.core_condition_numbers.values()),
        "maximum_target_l1": basis.maximum_target_l1,
        "maximum_exact_executor_error": basis.maximum_exact_executor_error,
        "prefixes_per_model": inspected_plan["capture"]["prefixes_per_model"],
        "query_branches_per_model": inspected_plan["capture"]
        ["post_reveal_query_branches_per_model"],
        "ordered_cross_family_pairs": inspected_plan["composition"]
        ["ordered_cross_family_pairs"],
        "linear_predictive_rank": basis.exact_rank,
        "normalization_aware_dimension": basis.normalization_aware_dimension,
        "robust_packet_bits": inspected_plan["rate_compute"]["primary_payload_bits"],
        "rate_minimal_packet_bits": inspected_plan["rate_compute"]
        ["normalization_aware_payload_bits"],
        "pair_specific_mappers": 0,
        "config_sha256": sha256_file(config_path),
        "plan_file_sha256": sha256_file(PLAN),
        "plan_sha256": inspected_plan["plan_sha256"],
        "runtime_image_sha256": SPQ0_RUNTIME_IMAGE_SHA256,
        "active_checkpoint_revision_registry_sha256": inspected_plan[
            "active_checkpoint_revision_registry_sha256"
        ],
        "reserved_checkpoint_non_access_contract_sha256": inspected_plan[
            "reserved_checkpoint_non_access_contract_sha256"
        ],
        "reserved_snapshot_resolution_attempts": 0,
        "reserved_files_opened": 0,
        "reserved_model_loads": 0,
        "launch_performed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
