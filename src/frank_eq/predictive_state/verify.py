"""Independent hash and reducer verification for the development-only PSR0 run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from frank_eq.utils import canonical_json_bytes, sha256_bytes, sha256_file

from .config import load_predictive_state_config
from .panel import PredictivePanel
from .probes import brier_score, quantize_probabilities
from .workflow import (
    _basis_from_config,
    _build_panels,
    _evaluate_model,
    _gate_decision,
    _load_capture,
)

_REQUIRED = (
    "config.yaml",
    "run_manifest.json",
    "dry_run_plan.json",
    "workflow_status.json",
    "automaton.json",
    "public_basis.json",
    "panels/train.json",
    "panels/validation.json",
    "capture_manifest.json",
    "probe_training.json",
    "predictions_manifest.json",
    "metrics.json",
    "decision.json",
    "run_summary.json",
    "artifact_manifest.json",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _numeric_equal(
    left: Any,
    right: Any,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> tuple[bool, float, list[str]]:
    """Compare JSON-like reducers without treating harmless float order as corruption."""

    mismatches: list[str] = []
    maximum = 0.0

    def visit(a: Any, b: Any, path: str) -> None:
        nonlocal maximum
        if isinstance(a, bool) or isinstance(b, bool):
            if a is not b:
                mismatches.append(path)
            return
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not (np.isfinite(float(a)) and np.isfinite(float(b))):
                if float(a) != float(b):
                    mismatches.append(path)
                return
            difference = abs(float(a) - float(b))
            maximum = max(maximum, difference)
            if not np.isclose(float(a), float(b), atol=atol, rtol=rtol):
                mismatches.append(path)
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                mismatches.append(path + ".keys")
                return
            for key in sorted(a):
                visit(a[key], b[key], f"{path}.{key}")
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                mismatches.append(path + ".length")
                return
            for index, (item_a, item_b) in enumerate(zip(a, b, strict=True)):
                visit(item_a, item_b, f"{path}[{index}]")
            return
        if a != b:
            mismatches.append(path)

    visit(left, right, "$data")
    return not mismatches, maximum, mismatches[:50]


def verify_predictive_state_run(
    run_root: str | Path,
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    missing = [relative for relative in _REQUIRED if not (root / relative).is_file()]
    if missing:
        return {
            "schema": "frank_eq_predictive_state_verification_v1",
            "passed": False,
            "checks": {"required_files_present": False},
            "missing": missing,
        }

    config = load_predictive_state_config(config_path)
    stored_config = load_predictive_state_config(root / "config.yaml")
    config_match = config.as_dict() == stored_config.as_dict()

    artifact_manifest = _json(root / "artifact_manifest.json")
    manifest_files = artifact_manifest.get("files", {})
    safe_manifest_paths = all(
        not Path(relative).is_absolute() and ".." not in Path(relative).parts
        for relative in manifest_files
    )
    artifact_hashes = (
        artifact_manifest.get("schema")
        == "frank_eq_predictive_state_artifact_manifest_v1"
        and safe_manifest_paths
        and set(_REQUIRED) - {"artifact_manifest.json"} <= set(manifest_files)
        and all(
            (root / relative).is_file()
            and sha256_file(root / relative) == expected
            for relative, expected in manifest_files.items()
        )
    )

    plan = _json(root / "dry_run_plan.json")
    plan_without_hash = dict(plan)
    plan_hash = plan_without_hash.pop("plan_sha256", None)
    plan_valid = (
        plan.get("schema") == "frank_eq_predictive_state_plan_v1"
        and plan_hash == sha256_bytes(canonical_json_bytes(plan_without_hash))
        and plan.get("config_sha256") == sha256_file(root / "config.yaml")
        and plan.get("development_only") is True
        and plan.get("access", {}).get("claim_bearing_test_role") is False
        and plan.get("access", {}).get("held_sender") is False
    )

    run_manifest = _json(root / "run_manifest.json")
    workflow = _json(root / "workflow_status.json")
    workflow_valid = (
        run_manifest.get("development_only") is True
        and run_manifest.get("stages") == ["audit"]
        and run_manifest.get("plan_sha256") == plan_hash
        and run_manifest.get("access_contract", {}).get("state_precedes_future_test")
        is True
        and run_manifest.get("access_contract", {}).get("literal_kv_reuse") is True
        and run_manifest.get("access_contract", {}).get("exact_replay_fallback")
        is False
        and workflow.get("state") == "completed"
        and workflow.get("completed_stages") == ["audit"]
    )

    automaton, basis = _basis_from_config(config)
    basis_match, basis_delta, basis_mismatches = _numeric_equal(
        basis.to_dict(), _json(root / "public_basis.json")
    )
    automaton_payload = {
        "schema": "frank_eq_predictive_automaton_v1",
        "state_names": list(automaton.state_names),
        "action_names": list(automaton.action_names),
        "observation_names": list(automaton.observation_names),
        "transitions": automaton.transitions.tolist(),
        "emissions": automaton.emissions.tolist(),
        "initial_belief": automaton.initial_belief.tolist(),
    }
    automaton_match, automaton_delta, automaton_mismatches = _numeric_equal(
        automaton_payload, _json(root / "automaton.json")
    )

    regenerated_panels = _build_panels(config, automaton, basis)
    panel_match = all(
        canonical_json_bytes(panel.to_dict())
        == canonical_json_bytes(_json(root / "panels" / f"{role}.json"))
        for role, panel in regenerated_panels.items()
    )
    loaded_panels = {
        role: PredictivePanel.from_dict(_json(root / "panels" / f"{role}.json"))
        for role in ("train", "validation")
    }
    role_ids = {
        history.history_id
        for history in loaded_panels["train"].histories
    } & {
        history.history_id
        for history in loaded_panels["validation"].histories
    }
    panel_roles_disjoint = not role_ids

    captures = _json(root / "capture_manifest.json")
    entries = captures.get("entries", {})
    expected_models = {model.model_id: model for model in config.models}
    capture_valid = set(entries) == set(expected_models)
    metrics_by_model: dict[str, Any] = {}
    training: dict[str, Any] = {}
    prediction_arrays_valid = True
    if capture_valid:
        expected_rows = int(plan["compute"]["prefixes_per_model"])
        expected_branches = int(plan["compute"]["response_branches_per_model"])
        predictions_manifest = _json(root / "predictions_manifest.json")
        for model_offset, (model_id, entry) in enumerate(sorted(entries.items())):
            arrays, metadata = _load_capture(root, entry)
            model = expected_models[model_id]
            capture_valid = capture_valid and (
                metadata.get("revision_requested") == model.revision
                and metadata.get("revision_observed") == model.revision
                and metadata.get("rows") == expected_rows
                and metadata.get("response_branches") == expected_branches
                and metadata.get("exact_prefix_continuity_checks")
                == expected_branches
                and metadata.get("runtime_basis_queries_are_development_tomography")
                is True
                and set(np.unique(arrays["role_ids"]).tolist()) == {0, 1}
            )
            metrics, training_artifact, predictions = _evaluate_model(
                config,
                basis,
                arrays,
                metadata,
                seed_offset=10_000 * model_offset,
            )
            metrics_by_model[model_id] = metrics
            training[model_id] = training_artifact
            prediction_entry = predictions_manifest.get("entries", {}).get(model_id, {})
            prediction_path = root / str(prediction_entry.get("path", ""))
            if (
                not prediction_path.is_file()
                or sha256_file(prediction_path) != prediction_entry.get("sha256")
            ):
                prediction_arrays_valid = False
                continue
            with np.load(prediction_path, allow_pickle=False) as stored:
                if set(stored.files) != set(predictions):
                    prediction_arrays_valid = False
                else:
                    for key, values in predictions.items():
                        if not np.allclose(
                            stored[key], values, atol=1e-12, rtol=1e-12
                        ):
                            prediction_arrays_valid = False

    oracle_rows = np.concatenate(
        [
            np.asarray([history.core_probabilities for history in panel.histories])
            for panel in regenerated_panels.values()
        ],
        axis=0,
    )
    target_rows = np.concatenate(
        [
            np.asarray([history.target_probabilities for history in panel.histories])
            for panel in regenerated_panels.values()
        ],
        axis=0,
    )
    oracle_error = float(
        np.max(np.abs(basis.execute(oracle_rows, clip=False) - target_rows))
    )
    quantization = {}
    for bits in config.probe.quantization_bits:
        quantized = quantize_probabilities(oracle_rows, bits)
        quantization[str(bits)] = {
            "bits_per_coordinate": bits,
            "message_bits": bits * len(basis.core_tests),
            "oracle_target_brier": brier_score(
                target_rows, basis.execute(quantized)
            ),
        }
    recomputed_metrics = {
        "schema": "frank_eq_predictive_state_metrics_v1",
        "scope": "development-only public predictive-state census",
        "public_basis": {
            "rank": basis.rank,
            "condition_number": basis.condition_number,
            "maximum_target_l1": basis.maximum_target_l1,
            "oracle_executor_max_abs_error": oracle_error,
            "core_tests": [test.to_dict() for test in basis.core_tests],
            "target_tests": [test.to_dict() for test in basis.target_tests],
        },
        "models": metrics_by_model,
        "quantization": quantization,
        "data_usage": {
            "train_histories": len(regenerated_panels["train"].histories),
            "validation_histories": len(regenerated_panels["validation"].histories),
            "claim_bearing_test_histories": 0,
            "held_sender_rows": 0,
            "receiver_rows": 0,
        },
    }
    recomputed_decision = _gate_decision(
        config, basis, metrics_by_model, oracle_error
    )
    metrics_match, metrics_delta, metrics_mismatches = _numeric_equal(
        recomputed_metrics, _json(root / "metrics.json")
    )
    decision_match, decision_delta, decision_mismatches = _numeric_equal(
        recomputed_decision, _json(root / "decision.json")
    )
    training_match, training_delta, training_mismatches = _numeric_equal(
        training, _json(root / "probe_training.json")
    )
    authorization = _json(root / "decision.json").get("authorization", {})
    protected_closed = not any(
        authorization.get(key, True)
        for key in (
            "psr_stage1_execution_authorized",
            "claim_bearing_test_access_authorized",
            "held_sender_onboarding_authorized",
            "receiver_execution_authorized",
            "scientific_claim_authorized",
            "paper_claim_authorized",
        )
    )

    checks = {
        "required_files_present": not missing,
        "config_snapshot_matches": config_match,
        "artifact_hashes_valid": artifact_hashes,
        "plan_hash_and_access_valid": plan_valid,
        "workflow_completed_exactly_audit": workflow_valid,
        "automaton_recomputed": automaton_match,
        "public_basis_recomputed": basis_match,
        "panels_recomputed": panel_match,
        "panel_roles_disjoint": panel_roles_disjoint,
        "captures_complete_and_pinned": capture_valid,
        "prediction_arrays_recomputed": prediction_arrays_valid,
        "probe_training_recomputed": training_match,
        "metrics_recomputed": metrics_match,
        "decision_recomputed": decision_match,
        "protected_authorizations_closed": protected_closed,
    }
    return {
        "schema": "frank_eq_predictive_state_verification_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "decision": recomputed_decision.get("decision"),
        "diagnosis": recomputed_decision.get("diagnosis"),
        "numeric_comparison": {
            "maximum_abs_delta": max(
                basis_delta,
                automaton_delta,
                metrics_delta,
                decision_delta,
                training_delta,
            ),
            "mismatches": {
                "basis": basis_mismatches,
                "automaton": automaton_mismatches,
                "metrics": metrics_mismatches,
                "decision": decision_mismatches,
                "training": training_mismatches,
            },
            "atol": 1e-12,
            "rtol": 1e-12,
        },
        "models": sorted(metrics_by_model),
        "artifact_files_checked": len(manifest_files),
    }
