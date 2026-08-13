"""Independent artifact and reducer verification for SPQ0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from frank_eq.utils import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

from .capture import load_capture
from .config import load_spq0_config
from .evaluation import deterministic_prediction_digest, evaluate_all_models, gate_decision
from .panel import build_panels
from .workflow import build_spq0_plan

_REQUIRED = (
    "config.yaml",
    "protocol.md",
    "registration.json",
    "dry_run_plan.json",
    "run_manifest.json",
    "workflow_status.json",
    "checkpoint_preflight.json",
    "systems.json",
    "public_basis.json",
    "panel_manifest.json",
    "panels/calibration.json",
    "panels/selection.json",
    "panels/validation.json",
    "models.json",
    "capture_manifest.json",
    "training_summary.json",
    "checkpoints_manifest.json",
    "predictions_manifest.json",
    "metrics.json",
    "decision.json",
    "rate_compute.json",
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
    return not mismatches, maximum, mismatches[:100]


def _array_groups_valid(
    root: Path,
    manifest_name: str,
    expected: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[bool, float, list[str]]:
    manifest = _json(root / manifest_name)
    entries = manifest.get("entries", {})
    if set(entries) != set(expected):
        return False, float("inf"), ["group registry"]
    maximum = 0.0
    mismatches: list[str] = []
    for group in sorted(expected):
        entry = entries[group]
        path = root / str(entry.get("path", ""))
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            mismatches.append(f"{group}.hash")
            continue
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(expected[group]):
                mismatches.append(f"{group}.arrays")
                continue
            for name in sorted(expected[group]):
                observed = loaded[name]
                wanted = np.asarray(expected[group][name])
                if observed.shape != wanted.shape:
                    mismatches.append(f"{group}.{name}.shape")
                    continue
                if np.issubdtype(observed.dtype, np.number):
                    difference = (
                        float(np.max(np.abs(observed.astype(np.float64) - wanted.astype(np.float64))))
                        if observed.size
                        else 0.0
                    )
                    maximum = max(maximum, difference)
                    if not np.allclose(observed, wanted, atol=1e-12, rtol=1e-12):
                        mismatches.append(f"{group}.{name}")
                elif not np.array_equal(observed, wanted):
                    mismatches.append(f"{group}.{name}")
    return not mismatches, maximum, mismatches[:100]


def _checkpoint_preflight_valid(
    receipt: Mapping[str, Any],
    config: Any,
) -> bool:
    expected_active = {model.model_id: model for model in config.models}
    active = receipt.get("active", {})
    if set(active) != set(expected_active):
        return False
    for model_id, entry in active.items():
        model = expected_active[model_id]
        if (
            entry.get("hf_id") != model.hf_id
            or entry.get("family") != model.family
            or entry.get("revision_requested") != model.revision
            or entry.get("revision_resolved") != model.revision
            or entry.get("model_loaded") is not False
            or entry.get("inference_executed") is not False
        ):
            return False
        snapshot = Path(str(entry.get("snapshot", "")))
        files = entry.get("files", {})
        if not snapshot.is_dir() or not files:
            return False
        for relative, file_entry in files.items():
            path = snapshot / relative
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not path.is_file()
                or path.stat().st_size != file_entry.get("bytes")
                or sha256_file(path) != file_entry.get("sha256")
            ):
                return False
        if (
            entry.get("file_count") != len(files)
            or entry.get("total_bytes")
            != sum(int(file_entry["bytes"]) for file_entry in files.values())
            or entry.get("snapshot_content_sha256")
            != sha256_bytes(canonical_json_bytes(files))
        ):
            return False
    if (
        receipt.get("status") != "passed"
        or receipt.get("local_files_only") is not True
        or receipt.get("active_snapshot_content_sha256")
        != sha256_bytes(canonical_json_bytes(active))
    ):
        return False
    reserved = receipt.get("reserved_unopened", [])
    expected_reserved = {
        model.model_id: (model.hf_id, model.family, model.revision)
        for model in config.reserved_unopened_models
    }
    observed_reserved = {
        row.get("model_id"): (
            row.get("hf_id"),
            row.get("family"),
            row.get("revision"),
        )
        for row in reserved
    }
    if observed_reserved != expected_reserved:
        return False
    if any(
        row.get("access") != "reserved_unopened"
        or row.get("snapshot_resolution_attempted") is not False
        or row.get("files_opened") != 0
        or row.get("model_adapter_instantiated") is not False
        or row.get("model_loaded") is not False
        or row.get("inference_executed") is not False
        for row in reserved
    ):
        return False
    return all(
        receipt.get(key) == 0
        for key in (
            "reserved_snapshot_resolution_attempts",
            "reserved_files_opened",
            "reserved_model_adapter_instantiations",
            "reserved_model_loads",
            "reserved_inference_calls",
        )
    )


def verify_spq0_run(
    run_root: str | Path,
    *,
    config_path: str | Path,
    write_verification: bool = False,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    missing = [relative for relative in _REQUIRED if not (root / relative).is_file()]
    if missing:
        result = {
            "schema": "frank_eq_spq0_verification_v1",
            "passed": False,
            "checks": {"required_files_present": False},
            "missing": missing,
        }
        if write_verification:
            atomic_write_json(root / "independent_verification.json", result)
        return result

    config = load_spq0_config(config_path)
    stored_config = load_spq0_config(root / "config.yaml")
    config_match = config.as_dict() == stored_config.as_dict()
    expected_plan = build_spq0_plan(config, config_path=config_path)
    stored_plan = _json(root / "dry_run_plan.json")
    plan_match = canonical_json_bytes(expected_plan) == canonical_json_bytes(stored_plan)

    manifest = _json(root / "artifact_manifest.json")
    manifest_files = manifest.get("files", {})
    safe_paths = all(
        not Path(relative).is_absolute() and ".." not in Path(relative).parts
        for relative in manifest_files
    )
    artifact_hashes = (
        manifest.get("schema") == "frank_eq_spq0_artifact_manifest_v1"
        and safe_paths
        and set(_REQUIRED) - {"artifact_manifest.json"} <= set(manifest_files)
        and all(
            (root / relative).is_file()
            and sha256_file(root / relative) == expected
            for relative, expected in manifest_files.items()
        )
    )

    run_manifest = _json(root / "run_manifest.json")
    workflow = _json(root / "workflow_status.json")
    workflow_valid = (
        run_manifest.get("development_only") is True
        and run_manifest.get("stages") == ["audit"]
        and run_manifest.get("plan_sha256") == stored_plan.get("plan_sha256")
        and run_manifest.get("environment", {}).get("runtime_image_sha256")
        == stored_plan.get("runtime", {}).get("image_sha256")
        and run_manifest.get("environment", {}).get("git_dirty") == "false"
        and run_manifest.get("access_contract", {}).get("state_precedes_future_test")
        is True
        and run_manifest.get("access_contract", {}).get("literal_cloned_kv_reuse")
        is True
        and run_manifest.get("access_contract", {}).get("exact_replay_fallback")
        is False
        and workflow.get("state") == "completed"
        and workflow.get("completed_stages") == ["checkpoint_preflight", "audit"]
    )

    checkpoint_preflight = _json(root / "checkpoint_preflight.json")
    checkpoint_valid = _checkpoint_preflight_valid(checkpoint_preflight, config)
    systems, basis = config.build_systems_and_basis()
    systems_payload = {
        "schema": "frank_eq_spq0_system_family_v1",
        "systems": [system.to_dict() for system in systems],
    }
    systems_match = canonical_json_bytes(systems_payload) == canonical_json_bytes(
        _json(root / "systems.json")
    )
    basis_match, basis_delta, basis_mismatches = _numeric_equal(
        basis.to_dict(), _json(root / "public_basis.json")
    )
    panels = build_panels(config, systems, basis)
    panel_match = all(
        canonical_json_bytes(panel.to_dict())
        == canonical_json_bytes(_json(root / "panels" / f"{role}.json"))
        for role, panel in panels.items()
    )
    role_sets = [
        {history.history_id for history in panels[role].histories}
        for role in ("calibration", "selection", "validation")
    ]
    role_separation = (
        not role_sets[0] & role_sets[1]
        and not role_sets[0] & role_sets[2]
        and not role_sets[1] & role_sets[2]
        and any(
            history.system_role == "validation_only"
            for history in panels["validation"].histories
        )
        and all(
            history.system_role == "fit"
            for role in ("calibration", "selection")
            for history in panels[role].histories
        )
    )

    capture_manifest = _json(root / "capture_manifest.json")
    entries = capture_manifest.get("entries", {})
    expected_models = {model.model_id: model for model in config.models}
    captures_valid = set(entries) == set(expected_models)
    captures: dict[str, tuple[Mapping[str, np.ndarray], Mapping[str, Any]]] = {}
    if captures_valid:
        for model_id, entry in sorted(entries.items()):
            try:
                arrays, metadata = load_capture(root, entry)
            except (FileNotFoundError, ValueError):
                captures_valid = False
                break
            model = expected_models[model_id]
            expected_rows = int(stored_plan["capture"]["prefixes_per_model"])
            expected_branches = int(
                stored_plan["capture"]["post_reveal_query_branches_per_model"]
            )
            captures_valid = captures_valid and (
                metadata.get("model_id") == model_id
                and metadata.get("family") == model.family
                and metadata.get("revision_requested") == model.revision
                and metadata.get("revision_observed") == model.revision
                and metadata.get("rows") == expected_rows
                and metadata.get("response_branches") == expected_branches
                and metadata.get("exact_prefix_continuity_checks") == expected_branches
                and metadata.get("exact_event_boundary_checks", 0) > expected_rows
                and metadata.get("branch_execution", {}).get("literal_kv_reuse") is True
                and metadata.get("branch_execution", {}).get("exclusive_cache_batching")
                is True
                and metadata.get("branch_execution", {}).get(
                    "exact_replay_response_branches"
                )
                == 0
                and metadata.get("selected_kv_surface_enabled") is False
                and set(np.unique(arrays["role_ids"]).tolist()) == {0, 1, 2}
                and set(np.unique(arrays["system_role_ids"]).tolist()) == {0, 1}
            )
            captures[model_id] = (arrays, metadata)

    recomputation_error: str | None = None
    recomputed_metrics: dict[str, Any] = {}
    recomputed_training: dict[str, Any] = {}
    recomputed_predictions: dict[str, dict[str, np.ndarray]] = {}
    recomputed_checkpoints: dict[str, dict[str, np.ndarray]] = {}
    recomputed_decision: dict[str, Any] = {}
    if captures_valid:
        try:
            (
                recomputed_metrics,
                recomputed_training,
                recomputed_predictions,
                recomputed_checkpoints,
            ) = evaluate_all_models(config, systems, basis, captures)
            recomputed_metrics["prediction_digest_sha256"] = deterministic_prediction_digest(
                recomputed_predictions
            )
            recomputed_decision = gate_decision(config, recomputed_metrics)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            recomputation_error = f"{type(error).__name__}: {error}"

    metrics_match, metrics_delta, metrics_mismatches = _numeric_equal(
        recomputed_metrics, _json(root / "metrics.json")
    )
    training_match, training_delta, training_mismatches = _numeric_equal(
        recomputed_training, _json(root / "training_summary.json")
    )
    decision_match = canonical_json_bytes(recomputed_decision) == canonical_json_bytes(
        _json(root / "decision.json")
    )
    predictions_match, predictions_delta, prediction_mismatches = _array_groups_valid(
        root,
        "predictions_manifest.json",
        recomputed_predictions,
    )
    checkpoints_match, checkpoints_delta, checkpoint_mismatches = _array_groups_valid(
        root,
        "checkpoints_manifest.json",
        recomputed_checkpoints,
    )
    decision = _json(root / "decision.json")
    protected = decision.get("authorization", {})
    protected_closed = not any(
        protected.get(key, True)
        for key in (
            "spq1_execution_authorized",
            "held_sender_access_authorized",
            "claim_bearing_test_access_authorized",
            "receiver_execution_authorized",
            "scientific_claim_authorized",
            "paper_claim_authorized",
        )
    )
    models_payload = _json(root / "models.json")
    reserved_models_absent = (
        {row["model_id"] for row in models_payload.get("active_founders", [])}
        == set(expected_models)
        and not (
            {row["model_id"] for row in models_payload.get("reserved_unopened", [])}
            & set(entries)
        )
    )
    rate = _json(root / "rate_compute.json")
    rate_valid = (
        rate.get("primary_packet", {}).get("rank") == basis.exact_rank
        and rate.get("primary_packet", {}).get("payload_bits") == 4 * basis.exact_rank
        and rate.get("primary_packet", {}).get("source_post_capture_queries") == 0
        and rate.get("primary_amortized_query_count") == 16
        and rate.get("one_direct_query_is_not_a_conjunctive_gate") is True
    )
    checks = {
        "required_files_present": not missing,
        "config_snapshot_matches": config_match,
        "inspected_plan_recomputed_exactly": plan_match,
        "artifact_hashes_valid": artifact_hashes,
        "workflow_and_runtime_identity_valid": workflow_valid,
        "active_checkpoint_files_rehashed": checkpoint_valid,
        "reserved_checkpoints_never_accessed": checkpoint_valid and reserved_models_absent,
        "systems_recomputed_exactly": systems_match,
        "public_basis_recomputed": basis_match,
        "panels_recomputed_exactly": panel_match,
        "roles_and_validation_only_system_separated": role_separation,
        "captures_complete_pinned_and_causal": captures_valid,
        "probe_and_reader_training_recomputed": training_match,
        "predictions_recomputed": predictions_match,
        "checkpoint_arrays_recomputed": checkpoints_match,
        "metrics_recomputed": metrics_match,
        "decision_recomputed_exactly": decision_match,
        "rate_compute_contract_valid": rate_valid,
        "protected_authorizations_closed": protected_closed,
    }
    result = {
        "schema": "frank_eq_spq0_verification_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "diagnosis": recomputed_decision.get("diagnosis"),
        "decision": recomputed_decision,
        "recomputation_error": recomputation_error,
        "numeric_comparison": {
            "maximum_abs_delta": max(
                basis_delta,
                metrics_delta,
                training_delta,
                predictions_delta,
                checkpoints_delta,
            ),
            "atol": 1e-12,
            "rtol": 1e-12,
            "mismatches": {
                "basis": basis_mismatches,
                "training": training_mismatches,
                "metrics": metrics_mismatches,
                "predictions": prediction_mismatches,
                "checkpoints": checkpoint_mismatches,
            },
        },
        "models": sorted(captures),
        "artifact_files_checked": len(manifest_files),
        "reserved_models": [model.model_id for model in config.reserved_unopened_models],
    }
    if write_verification:
        atomic_write_json(root / "independent_verification.json", result)
    return result
