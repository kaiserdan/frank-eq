"""Independent artifact verification and recomputation for Stage-A v3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from frank_eq.utils import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

from .config import load_stagea_v3_config
from .evaluation import evaluate_stagea_v3
from .panel import V3Panel
from .predictions import load_prediction_bundle

_REQUIRED_FILES = (
    "config.yaml",
    "protocol.md",
    "registration.json",
    "dry_run_plan.json",
    "implementation_manifest.json",
    "run_manifest.json",
    "workflow_status.json",
    "access_ledger.json",
    "train_panel_manifest.json",
    "validation_panel_manifest.json",
    "freeze_manifest.json",
    "founder_checkpoints_manifest.json",
    "held_onboarding_manifest.json",
    "held_checkpoints_manifest.json",
    "test_panel_manifest.json",
    "models.json",
    "capture_validation.json",
    "compiler_checkpoints_manifest.json",
    "training_summary.json",
    "baseline_manifest.json",
    "predictions_manifest.json",
    "identity_train_basis_manifest.json",
    "rate_compute.json",
    "metrics.json",
    "decision.json",
    "run_summary.json",
    "artifact_manifest.json",
)

_RUNTIME_IMAGE_SHA256 = "a3ca46f0db9971b4108e5c8694e72f3039166383efc06b01f4031183208aa3b1"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _canonical_equal(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _validate_bound_manifest(root: Path, name: str, schema: str, config_hash: str) -> bool:
    path = root / name
    if not path.is_file():
        return False
    payload = _json(path)
    if (
        payload.get("schema") != schema
        or payload.get("status") != "frozen"
        or payload.get("config_sha256") != config_hash
    ):
        return False
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    return all(
        (root / relative).is_file() and sha256_file(root / relative) == expected
        for relative, expected in artifacts.items()
    )


def _derive_integrity(root: Path, config: Any) -> dict[str, bool]:
    access = _json(root / "access_ledger.json")
    models = _json(root / "models.json")
    checkpoints = _json(root / "compiler_checkpoints_manifest.json")
    baseline = _json(root / "baseline_manifest.json")
    rate_compute = _json(root / "rate_compute.json")
    summaries = [
        group["summary"] for model in models for group in model.get("capture_groups", {}).values()
    ]
    branch_valid = bool(summaries) and all(
        row.get("exact_replay_response_branches") == 0
        and row.get("allow_exact_replay_fallback") is False
        and row.get("exclusive_cache_batching") is True
        and row.get("primary_compiler_post_capture_source_queries") == 0
        and row.get("behavioral_probability_storage_epsilon") == 1e-7
        and isinstance(row.get("behavioral_probability_clamp_count"), int)
        and row.get("behavioral_probability_clamp_count") >= 0
        and row.get("logical_post_capture_source_queries")
        == row.get("kv_cloned_response_branches")
        == row.get("exact_prefix_continuity_checks")
        for row in summaries
    )
    expected_revisions = {model.model_id: model.revision for model in config.models}
    observed_revisions = {
        str(model.get("model_id")): model.get("revision_observed") for model in models
    }
    expected_seeds = config.section("compiler")["seeds"]
    checkpoint_models = checkpoints.get("models", {})
    checkpoint_valid = set(checkpoint_models) == set(expected_revisions)
    if checkpoint_valid:
        for registry in checkpoint_models.values():
            for key in ("semantic", "behavioral", "token_id", "final_token", "continuous"):
                entries = registry.get(key, [])
                if [
                    entry.get("metadata", {}).get("registered_seed") for entry in entries
                ] != expected_seeds:
                    checkpoint_valid = False
                if not all(
                    (root / entry["path"]).is_file()
                    and sha256_file(root / entry["path"]) == entry["sha256"]
                    for entry in entries
                ):
                    checkpoint_valid = False
    expected_test_files = {
        *(
            f"panels/test_n{entity_count}.json"
            for entity_count in config.section("panel")["entity_counts"]
        ),
        "test_panel_manifest.json",
        *(
            f"captures/test/{model.model_id}-n{entity_count}.pt"
            for model in config.models
            for entity_count in config.section("panel")["entity_counts"]
        ),
        *(
            f"predictions/{model.model_id}-n{entity_count}.{suffix}"
            for model in config.models
            for entity_count in config.section("panel")["entity_counts"]
            for suffix in ("npz", "json")
        ),
    }
    registered = set(access.get("registered_test_files", []))
    open_rows = access.get("test_file_opens", [])
    opened = {row.get("path") for row in open_rows}
    opened_hashes_valid = all(
        isinstance(row.get("path"), str)
        and (root / row["path"]).is_file()
        and sha256_file(root / row["path"]) == row.get("sha256")
        for row in open_rows
    )
    authorization = config.section("authorization")
    protected_closed = not any(
        authorization[key]
        for key in (
            "receiver_execution_authorized",
            "new_receiver_world_access_authorized",
            "scientific_claim_authorized",
            "paper_claim_authorized",
        )
    )
    return {
        "config_snapshot_hash": sha256_file(root / "config.yaml") == config.config_sha256,
        "exclusive_kv_and_prefix_continuity": branch_valid,
        "model_revisions_exact": observed_revisions == expected_revisions,
        "checkpoint_seed_registry_complete": checkpoint_valid,
        "test_access_consumed_once": access.get("test_access_count") == 1
        and access.get("current_stage") == "evaluate",
        "test_files_registered_and_opened": (
            registered == expected_test_files
            and opened == expected_test_files
            and opened_hashes_valid
        ),
        "founder_freeze_present": _validate_bound_manifest(
            root,
            "freeze_manifest.json",
            "frank_eq_stagea_v3_freeze_v1",
            config.config_sha256,
        ),
        "held_freeze_present": _validate_bound_manifest(
            root,
            "held_onboarding_manifest.json",
            "frank_eq_stagea_v3_held_onboarding_v1",
            config.config_sha256,
        ),
        "protected_authorizations_closed": protected_closed,
        "required_baselines_complete": baseline.get("required")
        == config.section("baselines")["required"]
        and baseline.get("conditions_complete") is True,
        "consumer_compute_declared": rate_compute.get("consumer_compute_declared") is True
        and rate_compute.get("framing_counted_separately") is True,
    }


def _load_identity_basis(root: Path) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    manifest = _json(root / "identity_train_basis_manifest.json")
    if (
        manifest.get("schema") != "frank_eq_stagea_v3_identity_train_basis_v1"
        or manifest.get("fit_role") != "train"
    ):
        raise ValueError("identity basis manifest has an invalid schema or fit role")
    result: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for relative, expected_hash in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"identity train-basis hash mismatch: {relative}")
        stem = path.stem
        model_id, entity = stem.rsplit("-n", 1)
        with np.load(path, allow_pickle=False) as loaded:
            result[(model_id, int(entity))] = {
                "probabilities": loaded["probabilities"],
                "world_ids": loaded["world_ids"],
                "renderer_ids": loaded["renderer_ids"],
            }
    return result


def verify_stagea_v3_run(
    run_root: str | Path,
    *,
    config_path: str | Path,
    write_audit: bool,
    require_existing_audit: bool,
) -> dict[str, Any]:
    """Verify hashes and recompute every metric/decision from prediction arrays."""

    root = Path(run_root).resolve()
    config = load_stagea_v3_config(config_path)
    missing = [relative for relative in _REQUIRED_FILES if not (root / relative).is_file()]
    if require_existing_audit and not (root / "independent_audit.json").is_file():
        missing.append("independent_audit.json")
    if missing:
        raise FileNotFoundError("Stage-A v3 run is missing: " + ", ".join(missing))

    artifact_manifest = _json(root / "artifact_manifest.json")
    required_manifest_entries = set(_REQUIRED_FILES) - {"artifact_manifest.json"}
    if require_existing_audit:
        required_manifest_entries.add("independent_audit.json")
    manifest_files = artifact_manifest.get("files", {})
    artifact_hashes_valid = (
        artifact_manifest.get("schema") == "frank_eq_stagea_v3_artifact_manifest_v1"
        and required_manifest_entries <= set(manifest_files)
        and all(
            (root / relative).is_file() and sha256_file(root / relative) == expected
            for relative, expected in manifest_files.items()
        )
    )
    run_manifest = _json(root / "run_manifest.json")
    workflow_status = _json(root / "workflow_status.json")
    dry_run_plan = _json(root / "dry_run_plan.json")
    plan_without_hash = dict(dry_run_plan)
    internal_plan_sha256 = plan_without_hash.pop("plan_sha256", None)
    implementation = _json(root / "implementation_manifest.json")
    implementation_files = implementation.get("files", {})
    environment = run_manifest.get("environment", {})
    plan_valid = (
        dry_run_plan.get("schema") == "frank_eq_stagea_v3_dry_run_plan_v1"
        and dry_run_plan.get("config_path") == "configs/stagea_v3/real_olivia_v3.yaml"
        and dry_run_plan.get("config_sha256") == config.config_sha256
        and internal_plan_sha256 == sha256_bytes(canonical_json_bytes(plan_without_hash))
        and run_manifest.get("inspected_plan_sha256") == internal_plan_sha256
        and run_manifest.get("dry_run_plan_sha256") == sha256_file(root / "dry_run_plan.json")
        and dry_run_plan.get("held_model_task_opened") is False
        and dry_run_plan.get("test_panel_instantiated") is False
    )
    implementation_valid = (
        implementation.get("schema") == "frank_eq_stagea_v3_implementation_manifest_v1"
        and implementation.get("config_sha256") == config.config_sha256
        and implementation.get("implementation_tree_sha256")
        == sha256_bytes(canonical_json_bytes(implementation_files))
        and run_manifest.get("implementation_manifest_sha256")
        == sha256_file(root / "implementation_manifest.json")
        and implementation.get("source_archive_sha256") == environment.get("source_sha256")
        and _is_sha256(environment.get("source_sha256"))
        and environment.get("cluster") == "olivia"
        and environment.get("git_dirty") == "false"
        and implementation.get("runtime_image_sha256") == _RUNTIME_IMAGE_SHA256
        and environment.get("runtime_image_sha256") == _RUNTIME_IMAGE_SHA256
    )
    config_valid = (
        sha256_file(root / "config.yaml") == config.config_sha256
        and run_manifest.get("config_sha256") == config.config_sha256
        and run_manifest.get("protocol_version") == config.protocol_version
        and run_manifest.get("stages")
        == ["prepare", "founder_fit", "freeze", "held_onboard", "evaluate"]
        and plan_valid
        and implementation_valid
    )
    workflow_valid = (
        workflow_status.get("state") == "completed"
        and workflow_status.get("completed_stages")
        == ["prepare", "founder_fit", "freeze", "held_onboard", "evaluate"]
        and workflow_status.get("test_access_consumed") is True
    )

    test_panel_manifest = _json(root / "test_panel_manifest.json")
    if test_panel_manifest.get("role") != "test":
        raise ValueError("test panel manifest does not carry the test role")
    panels = {
        entity_count: V3Panel.from_dict(_json(root / f"panels/test_n{entity_count}.json"))
        for entity_count in config.section("panel")["entity_counts"]
    }
    for entity_count, panel in panels.items():
        expected_hash = test_panel_manifest["files"][f"panels/test_n{entity_count}.json"]
        if sha256_file(root / f"panels/test_n{entity_count}.json") != expected_hash:
            raise ValueError("test panel manifest hash mismatch")
        if panel.role != "test":
            raise ValueError("loaded outcome panel does not carry the test role")
        panel_config = panel.panel.config
        expected_role = config.section("panel")["roles"]["test"]
        if (
            panel_config.get("world_seed") != expected_role["seed"]
            or len(panel.panel.worlds) != expected_role["worlds_per_complexity"]
            or panel_config.get("operation_seed") != config.section("panel")["operation_seed"]
        ):
            raise ValueError("test panel differs from the frozen role registration")

    prediction_manifest = _json(root / "predictions_manifest.json")
    if prediction_manifest.get("config_sha256") != config.config_sha256:
        raise ValueError("prediction manifest belongs to another config")
    bundles = []
    for key, entry in sorted(prediction_manifest.get("entries", {}).items()):
        bundle = load_prediction_bundle(
            root / entry["array"],
            root / entry["metadata"],
            config_sha256=config.config_sha256,
            expected_array_sha256=entry["array_sha256"],
            expected_metadata_sha256=entry["metadata_sha256"],
        )
        if key != f"{bundle.model_id}|{bundle.entity_count}":
            raise ValueError("prediction manifest group key differs from bundle identity")
        bundles.append(bundle)
    identity = _load_identity_basis(root)
    integrity = _derive_integrity(root, config)
    recomputed_metrics, recomputed_decision, recomputed_rate = evaluate_stagea_v3(
        config,
        bundles=bundles,
        panels=panels,
        train_identity_basis=identity,
        integrity_checks=integrity,
    )
    stored_metrics = _json(root / "metrics.json")
    stored_decision = _json(root / "decision.json")
    stored_rate = _json(root / "rate_compute.json")
    metrics_match = _canonical_equal(recomputed_metrics, stored_metrics)
    decision_match = _canonical_equal(recomputed_decision, stored_decision)
    rate_match = _canonical_equal(recomputed_rate, stored_rate)
    protected_closed = not any(
        stored_decision.get("authorization", {}).get(key, True)
        for key in (
            "receiver_execution_authorized",
            "new_receiver_world_access_authorized",
            "scientific_claim_authorized",
            "paper_claim_authorized",
        )
    )
    existing_audit_valid = True
    if require_existing_audit:
        existing = _json(root / "independent_audit.json")
        existing_audit_valid = (
            existing.get("schema") == "frank_eq_stagea_v3_independent_audit_v1"
            and existing.get("passed") is True
            and all(existing.get("checks", {}).values())
            and _canonical_equal(existing.get("integrity_checks"), integrity)
            and _canonical_equal(existing.get("decision"), recomputed_decision)
            and existing.get("prediction_groups") == len(bundles)
        )
    checks = {
        "required_files_present": not missing,
        "artifact_hashes_valid": artifact_hashes_valid,
        "config_and_stage_contract_valid": config_valid,
        "workflow_completed": workflow_valid,
        "integrity_checks_pass": all(integrity.values()),
        "metrics_recomputed_exactly": metrics_match,
        "decision_recomputed_exactly": decision_match,
        "rate_compute_recomputed_exactly": rate_match,
        "protected_authorizations_closed": protected_closed,
        "existing_independent_audit_valid": existing_audit_valid,
    }
    result = {
        "schema": "frank_eq_stagea_v3_independent_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "integrity_checks": integrity,
        "decision": recomputed_decision,
        "prediction_groups": len(bundles),
        "artifact_files_checked": len(artifact_manifest.get("files", {})),
        "test_artifacts_verified": {
            relative: sha256_file(root / relative)
            for relative in sorted(_json(root / "access_ledger.json")["registered_test_files"])
        },
    }
    if write_audit:
        atomic_write_json(root / "independent_audit.json", result)
    return result
