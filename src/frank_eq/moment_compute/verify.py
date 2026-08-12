"""Independent hash and metric verification for Stage M0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from frank_eq.data.real_panel import RealPanel
from frank_eq.utils import atomic_write_json, canonical_json_bytes, sha256_file

from .config import load_moment_compute_config
from .events import EventRegistry
from .workflow import _read_jsonl, compile_validation_predictions, evaluate_moment_compute

_REQUIRED_FILES = (
    "config.yaml",
    "run_manifest.json",
    "workflow_status.json",
    "development_splits.json",
    "panels/n4.json",
    "panels/n6.json",
    "event_registry.json",
    "models.json",
    "records_raw.jsonl",
    "calibration.json",
    "records_calibrated.jsonl",
    "compiled_predictions.jsonl",
    "direct_protocol_selection.json",
    "metrics.json",
    "decision.json",
    "run_summary.json",
    "artifact_manifest.json",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _canonical_equal(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def verify_moment_compute_run(
    run_root: str | Path,
    *,
    config_path: str | Path,
    write_verification: bool,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    missing = [relative for relative in _REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError("Stage M0 run is missing: " + ", ".join(missing))
    config = load_moment_compute_config(config_path)
    manifest = _json(root / "artifact_manifest.json")
    files = manifest.get("files", {})
    artifact_hashes_valid = (
        manifest.get("schema") == "frank_eq_moment_compute_artifact_manifest_v1"
        and set(files) == set(_REQUIRED_FILES) - {"artifact_manifest.json"}
        and all(
            (root / relative).is_file() and sha256_file(root / relative) == expected
            for relative, expected in files.items()
        )
    )
    status = _json(root / "workflow_status.json")
    run_manifest = _json(root / "run_manifest.json")
    roles = _json(root / "development_splits.json")
    protected_roles_valid = (
        roles.get("test_world_ids") == []
        and set(roles.get("calibration_world_ids", [])).isdisjoint(
            roles.get("selection_world_ids", [])
        )
        and set(roles.get("calibration_world_ids", [])).isdisjoint(
            roles.get("validation_world_ids", [])
        )
        and set(roles.get("selection_world_ids", [])).isdisjoint(
            roles.get("validation_world_ids", [])
        )
    )
    workflow_valid = (
        status.get("state") == "completed"
        and status.get("completed_stages") == ["audit"]
        and run_manifest.get("development_only") is True
        and run_manifest.get("access_contract", {}).get(
            "claim_bearing_test_worlds_available"
        )
        is False
    )
    panel = RealPanel.from_dict(_json(root / "panels/n4.json"))
    registry = EventRegistry.from_dict(_json(root / "event_registry.json"))
    records = _read_jsonl(root / "records_calibrated.jsonl")
    selection = _json(root / "direct_protocol_selection.json")
    predictions = compile_validation_predictions(
        records,
        panel,
        registry,
        selection,
        config,
    )
    stored_predictions = _read_jsonl(root / "compiled_predictions.jsonl")
    recomputed_metrics, recomputed_decision = evaluate_moment_compute(
        records,
        predictions,
        panel,
        registry,
        config,
    )
    stored_metrics = _json(root / "metrics.json")
    stored_decision = _json(root / "decision.json")
    protected_closed = not any(
        stored_decision.get("authorization", {}).get(key, True)
        for key in (
            "one_shot_compiler_run_authorized",
            "held_sender_authorized",
            "claim_bearing_test_authorized",
            "receiver_protocol_draft_authorized",
            "receiver_execution_authorized",
            "scientific_claim_authorized",
            "paper_claim_authorized",
        )
    )
    checks = {
        "required_files_present": not missing,
        "artifact_hashes_valid": artifact_hashes_valid,
        "workflow_completed": workflow_valid,
        "development_roles_disjoint": protected_roles_valid,
        "event_registry_hash_valid": True,
        "predictions_recomputed_exactly": _canonical_equal(predictions, stored_predictions),
        "metrics_recomputed_exactly": _canonical_equal(recomputed_metrics, stored_metrics),
        "decision_recomputed_exactly": _canonical_equal(recomputed_decision, stored_decision),
        "protected_authorizations_closed": protected_closed,
    }
    result = {
        "schema": "frank_eq_moment_compute_independent_verification_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "decision": recomputed_decision,
        "event_coordinates": len(registry.events),
        "records": len(records),
        "compiled_predictions": len(predictions),
        "artifact_files_checked": len(files),
    }
    if write_verification:
        atomic_write_json(root / "independent_verification.json", result)
    return result
