#!/usr/bin/env python3
"""Verify a fetched rate--compute run independently of scheduler success."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from frank_eq.data.real_panel import RealPanel
from frank_eq.rate_compute.config import load_rate_compute_config
from frank_eq.utils import sha256_file

EXPECTED_REVISIONS = {
    "qwen3-4b": "1cfa9a7208912126459214e8b04321603b3df60c",
    "qwen3-8b": "b968826d9c46dd6066d109eabc6255188de91218",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL in {path.name} at line {line_number}") from error


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool | str) or value is None:
        return True
    if isinstance(value, int | float):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return False


def _append_once(failures: list[str], message: str) -> None:
    if message not in failures:
        failures.append(message)


def verify_run(root: str | Path) -> dict[str, object]:
    source = Path(root)
    required = (
        "config.yaml",
        "run_manifest.json",
        "workflow_status.json",
        "development_splits.json",
        "panels/n4.json",
        "panels/n6.json",
        "models.json",
        "records_raw.jsonl",
        "calibration.json",
        "records_calibrated.jsonl",
        "compiled_predictions.jsonl",
        "direct_protocol_selection.json",
        "metrics.json",
        "decision.json",
        "artifact_manifest.json",
        "run_summary.json",
    )
    failures = [f"missing {name}" for name in required if not (source / name).is_file()]
    if failures:
        return {
            "schema": "frank_eq_rate_compute_verification_v1",
            "overall": "failed",
            "failures": failures,
        }

    workflow = _read_json(source / "workflow_status.json")
    if workflow.get("state") != "completed" or workflow.get("completed_stages") != ["audit"]:
        failures.append("workflow did not complete exactly the audit stage")

    run_manifest = _read_json(source / "run_manifest.json")
    if run_manifest.get("development_only") is not True or run_manifest.get("stages") != ["audit"]:
        failures.append("run manifest is not the development-only audit contract")
    access = run_manifest.get("access_contract", {})
    expected_access = {
        "state_precedes_operation_reveal": True,
        "literal_kv_reuse": True,
        "exact_replay_fallback": False,
        "test_worlds_available": False,
        "held_sender_loaded": False,
        "receiver_tensors_available": False,
        "claim_bearing_role": False,
    }
    if access != expected_access:
        failures.append("run manifest access contract differs from frozen RC0")
    if run_manifest.get("config_snapshot") != "config.yaml":
        failures.append("run manifest does not identify the fetched config snapshot")
    if sha256_file(source / "config.yaml") != run_manifest.get("config_sha256"):
        failures.append("fetched config hash differs from run manifest")

    recovery = run_manifest.get("recovery")
    recovery_provenance = None
    if recovery is not None:
        provenance_name = recovery.get("recovery_provenance")
        provenance_path = source / str(provenance_name)
        if (
            recovery.get("artifact_only") is not True
            or recovery.get("model_capture_executed") is not False
            or provenance_name != "recovery_provenance.json"
            or not provenance_path.is_file()
        ):
            failures.append("artifact-only recovery provenance is incomplete")
        else:
            recovery_provenance = _read_json(provenance_path)
            if (
                sha256_file(provenance_path)
                != recovery.get("recovery_provenance_sha256")
                or recovery_provenance.get("schema")
                != "frank_eq_rate_compute_recovery_provenance_v1"
                or recovery_provenance.get("capture_reused") is not True
                or recovery_provenance.get("capture_executed_in_recovery") is not False
                or recovery_provenance.get("calibration_reused") is not True
                or recovery_provenance.get("post_calibration_outcomes_preexisting")
                is not False
                or recovery_provenance.get("source_workflow_state") != "failed"
                or recovery_provenance.get("recovery_input_sha256")
                != recovery.get("recovery_input_sha256")
            ):
                failures.append("artifact-only recovery provenance is invalid")
            source_hashes = recovery_provenance.get("source_files", {})
            copied_inputs = (
                "config.yaml",
                "development_splits.json",
                "panels/n4.json",
                "panels/n6.json",
                "models.json",
                "records_raw.jsonl",
                "calibration.json",
                "records_calibrated.jsonl",
            )
            if any(
                source_hashes.get(relative) != sha256_file(source / relative)
                for relative in copied_inputs
            ):
                failures.append("recovered capture artifacts differ from the frozen source")

    try:
        config = load_rate_compute_config(source / "config.yaml")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        failures.append(f"fetched config is invalid: {error}")
        config = None

    manifest = _read_json(source / "artifact_manifest.json")
    registered_files = manifest.get("files", {})
    expected_hashed = set(required) - {"artifact_manifest.json"}
    if recovery is not None:
        expected_hashed.add("recovery_provenance.json")
    if not expected_hashed.issubset(registered_files):
        failures.append("artifact manifest does not cover every required run artifact")
    observed = {name: sha256_file(source / name) for name in registered_files}
    if observed != registered_files:
        failures.append("artifact hash manifest mismatch")

    panels: dict[int, RealPanel] = {}
    for n_entities in (4, 6):
        try:
            panel = RealPanel.from_dict(_read_json(source / f"panels/n{n_entities}.json"))
            if panel.n_entities != n_entities:
                failures.append(f"n{n_entities} panel reports the wrong entity count")
            panels[n_entities] = panel
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"n{n_entities} panel is invalid: {error}")

    splits = _read_json(source / "development_splits.json")
    split_by_world: dict[tuple[int, int], str] = {}
    for n_entities in (4, 6):
        payload = splits.get(str(n_entities), {})
        train = {int(value) for value in payload.get("train_world_ids", [])}
        validation = {int(value) for value in payload.get("validation_world_ids", [])}
        test = payload.get("test_world_ids", [])
        expected_worlds = set(range(96))
        if train & validation or train | validation != expected_worlds or test:
            failures.append(f"n{n_entities} development split is incomplete or exposes test worlds")
        for world_id in train:
            split_by_world[(n_entities, world_id)] = "train"
        for world_id in validation:
            split_by_world[(n_entities, world_id)] = "validation"

    models = _read_json(source / "models.json")
    model_ids = [str(item.get("model_id")) for item in models]
    if model_ids != list(EXPECTED_REVISIONS):
        failures.append("model roster differs from frozen Qwen3-4B/Qwen3-8B order")
    metadata_record_count = 0
    for item in models:
        model_id = str(item.get("model_id"))
        expected_revision = EXPECTED_REVISIONS.get(model_id)
        if (
            expected_revision is None
            or item.get("revision_requested") != expected_revision
            or item.get("revision_observed") != expected_revision
        ):
            failures.append(f"{model_id} requested/observed revision is not the frozen snapshot")
        answer_ids = item.get("answer_token_ids", [])
        candidates = item.get("semantic_candidates", {})
        if len(answer_ids) != 2 or len(set(answer_ids)) != 2:
            failures.append(f"{model_id} answer-token IDs are missing or invalid")
        for label in ("false", "true"):
            token_ids = candidates.get(label, {}).get("token_ids", [])
            if not token_ids or not all(isinstance(value, int) for value in token_ids):
                failures.append(f"{model_id} semantic {label} candidate IDs are missing")
        branch = item.get("branch_execution", {})
        records = int(item.get("records", -1))
        metadata_record_count += max(records, 0)
        if (
            branch.get("mode") != "kv_reuse"
            or branch.get("allow_exact_replay_fallback") is not False
            or branch.get("exact_replay_response_branches") != 0
            or branch.get("kv_cloned_response_branches") != records
            or branch.get("exact_prefix_continuity_checks") != records
            or branch.get("configured_branch_batch_size") != 8
            or branch.get("exclusive_cache_batching") is not True
            or int(branch.get("response_batches", 0)) < 1
            or int(branch.get("response_batches", records + 1)) > records
            or branch.get("max_observed_batch_size") != 8
        ):
            failures.append(f"{model_id} branch accounting violates exclusive cloned-KV RC0")

    coverage: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    basis_slots: dict[tuple[str, int, int, int], set[int]] = defaultdict(set)
    target_protocols: dict[tuple[str, int, int, int, int], set[str]] = defaultdict(set)
    raw_record_count = 0
    for row in _read_jsonl(source / "records_raw.jsonl"):
        raw_record_count += 1
        model_id = str(row.get("model_id"))
        n_entities = int(row.get("entity_count", -1))
        panel_world_id = int(row.get("panel_world_id", -1))
        renderer_id = int(row.get("renderer_id", -1))
        protocol = str(row.get("protocol"))
        if model_id not in EXPECTED_REVISIONS or n_entities not in {4, 6}:
            _append_once(failures, "response rows contain an unregistered model or complexity")
            continue
        if row.get("split") != split_by_world.get((n_entities, panel_world_id)):
            _append_once(failures, "response row split differs from development split manifest")
        if renderer_id not in {0, 1}:
            _append_once(failures, "response rows contain an unregistered renderer")
        coverage[(model_id, n_entities, panel_world_id)].add(renderer_id)
        generated = int(row.get("generated_token_count", -1))
        expected_generated = 32 if protocol in {"reason", "pause"} else 0
        if generated != expected_generated:
            _append_once(failures, "response rows violate registered generated-token budgets")
        if not _finite_tree(
            {
                "probability_true": row.get("probability_true"),
                "log_odds_score": row.get("log_odds_score"),
                "truth": row.get("truth"),
            }
        ):
            _append_once(failures, "response rows contain non-finite scores")
        if row.get("kind") == "basis":
            if protocol != "sequence":
                _append_once(failures, "basis rows do not use semantic sequence likelihood")
            basis_slots[(model_id, n_entities, panel_world_id, renderer_id)].add(
                int(row.get("item_id", -1))
            )
        elif row.get("kind") == "target":
            target_protocols[
                (
                    model_id,
                    n_entities,
                    panel_world_id,
                    renderer_id,
                    int(row.get("operation_id", -1)),
                )
            ].add(protocol)
        else:
            _append_once(failures, "response rows contain an unknown record kind")

    if raw_record_count != metadata_record_count:
        failures.append("response row count differs from model branch accounting")
    if config is not None:
        if [model.model_id for model in config.models] != list(EXPECTED_REVISIONS):
            failures.append("fetched config model roster differs from frozen RC0")
        for model_id in EXPECTED_REVISIONS:
            for n_entities, panel in panels.items():
                expected_slots = set(range(n_entities * (n_entities - 1)))
                for world in panel.worlds:
                    key = (model_id, n_entities, world.world_id)
                    if coverage[key] != {0, 1}:
                        failures.append(f"missing renderer coverage for {key}")
                    for renderer_id in (0, 1):
                        basis_key = (*key, renderer_id)
                        if basis_slots[basis_key] != expected_slots:
                            failures.append(f"incomplete public basis for {basis_key}")
                        for operation in panel.operations:
                            family = operation.definition.family
                            expected_protocols = {"answer_token", "sequence"}
                            if family in config.protocols.compute_families:
                                expected_protocols.update({"reason", "pause"})
                            target_key = (*key, renderer_id, operation.definition.operation_id)
                            if target_protocols[target_key] != expected_protocols:
                                failures.append(f"incomplete target protocol set for {target_key}")

    calibrated_count = 0
    for row in _read_jsonl(source / "records_calibrated.jsonl"):
        calibrated_count += 1
        probability = row.get("calibrated_probability")
        prior = row.get("prior_probability")
        if (
            not _finite_tree({"calibrated": probability, "prior": prior})
            or not 0.0 < float(probability) < 1.0
            or not 0.0 < float(prior) < 1.0
        ):
            _append_once(failures, "calibrated response rows contain invalid probabilities")
    if calibrated_count != raw_record_count:
        failures.append("raw and calibrated response row counts differ")

    calibration = _read_json(source / "calibration.json")
    direct_selection = _read_json(source / "direct_protocol_selection.json")
    if calibration.get("fit_split") != "train":
        failures.append("calibration was not frozen on training worlds only")
    if direct_selection.get("fit_split") != "train":
        failures.append("direct protocol selection was not frozen on training worlds only")

    metrics = _read_json(source / "metrics.json")
    if not _finite_tree(metrics):
        failures.append("metrics contain non-finite values")
    decision = _read_json(source / "decision.json")
    authorization = decision.get("authorization", {})
    prohibited = (
        "stagea_outcome_run_authorized",
        "claim_bearing_test_access_authorized",
        "receiver_execution_authorized",
        "scientific_claim_authorized",
    )
    if any(authorization.get(key) is not False for key in prohibited):
        failures.append("development decision improperly authorizes a protected role")
    expected_draft = decision.get("status") == "pass"
    if authorization.get("stagea_registration_draft_authorized") is not expected_draft:
        failures.append("Stage-A draft authorization does not match the RC0 machine status")

    summary = _read_json(source / "run_summary.json")
    if (
        summary.get("workflow_integrity_passed") is not True
        or summary.get("records") != raw_record_count
        or summary.get("authorizes_scientific_claim") is not False
    ):
        failures.append("run summary does not preserve workflow or authorization integrity")
    if recovery is not None and (
        summary.get("artifact_only_recovery") is not True
        or summary.get("model_capture_executed") is not False
    ):
        failures.append("run summary does not preserve artifact-only recovery semantics")
    return {
        "schema": "frank_eq_rate_compute_verification_v1",
        "overall": "passed" if not failures else "failed",
        "root": str(source),
        "workflow": workflow,
        "decision": decision.get("decision"),
        "diagnosis": decision.get("diagnosis"),
        "models": model_ids,
        "records": raw_record_count,
        "panels": sorted(panels),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    result = verify_run(args.run)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["overall"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
