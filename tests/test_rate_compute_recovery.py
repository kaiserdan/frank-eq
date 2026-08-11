import json
from pathlib import Path

from frank_eq.rate_compute.config import load_rate_compute_config
from frank_eq.rate_compute.workflow import (
    RATE_COMPUTE_ACCESS_CONTRACT,
    RATE_COMPUTE_RECOVERY_INPUTS,
    build_rate_compute_panels,
    recover_rate_compute_audit,
)
from frank_eq.utils import atomic_write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_recovery_reuses_hash_frozen_capture_without_model_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed-run"
    output = tmp_path / "recovered-run"
    source.mkdir()
    config_path = source / "config.yaml"
    config_path.write_text(
        (ROOT / "configs/rate_compute/real_olivia_rc0.yaml").read_text()
    )
    config = load_rate_compute_config(config_path)

    panels = build_rate_compute_panels(config)
    for n_entities, panel in panels.items():
        atomic_write_json(source / f"panels/n{n_entities}.json", panel.to_dict())
    atomic_write_json(source / "development_splits.json", {})
    atomic_write_json(source / "calibration.json", {"fit_split": "train"})
    atomic_write_json(
        source / "run_manifest.json",
        {
            "schema": "frank_eq_rate_compute_run_manifest_v1",
            "development_only": True,
            "stages": ["audit"],
            "access_contract": RATE_COMPUTE_ACCESS_CONTRACT,
            "config_sha256": sha256_file(config_path),
        },
    )
    atomic_write_json(
        source / "workflow_status.json",
        {
            "state": "failed",
            "current_stage": "audit",
            "completed_stages": [],
            "failure": {
                "type": "ValueError",
                "message": "too many values to unpack (expected 2)",
            },
        },
    )
    models = []
    for spec in config.models:
        models.append(
            {
                "model_id": spec.model_id,
                "revision_observed": spec.revision,
                "records": 1,
                "branch_execution": {
                    "mode": "kv_reuse",
                    "kv_cloned_response_branches": 1,
                    "exact_prefix_continuity_checks": 1,
                    "exact_replay_response_branches": 0,
                    "allow_exact_replay_fallback": False,
                    "configured_branch_batch_size": 8,
                    "max_observed_batch_size": 8,
                    "exclusive_cache_batching": True,
                },
            }
        )
    atomic_write_json(source / "models.json", models)
    (source / "records_raw.jsonl").write_text("{}\n{}\n")
    calibrated_rows = [
        {"calibrated_probability": 0.25, "prior_probability": 0.5},
        {"calibrated_probability": 0.75, "prior_probability": 0.5},
    ]
    (source / "records_calibrated.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in calibrated_rows)
    )

    recovery_input = {
        "schema": "frank_eq_rate_compute_recovery_input_v1",
        "source_cluster": "olivia",
        "source_job_name": "frank-eq-failed-audit",
        "source_slurm_job_id": "12345",
        "source_remote_run_root": str(source),
        "source_archive_sha256": "original-source-sha",
        "source_config_sha256": sha256_file(config_path),
        "files": {
            relative: sha256_file(source / relative)
            for relative in RATE_COMPUTE_RECOVERY_INPUTS
        },
    }
    recovery_manifest = tmp_path / "recovery_input.json"
    atomic_write_json(recovery_manifest, recovery_input)

    def fail_capture(*args, **kwargs):
        raise AssertionError("artifact-only recovery must not execute model capture")

    def fake_evaluate(records, observed_panels, observed_config):
        assert records == calibrated_rows
        assert sorted(observed_panels) == [4, 6]
        assert observed_config == config
        return (
            {"schema": "frank_eq_rate_compute_metrics_v1"},
            {
                "schema": "frank_eq_rate_compute_decision_v1",
                "status": "fail",
                "decision": "STOP_BEFORE_STAGEA_V3",
                "diagnosis": "BASIS_READOUT_NOT_QUALIFIED",
                "authorization": {
                    "stagea_registration_draft_authorized": False,
                    "stagea_outcome_run_authorized": False,
                    "claim_bearing_test_access_authorized": False,
                    "receiver_execution_authorized": False,
                    "scientific_claim_authorized": False,
                },
            },
            [{"compiled_probability": 0.5}],
            {"schema": "frank_eq_direct_protocol_selection_v1", "fit_split": "train"},
        )

    monkeypatch.setattr("frank_eq.rate_compute.workflow.capture_records", fail_capture)
    monkeypatch.setattr("frank_eq.rate_compute.workflow.evaluate_rate_compute", fake_evaluate)

    summary = recover_rate_compute_audit(
        config,
        config_path=config_path,
        source_run=source,
        recovery_manifest_path=recovery_manifest,
        recovery_manifest_sha256=sha256_file(recovery_manifest),
        output_dir=output,
    )

    assert summary["status"] == "completed"
    assert summary["artifact_only_recovery"] is True
    assert summary["model_capture_executed"] is False
    recovered_manifest = json.loads((output / "run_manifest.json").read_text())
    assert recovered_manifest["recovery"]["artifact_only"] is True
    assert recovered_manifest["recovery"]["model_capture_executed"] is False
    provenance = json.loads((output / "recovery_provenance.json").read_text())
    assert provenance["source_job_name"] == "frank-eq-failed-audit"
    assert provenance["post_calibration_outcomes_preexisting"] is False
    assert sha256_file(output / "records_raw.jsonl") == recovery_input["files"][
        "records_raw.jsonl"
    ]
