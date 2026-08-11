#!/usr/bin/env python3
"""Verify a fetched rate--compute run independently of scheduler success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frank_eq.utils import sha256_file


def verify_run(root: str | Path) -> dict[str, object]:
    source = Path(root)
    required = (
        "run_manifest.json",
        "workflow_status.json",
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

    workflow = json.loads((source / "workflow_status.json").read_text())
    if workflow.get("state") != "completed" or workflow.get("completed_stages") != ["audit"]:
        failures.append("workflow did not complete exactly the audit stage")
    manifest = json.loads((source / "artifact_manifest.json").read_text())
    observed = {name: sha256_file(source / name) for name in manifest.get("files", {})}
    if observed != manifest.get("files", {}):
        failures.append("artifact hash manifest mismatch")
    decision = json.loads((source / "decision.json").read_text())
    authorization = decision.get("authorization", {})
    prohibited = (
        "stagea_outcome_run_authorized",
        "claim_bearing_test_access_authorized",
        "receiver_execution_authorized",
        "scientific_claim_authorized",
    )
    if any(authorization.get(key) is not False for key in prohibited):
        failures.append("development decision improperly authorizes a protected role")
    return {
        "schema": "frank_eq_rate_compute_verification_v1",
        "overall": "passed" if not failures else "failed",
        "root": str(source),
        "workflow": workflow,
        "decision": decision.get("decision"),
        "diagnosis": decision.get("diagnosis"),
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
