#!/usr/bin/env python3
"""Validate repository contracts, configs, docs, cluster surfaces, and adopted evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from frank_eq.config import load_config  # noqa: E402
from frank_eq.real_config import load_real_config  # noqa: E402
from frank_eq.utils import sha256_file  # noqa: E402

REQUIRED = (
    "README.md",
    "AGENTS.md",
    "HANDOFF.md",
    "docs/00_PROJECT_CHARTER.md",
    "docs/01_SCIENTIFIC_HYPOTHESIS.md",
    "docs/02_ARCHITECTURE.md",
    "docs/03_INFORMATION_ACCESS_CONTRACT.md",
    "docs/04_STAGE0_PROTOCOL.md",
    "docs/05_GATES_AND_STOP_RULES.md",
    "docs/06_REAL_MODEL_PLAN.md",
    "docs/07_ARTIFACT_SCHEMAS.md",
    "docs/08_LINEAGE_AND_NEGATIVE_EVIDENCE.md",
    "docs/09_IMPLEMENTATION_STATUS.md",
    "docs/10_DECISION_LOG.md",
    "docs/11_REAL_STAGEA_IMPLEMENTATION.md",
    "docs/12_STAGEA_V1_AUDIT_AND_V2_PROTOCOL.md",
    "docs/13_STAGEA_V1_CORRECTION_LOG.md",
    "docs/OLIVIA.md",
    "docs/LUMI.md",
    "configs/stage0/synthetic_smoke.yaml",
    "configs/stage0/synthetic_full.yaml",
    "configs/stage0/real_smoke.yaml",
    "configs/stage0/real_olivia.yaml",
    "configs/stage0/real_lumi.yaml",
    "configs/stage0/real_lumi_v2.yaml",
    "docs/14_STAGEA_V2_PROTOCOL.md",
    "olivia/cli.py",
    "olivia/run.slurm",
    "olivia/quickstart.sh",
    "lumi/cli.py",
    "lumi/run.slurm",
    "lumi/quickstart.sh",
    ".agents/skills/olivia-cluster-runner/SKILL.md",
    ".agents/skills/lumi-cluster-runner/SKILL.md",
    "evidence/reference_stage0/metrics.json",
    "evidence/reference_stage0/decision.json",
    "evidence/reference_stage0/manifest.json",
    "evidence/real_stagea_devg_v2/decision.json",
    "evidence/real_stagea_devg_v2/metrics.json",
    "evidence/real_stagea_devg_v2/run_manifest.json",
    "evidence/real_stagea_devg_v2/verification_summary.json",
    "evidence/real_stagea_devg_v2/AUDIT.md",
    "evidence/real_stagea_devg_v2/audit.json",
    "evidence/real_stagea_devg_v2/manifest.json",
    "evidence/real_stagea_lumi_v2/AUDIT.md",
    "evidence/real_stagea_lumi_v2/decision.json",
    "evidence/real_stagea_lumi_v2/metrics.json",
    "evidence/real_stagea_lumi_v2/run_manifest.json",
    "evidence/real_stagea_lumi_v2/verification_summary.json",
    "evidence/real_stagea_lumi_v2/manifest.json",
)


def _validate_hash_manifest(directory: Path, manifest_name: str = "manifest.json") -> dict:
    manifest = json.loads((directory / manifest_name).read_text())
    expected = manifest.get("files", {})
    observed = {name: sha256_file(directory / name) for name in expected}
    if expected != observed:
        raise SystemExit(
            f"evidence hash mismatch in {directory}: expected={expected}, observed={observed}"
        )
    return manifest


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    synthetic_configs = (
        ROOT / "configs/stage0/synthetic_smoke.yaml",
        ROOT / "configs/stage0/synthetic_full.yaml",
    )
    real_configs = (
        ROOT / "configs/stage0/real_smoke.yaml",
        ROOT / "configs/stage0/real_olivia.yaml",
        ROOT / "configs/stage0/real_lumi.yaml",
        ROOT / "configs/stage0/real_lumi_v2.yaml",
    )
    for config_path in synthetic_configs:
        load_config(config_path)
    for config_path in real_configs:
        load_real_config(config_path)

    synthetic_dir = ROOT / "evidence/reference_stage0"
    synthetic_decision = json.loads((synthetic_dir / "decision.json").read_text())
    synthetic_metrics = json.loads((synthetic_dir / "metrics.json").read_text())
    synthetic_manifest = json.loads((synthetic_dir / "manifest.json").read_text())
    if synthetic_decision.get("schema") != "frank_eq_stage0_decision_v1":
        raise SystemExit("reference decision has the wrong schema")
    if synthetic_decision.get("status") != "pass":
        raise SystemExit("reference Stage-0 decision is not a pass")
    if synthetic_decision.get("authorizes_scientific_claim") is not False:
        raise SystemExit("synthetic evidence must not authorize a scientific claim")
    if synthetic_metrics.get("schema") != "frank_eq_stage0_metrics_v1":
        raise SystemExit("reference metrics have the wrong schema")
    synthetic_expected = synthetic_manifest.get("files", {})
    synthetic_observed = {
        "metrics.json": sha256_file(synthetic_dir / "metrics.json"),
        "decision.json": sha256_file(synthetic_dir / "decision.json"),
    }
    if synthetic_expected != synthetic_observed:
        raise SystemExit(
            "reference evidence hash mismatch: "
            f"expected={synthetic_expected}, observed={synthetic_observed}"
        )

    real_dir = ROOT / "evidence/real_stagea_devg_v2"
    real_manifest = _validate_hash_manifest(real_dir)
    real_decision = json.loads((real_dir / "decision.json").read_text())
    real_metrics = json.loads((real_dir / "metrics.json").read_text())
    real_audit = json.loads((real_dir / "audit.json").read_text())
    real_verification = json.loads((real_dir / "verification_summary.json").read_text())

    v2_dir = ROOT / "evidence/real_stagea_lumi_v2"
    v2_manifest = _validate_hash_manifest(v2_dir)
    v2_decision = json.loads((v2_dir / "decision.json").read_text())
    v2_verification = json.loads((v2_dir / "verification_summary.json").read_text())
    if v2_manifest.get("schema") != "frank_eq_real_stagea_evidence_manifest_v1":
        raise SystemExit("real Stage-A v2 evidence manifest has the wrong schema")
    if v2_decision.get("status") != "fail":
        raise SystemExit("adopted real Stage-A v2 outcome must remain a failure")
    if v2_decision.get("decision") != "STOP_OR_REVISE_STAGE0":
        raise SystemExit("adopted real Stage-A v2 decision changed")
    if v2_decision.get("authorizes_scientific_claim") is not False:
        raise SystemExit("negative v2 evidence must not authorize a scientific claim")
    if v2_verification.get("overall") != "passed":
        raise SystemExit("real Stage-A v2 workflow verification did not pass")
    if v2_verification.get("workflow", {}).get("completed_stages") != [
        "cache",
        "validate",
        "train",
        "eval",
    ]:
        raise SystemExit("real Stage-A v2 workflow stage list changed")

    if real_manifest.get("schema") != "frank_eq_real_stagea_evidence_manifest_v1":
        raise SystemExit("real Stage-A evidence manifest has the wrong schema")
    if real_decision.get("status") != "fail":
        raise SystemExit("adopted real Stage-A v1 outcome must remain a failure")
    if real_decision.get("decision") != "STOP_OR_REVISE_STAGE0":
        raise SystemExit("adopted real Stage-A decision changed")
    if real_decision.get("authorizes_scientific_claim") is not False:
        raise SystemExit("negative real evidence must not authorize a scientific claim")
    if real_metrics.get("scope") != "real frozen-LLM future-defined causal-state Stage A":
        raise SystemExit("real Stage-A metrics have the wrong scope")
    if real_audit.get("outcome", {}).get("scientific_decision_valid") is not True:
        raise SystemExit("real Stage-A audit must preserve the valid negative decision")
    if real_audit.get("audit_findings", {}).get("failure_localization") != "unresolved":
        raise SystemExit("real Stage-A v1 localization must remain unresolved")
    authorization = real_audit.get("authorization", {})
    if any(
        authorization.get(key) is not False
        for key in (
            "test_reuse_authorized",
            "new_outcome_run_authorized",
            "receiver_execution_authorized",
            "scientific_claim_authorized",
        )
    ):
        raise SystemExit("real Stage-A audit improperly authorizes continuation")

    if real_verification.get("schema") != "frank_eq_adopted_verification_summary_v1":
        raise SystemExit("real Stage-A verification summary has the wrong schema")
    if real_verification.get("overall") != "passed":
        raise SystemExit("real Stage-A workflow verification did not pass")
    workflow = real_verification.get("workflow", {})
    if workflow.get("state") != "completed" or workflow.get("failure") is not None:
        raise SystemExit("real Stage-A workflow did not complete cleanly")
    if workflow.get("completed_stages") != ["cache", "validate", "train", "eval"]:
        raise SystemExit("real Stage-A workflow stage list changed")
    if not all(real_verification.get("artifacts", {}).values()):
        raise SystemExit("real Stage-A verification summary has missing artifacts")
    verification_decision = real_verification.get("scientific_decision", {})
    if verification_decision != {
        "decision": "STOP_OR_REVISE_STAGE0",
        "status": "fail",
    }:
        raise SystemExit("verification summary does not preserve the scientific decision")
    if real_verification.get("authorizes_scientific_claim") is not False:
        raise SystemExit("verification summary improperly authorizes a claim")

    gitignore = (ROOT / ".gitignore").read_text()
    if ".agents/state/" not in gitignore:
        raise SystemExit(".agents/state/ must remain ignored")

    print(
        json.dumps(
            {
                "status": "passed",
                "required_files": len(REQUIRED),
                "synthetic_configs": len(synthetic_configs),
                "real_configs": len(real_configs),
                "synthetic_decision": synthetic_decision["decision"],
                "real_stagea_v1_decision": real_decision["decision"],
                "real_stagea_v1_localization": real_audit["audit_findings"][
                    "failure_localization"
                ],
                "real_stagea_v1_workflow": real_verification["overall"],
                "real_stagea_v2_decision": v2_decision["decision"],
                "real_stagea_v2_workflow": v2_verification["overall"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
