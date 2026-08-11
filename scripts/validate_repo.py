#!/usr/bin/env python3
"""Validate repository contracts, configs, docs, and adopted evidence."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from frank_eq.config import load_config  # noqa: E402
from frank_eq.real_config import load_real_config  # noqa: E402
from frank_eq.utils import sha256_file  # noqa: E402
from frank_eq.workflow import validate_real_stage_role  # noqa: E402

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
    "docs/14_STAGEA_V2_PROTOCOL.md",
    "docs/15_STAGEA_V2_REVIEW_AND_STAGEQ.md",
    "docs/16_STAGEA_V2_INTERPRETATION_CORRECTION.md",
    "docs/17_STAGEQ_EXECUTION_AND_GATE_CONTRACT.md",
    "docs/18_RATE_COMPUTE_OPERATIONAL_BASIS.md",
    "docs/19_STAGE_R_CLUSTER_RUNBOOK.md",
    "docs/20_STAGEA_V3_PROTOCOL.md",
    "docs/OLIVIA.md",
    "docs/LUMI.md",
    "configs/stage0/synthetic_smoke.yaml",
    "configs/stage0/synthetic_full.yaml",
    "configs/stage0/real_smoke.yaml",
    "configs/stage0/real_olivia.yaml",
    "configs/stage0/real_lumi.yaml",
    "configs/stage0/real_lumi_v2.yaml",
    "configs/stageq/real_lumi_legacy_chat.yaml",
    "configs/stageq/real_lumi_chat_turn.yaml",
    "configs/stageq/real_lumi_screen_strong.yaml",
    "configs/stageq/real_lumi_screen_8b.yaml",
    "configs/stagea_v3/real_olivia_v3.yaml",
    "configs/stagea_v3/registration.json",
    "scripts/qualify_real_cache.py",
    "scripts/compare_stageq_caches.py",
    "scripts/audit_rate_compute_result.py",
    "scripts/verify_stagea_v3_run.py",
    "olivia/cli.py",
    "olivia/run.slurm",
    "olivia/quickstart.sh",
    "olivia/stagea_v3.slurm",
    "olivia/stagea_v3_cache_held.slurm",
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
    "evidence/real_stagea_lumi_v2/REVIEW.md",
    "evidence/real_stagea_lumi_v2/review.json",
    "evidence/real_stagea_lumi_v2/review_manifest.json",
    "evidence/real_stage_r_olivia_rc0/AUDIT.md",
    "evidence/real_stage_r_olivia_rc0/artifact_manifest.json",
    "evidence/real_stage_r_olivia_rc0/calibration.json",
    "evidence/real_stage_r_olivia_rc0/config.yaml",
    "evidence/real_stage_r_olivia_rc0/decision.json",
    "evidence/real_stage_r_olivia_rc0/direct_protocol_selection.json",
    "evidence/real_stage_r_olivia_rc0/independent_audit.json",
    "evidence/real_stage_r_olivia_rc0/metrics.json",
    "evidence/real_stage_r_olivia_rc0/models.json",
    "evidence/real_stage_r_olivia_rc0/recovery_provenance.json",
    "evidence/real_stage_r_olivia_rc0/run_manifest.json",
    "evidence/real_stage_r_olivia_rc0/run_summary.json",
    "evidence/real_stage_r_olivia_rc0/verification_summary.json",
    "evidence/real_stage_r_olivia_rc0/workflow_status.json",
    "evidence/real_stage_r_olivia_rc0/manifest.json",
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


def _normalize_stageq_config(payload: dict) -> dict:
    normalized = copy.deepcopy(payload)
    normalized.pop("run_name")
    normalized.pop("output_dir")
    normalized["logging"]["wandb"]["tags"] = []
    normalized["capture"]["prompt_format"] = "paired-placeholder"
    return normalized


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    synthetic_configs = (
        ROOT / "configs/stage0/synthetic_smoke.yaml",
        ROOT / "configs/stage0/synthetic_full.yaml",
    )
    stagea_configs = (
        ROOT / "configs/stage0/real_smoke.yaml",
        ROOT / "configs/stage0/real_olivia.yaml",
        ROOT / "configs/stage0/real_lumi.yaml",
        ROOT / "configs/stage0/real_lumi_v2.yaml",
    )
    stageq_configs = (
        ROOT / "configs/stageq/real_lumi_legacy_chat.yaml",
        ROOT / "configs/stageq/real_lumi_chat_turn.yaml",
    )
    for config_path in synthetic_configs:
        load_config(config_path)
    for config_path in (*stagea_configs, *stageq_configs):
        load_real_config(config_path)
    load_real_config(ROOT / "configs/stageq/real_lumi_screen_strong.yaml")
    load_real_config(ROOT / "configs/stageq/real_lumi_screen_8b.yaml")

    legacy_stageq = load_real_config(stageq_configs[0])
    candidate_stageq = load_real_config(stageq_configs[1])
    if legacy_stageq.capture.prompt_format != "chat":
        raise SystemExit("Stage-Q baseline must use historical chat turn placement")
    if candidate_stageq.capture.prompt_format != "chat_turn":
        raise SystemExit("Stage-Q candidate must use proper chat_turn capture")
    if _normalize_stageq_config(legacy_stageq.as_dict()) != _normalize_stageq_config(
        candidate_stageq.as_dict()
    ):
        raise SystemExit("Stage-Q configs differ outside prompt contract and run identity")
    role, stages = validate_real_stage_role(
        candidate_stageq,
        stageq_configs[1],
        "cache,validate",
    )
    if role != "stageq" or stages != ("cache", "validate"):
        raise SystemExit("Stage-Q cache-only workflow role did not resolve correctly")
    try:
        validate_real_stage_role(
            candidate_stageq,
            stageq_configs[1],
            "cache,validate,train,eval",
        )
    except ValueError:
        pass
    else:
        raise SystemExit("Stage-Q config improperly permits train/eval")

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

    v1_dir = ROOT / "evidence/real_stagea_devg_v2"
    v1_manifest = _validate_hash_manifest(v1_dir)
    v1_decision = json.loads((v1_dir / "decision.json").read_text())
    v1_metrics = json.loads((v1_dir / "metrics.json").read_text())
    v1_audit = json.loads((v1_dir / "audit.json").read_text())
    v1_verification = json.loads((v1_dir / "verification_summary.json").read_text())
    if v1_manifest.get("schema") != "frank_eq_real_stagea_evidence_manifest_v1":
        raise SystemExit("real Stage-A v1 evidence manifest has the wrong schema")
    if v1_decision.get("status") != "fail" or v1_decision.get("decision") != "STOP_OR_REVISE_STAGE0":
        raise SystemExit("adopted real Stage-A v1 outcome changed")
    if v1_decision.get("authorizes_scientific_claim") is not False:
        raise SystemExit("negative v1 evidence must not authorize a claim")
    if v1_metrics.get("scope") != "real frozen-LLM future-defined causal-state Stage A":
        raise SystemExit("real Stage-A v1 metrics have the wrong scope")
    if v1_audit.get("audit_findings", {}).get("failure_localization") != "unresolved":
        raise SystemExit("real Stage-A v1 localization must remain unresolved")
    if v1_verification.get("overall") != "passed":
        raise SystemExit("real Stage-A v1 workflow verification did not pass")

    v2_dir = ROOT / "evidence/real_stagea_lumi_v2"
    v2_manifest = _validate_hash_manifest(v2_dir)
    v2_decision = json.loads((v2_dir / "decision.json").read_text())
    v2_verification = json.loads((v2_dir / "verification_summary.json").read_text())
    if v2_manifest.get("schema") != "frank_eq_real_stagea_evidence_manifest_v1":
        raise SystemExit("real Stage-A v2 evidence manifest has the wrong schema")
    if v2_decision.get("status") != "fail" or v2_decision.get("decision") != "STOP_OR_REVISE_STAGE0":
        raise SystemExit("adopted real Stage-A v2 outcome changed")
    if v2_decision.get("authorizes_scientific_claim") is not False:
        raise SystemExit("negative v2 evidence must not authorize a claim")
    if v2_verification.get("overall") != "passed":
        raise SystemExit("real Stage-A v2 workflow verification did not pass")
    if v2_verification.get("workflow", {}).get("completed_stages") != [
        "cache",
        "validate",
        "train",
        "eval",
    ]:
        raise SystemExit("real Stage-A v2 workflow stage list changed")

    supplemental = _validate_hash_manifest(v2_dir, "review_manifest.json")
    review = json.loads((v2_dir / "review.json").read_text())
    if supplemental.get("schema") != "frank_eq_stagea_supplemental_review_manifest_v1":
        raise SystemExit("Stage-A v2 supplemental review manifest has the wrong schema")
    if review.get("preserved_outcome", {}).get("exact_pipeline_negative_valid") is not True:
        raise SystemExit("supplemental review must preserve the exact v2 negative")
    interpretation = review.get("interpretation", {})
    if interpretation.get("native_chat_template_falsified") is not False:
        raise SystemExit("supplemental review improperly generalizes v2 to native chat")
    if interpretation.get("prompt_surface_isolated") is not False:
        raise SystemExit("supplemental review improperly treats v1/v2 as paired")
    if any(review.get("authorization", {}).values()):
        raise SystemExit("supplemental review improperly authorizes continuation")

    rc0_dir = ROOT / "evidence/real_stage_r_olivia_rc0"
    rc0_manifest = _validate_hash_manifest(rc0_dir)
    rc0_artifact_manifest = json.loads((rc0_dir / "artifact_manifest.json").read_text())
    rc0_decision = json.loads((rc0_dir / "decision.json").read_text())
    rc0_verification = json.loads((rc0_dir / "verification_summary.json").read_text())
    rc0_audit = json.loads((rc0_dir / "independent_audit.json").read_text())
    rc0_run_manifest = json.loads((rc0_dir / "run_manifest.json").read_text())
    if rc0_manifest.get("schema") != "frank_eq_rate_compute_evidence_manifest_v1":
        raise SystemExit("Stage R / RC0 evidence manifest has the wrong schema")
    if rc0_artifact_manifest.get("schema") != "frank_eq_rate_compute_artifact_manifest_v1":
        raise SystemExit("Stage R / RC0 run artifact manifest has the wrong schema")
    adopted_hashes = rc0_manifest.get("files", {})
    run_hashes = rc0_artifact_manifest.get("files", {})
    for name in (
        "calibration.json",
        "config.yaml",
        "decision.json",
        "direct_protocol_selection.json",
        "metrics.json",
        "models.json",
        "recovery_provenance.json",
        "run_manifest.json",
        "run_summary.json",
        "workflow_status.json",
    ):
        if adopted_hashes.get(name) != run_hashes.get(name):
            raise SystemExit(f"adopted RC0 artifact differs from verified run: {name}")
    if (
        rc0_decision.get("status") != "pass"
        or rc0_decision.get("diagnosis") != "PUBLIC_BASIS_COMPOSITION_SUPPORTED"
    ):
        raise SystemExit("adopted Stage R / RC0 decision changed")
    authorization = rc0_decision.get("authorization", {})
    if authorization.get("stagea_registration_draft_authorized") is not True:
        raise SystemExit("RC0 pass must authorize exactly one registration draft")
    if any(
        authorization.get(key) is not False
        for key in (
            "stagea_outcome_run_authorized",
            "claim_bearing_test_access_authorized",
            "receiver_execution_authorized",
            "scientific_claim_authorized",
        )
    ):
        raise SystemExit("RC0 evidence improperly opens a protected authorization")
    if rc0_verification.get("overall") != "passed":
        raise SystemExit("Stage R / RC0 workflow verification did not pass")
    if rc0_audit.get("overall") != "passed" or rc0_audit.get("failures") != []:
        raise SystemExit("Stage R / RC0 independent recomputation audit did not pass")
    if rc0_audit.get("independent_gate_reduction", {}).get("diagnosis") != (
        rc0_decision["diagnosis"]
    ):
        raise SystemExit("RC0 machine and independent diagnoses differ")
    recovery = rc0_run_manifest.get("recovery", {})
    if (
        recovery.get("artifact_only") is not True
        or recovery.get("model_capture_executed") is not False
    ):
        raise SystemExit("RC0 evidence does not preserve artifact-only recovery provenance")

    v3_path = ROOT / "configs/stagea_v3/real_olivia_v3.yaml"
    v3_registration = yaml.safe_load(v3_path.read_text())
    if v3_registration.get("schema") != "frank_eq_stagea_v3_registration_v1":
        raise SystemExit("Stage-A v3 registration has the wrong schema")
    if v3_registration.get("protocol_version") != "stagea-v3-2":
        raise SystemExit("Stage-A v3 protocol version changed")
    expected_v3_models = {
        "qwen3-4b": ("founder", "1cfa9a7208912126459214e8b04321603b3df60c"),
        "qwen3-8b": ("founder", "b968826d9c46dd6066d109eabc6255188de91218"),
        "qwen3-14b-held": ("held", "40c069824f4251a91eefaf281ebe4c544efd3e18"),
    }
    observed_v3_models = {
        row["model_id"]: (row["role"], row["revision"])
        for row in v3_registration.get("models", [])
    }
    if observed_v3_models != expected_v3_models:
        raise SystemExit("Stage-A v3 model roles or revisions changed")
    panel_roles = v3_registration.get("panel", {}).get("roles", {})
    if v3_registration.get("panel", {}).get("operation_seed") != 2026081213:
        raise SystemExit("Stage-A v3 operation registry seed changed")
    observed_v3_seeds = [
        panel_roles.get(role, {}).get("seed") for role in ("train", "validation", "test")
    ]
    if observed_v3_seeds != [2026081201, 2026081202, 2026081297]:
        raise SystemExit("Stage-A v3 fresh panel seeds changed")
    access = v3_registration.get("access", {})
    if access.get("test_creation_after_freeze") is not True or access.get(
        "test_access_count"
    ) != 1:
        raise SystemExit("Stage-A v3 delayed one-time test access changed")
    v3_authorization = v3_registration.get("authorization", {})
    if v3_authorization.get(
        "one_representation_run_authorized_after_protocol_and_implementation_commits"
    ) is not True:
        raise SystemExit("Stage-A v3 representation authorization changed")
    if any(
        v3_authorization.get(key) is not False
        for key in (
            "receiver_execution_authorized",
            "new_receiver_world_access_authorized",
            "scientific_claim_authorized",
            "paper_claim_authorized",
        )
    ):
        raise SystemExit("Stage-A v3 registration opens a protected authorization")

    v3_manifest = json.loads((ROOT / "configs/stagea_v3/registration.json").read_text())
    if v3_manifest.get("schema") != "frank_eq_stagea_v3_registration_manifest_v1":
        raise SystemExit("Stage-A v3 registration manifest has the wrong schema")
    v3_registered_files = v3_manifest.get("files", {})
    if set(v3_registered_files) != {
        "configs/stagea_v3/real_olivia_v3.yaml",
        "docs/20_STAGEA_V3_PROTOCOL.md",
    }:
        raise SystemExit("Stage-A v3 registration manifest file set changed")
    for relative_path, expected_hash in v3_registered_files.items():
        if sha256_file(ROOT / relative_path) != expected_hash:
            raise SystemExit(f"Stage-A v3 registration hash changed: {relative_path}")

    gitignore = (ROOT / ".gitignore").read_text()
    if ".agents/state/" not in gitignore:
        raise SystemExit(".agents/state/ must remain ignored")

    print(
        json.dumps(
            {
                "status": "passed",
                "required_files": len(REQUIRED),
                "synthetic_configs": len(synthetic_configs),
                "stagea_configs": len(stagea_configs),
                "stageq_configs": len(stageq_configs),
                "synthetic_decision": synthetic_decision["decision"],
                "real_stagea_v1_decision": v1_decision["decision"],
                "real_stagea_v2_decision": v2_decision["decision"],
                "stage_r_rc0_diagnosis": rc0_decision["diagnosis"],
                "stage_r_rc0_recovery": "artifact_only",
                "stagea_v3_protocol": v3_registration["protocol_version"],
                "stageq_pair_registered": True,
                "stageq_cache_only_enforced": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
