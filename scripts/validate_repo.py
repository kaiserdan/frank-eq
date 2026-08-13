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
from frank_eq.shared_predictive_quotient.config import load_spq0_config  # noqa: E402
from frank_eq.shared_predictive_quotient.workflow import build_spq0_plan  # noqa: E402
from frank_eq.utils import canonical_json_bytes, sha256_file  # noqa: E402
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
    "docs/22_MAIN_RESULTS_AUDIT_AND_STAGE_M.md",
    "docs/23_STAGE_M_OPERATION_CLOSED_BASIS.md",
    "docs/24_STAGE_M_OLIVIA_RUNBOOK.md",
    "docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md",
    "docs/26_SPQ0_OLIVIA_RUNBOOK.md",
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
    "configs/moment_compute/real_olivia_m0.yaml",
    "configs/spq0/real_olivia_spq0.yaml",
    "configs/spq0/inspected_plan.json",
    "configs/spq0/registration.json",
    "frank_eq_spq0_research_and_implementation_plan.md",
    "frank_eq_spq0_config_skeleton.yaml",
    "scripts/qualify_real_cache.py",
    "scripts/compare_stageq_caches.py",
    "scripts/audit_rate_compute_result.py",
    "scripts/verify_stagea_v3_run.py",
    "scripts/smoke_stagea_v3_held_runtime.py",
    "scripts/validate_moment_compute.py",
    "scripts/verify_moment_compute_run.py",
    "scripts/validate_spq0.py",
    "scripts/verify_spq0_run.py",
    "olivia/cli.py",
    "olivia/run.slurm",
    "olivia/quickstart.sh",
    "olivia/stagea_v3.slurm",
    "olivia/stagea_v3_cache_held.slurm",
    "olivia/stagea_v3_held_smoke.slurm",
    "lumi/cli.py",
    "lumi/run.slurm",
    "lumi/quickstart.sh",
    ".agents/skills/olivia-cluster-runner/SKILL.md",
    ".agents/skills/lumi-cluster-runner/SKILL.md",
    ".agents/skills/moment-compute-runner/SKILL.md",
    ".agents/skills/spq0-runner/SKILL.md",
    "src/frank_eq/shared_predictive_quotient/automaton.py",
    "src/frank_eq/shared_predictive_quotient/capture.py",
    "src/frank_eq/shared_predictive_quotient/checkpoints.py",
    "src/frank_eq/shared_predictive_quotient/cli.py",
    "src/frank_eq/shared_predictive_quotient/config.py",
    "src/frank_eq/shared_predictive_quotient/evaluation.py",
    "src/frank_eq/shared_predictive_quotient/panel.py",
    "src/frank_eq/shared_predictive_quotient/probes.py",
    "src/frank_eq/shared_predictive_quotient/verify.py",
    "src/frank_eq/shared_predictive_quotient/workflow.py",
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
    "evidence/real_stagea_v3_olivia/AUDIT.md",
    "evidence/real_stagea_v3_olivia/access_ledger.json",
    "evidence/real_stagea_v3_olivia/artifact_manifest.json",
    "evidence/real_stagea_v3_olivia/baseline_manifest.json",
    "evidence/real_stagea_v3_olivia/capture_validation.json",
    "evidence/real_stagea_v3_olivia/compiler_checkpoints_manifest.json",
    "evidence/real_stagea_v3_olivia/config.yaml",
    "evidence/real_stagea_v3_olivia/decision.json",
    "evidence/real_stagea_v3_olivia/dry_run_plan.json",
    "evidence/real_stagea_v3_olivia/freeze_manifest.json",
    "evidence/real_stagea_v3_olivia/held_onboarding_manifest.json",
    "evidence/real_stagea_v3_olivia/identity_train_basis_manifest.json",
    "evidence/real_stagea_v3_olivia/implementation_manifest.json",
    "evidence/real_stagea_v3_olivia/independent_audit.json",
    "evidence/real_stagea_v3_olivia/manifest.json",
    "evidence/real_stagea_v3_olivia/metrics.json",
    "evidence/real_stagea_v3_olivia/models.json",
    "evidence/real_stagea_v3_olivia/predictions_manifest.json",
    "evidence/real_stagea_v3_olivia/rate_compute.json",
    "evidence/real_stagea_v3_olivia/registration.json",
    "evidence/real_stagea_v3_olivia/run_manifest.json",
    "evidence/real_stagea_v3_olivia/run_summary.json",
    "evidence/real_stagea_v3_olivia/test_panel_manifest.json",
    "evidence/real_stagea_v3_olivia/training_summary.json",
    "evidence/real_stagea_v3_olivia/verification_summary.json",
    "evidence/real_stagea_v3_olivia/verifier_order_diagnostic.json",
    "evidence/real_stagea_v3_olivia/workflow_status.json",
    "evidence/real_stage_m_olivia_m0/AUDIT.md",
    "evidence/real_stage_m_olivia_m0/artifact_manifest.json",
    "evidence/real_stage_m_olivia_m0/calibration.json",
    "evidence/real_stage_m_olivia_m0/config.yaml",
    "evidence/real_stage_m_olivia_m0/decision.json",
    "evidence/real_stage_m_olivia_m0/development_splits.json",
    "evidence/real_stage_m_olivia_m0/direct_protocol_selection.json",
    "evidence/real_stage_m_olivia_m0/independent_verification.json",
    "evidence/real_stage_m_olivia_m0/manifest.json",
    "evidence/real_stage_m_olivia_m0/metrics.json",
    "evidence/real_stage_m_olivia_m0/models.json",
    "evidence/real_stage_m_olivia_m0/numpy_runtime_diagnostic.json",
    "evidence/real_stage_m_olivia_m0/run_manifest.json",
    "evidence/real_stage_m_olivia_m0/run_summary.json",
    "evidence/real_stage_m_olivia_m0/verification_summary.json",
    "evidence/real_stage_m_olivia_m0/workflow_status.json",
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

    spq0_path = ROOT / "configs/spq0/real_olivia_spq0.yaml"
    spq0 = load_spq0_config(spq0_path)
    spq0_plan = json.loads((ROOT / "configs/spq0/inspected_plan.json").read_text())
    expected_spq0_plan = build_spq0_plan(spq0, config_path=spq0_path)
    if canonical_json_bytes(spq0_plan) != canonical_json_bytes(expected_spq0_plan):
        raise SystemExit("SPQ0 inspected plan differs from deterministic recomputation")
    spq0_registration = json.loads(
        (ROOT / "configs/spq0/registration.json").read_text()
    )
    if spq0_registration.get("schema") != "frank_eq_spq0_registration_manifest_v1":
        raise SystemExit("SPQ0 registration manifest has the wrong schema")
    if (
        spq0_registration.get("status") != "prospective_development_only"
        or spq0_registration.get("implementation_pr_launch_authorized") is not False
        or any(spq0_registration.get("access", {}).values())
    ):
        raise SystemExit("SPQ0 registration improperly opens execution or access")
    for relative, expected_hash in spq0_registration.get("files", {}).items():
        if sha256_file(ROOT / relative) != expected_hash:
            raise SystemExit(f"SPQ0 registration hash changed: {relative}")
    if (
        spq0_registration.get("inspected_plan_sha256")
        != spq0_plan.get("plan_sha256")
        or spq0_registration.get("active_checkpoint_revision_registry_sha256")
        != spq0_plan.get("active_checkpoint_revision_registry_sha256")
        or spq0_registration.get("reserved_checkpoint_non_access_contract_sha256")
        != spq0_plan.get("reserved_checkpoint_non_access_contract_sha256")
    ):
        raise SystemExit("SPQ0 registration hashes differ from the inspected plan")

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

    v3_evidence_dir = ROOT / "evidence/real_stagea_v3_olivia"
    v3_evidence_manifest = _validate_hash_manifest(v3_evidence_dir)
    v3_evidence_decision = json.loads((v3_evidence_dir / "decision.json").read_text())
    v3_evidence_metrics = json.loads((v3_evidence_dir / "metrics.json").read_text())
    v3_evidence_workflow = json.loads(
        (v3_evidence_dir / "workflow_status.json").read_text()
    )
    v3_evidence_run_summary = json.loads(
        (v3_evidence_dir / "run_summary.json").read_text()
    )
    v3_evidence_run_manifest = json.loads(
        (v3_evidence_dir / "run_manifest.json").read_text()
    )
    v3_evidence_access = json.loads((v3_evidence_dir / "access_ledger.json").read_text())
    v3_evidence_capture = json.loads(
        (v3_evidence_dir / "capture_validation.json").read_text()
    )
    v3_evidence_audit = json.loads(
        (v3_evidence_dir / "independent_audit.json").read_text()
    )
    v3_evidence_verification = json.loads(
        (v3_evidence_dir / "verification_summary.json").read_text()
    )
    v3_order_diagnostic = json.loads(
        (v3_evidence_dir / "verifier_order_diagnostic.json").read_text()
    )
    if v3_evidence_manifest.get("schema") != "frank_eq_stagea_v3_evidence_manifest_v1":
        raise SystemExit("Stage-A v3 evidence manifest has the wrong schema")
    expected_v3_evidence_files = set(v3_evidence_manifest.get("files", {})) | {
        "manifest.json"
    }
    observed_v3_evidence_files = {
        path.name for path in v3_evidence_dir.iterdir() if path.is_file()
    }
    if observed_v3_evidence_files != expected_v3_evidence_files:
        raise SystemExit("Stage-A v3 compact evidence file set changed")
    v3_run_environment = v3_evidence_run_manifest.get("environment", {})
    if (
        v3_evidence_manifest.get("source_archive_sha256")
        != v3_run_environment.get("source_sha256")
        or v3_evidence_manifest.get("job_name")
        != v3_run_environment.get("project_version")
        or v3_evidence_manifest.get("slurm_job_id")
        != v3_run_environment.get("slurm_job_id")
    ):
        raise SystemExit("Stage-A v3 evidence source lineage changed")
    if sha256_file(v3_evidence_dir / "config.yaml") != sha256_file(v3_path):
        raise SystemExit("adopted Stage-A v3 config differs from the frozen registration")
    if (
        v3_evidence_decision.get("status") != "fail"
        or v3_evidence_decision.get("diagnosis")
        != "ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED"
    ):
        raise SystemExit("adopted Stage-A v3 negative decision changed")
    expected_v3_checks = {
        "activation_specificity": False,
        "behavioral_basis": True,
        "composition": False,
        "held_sender": True,
        "integrity": True,
        "oracle_executor": True,
        "public_alignment": True,
        "quantization": True,
        "semantic_basis": False,
        "unseen_renderer": False,
    }
    if v3_evidence_decision.get("checks") != expected_v3_checks:
        raise SystemExit("adopted Stage-A v3 gate vector changed")
    if v3_evidence_metrics.get("gate_checks") != expected_v3_checks:
        raise SystemExit("Stage-A v3 metrics and decision gates differ")
    expected_v3_authorization = {
        "new_receiver_world_access_authorized": False,
        "paper_claim_authorized": False,
        "receiver_execution_authorized": False,
        "receiver_protocol_draft_authorized": False,
        "scientific_claim_authorized": False,
    }
    if v3_evidence_decision.get("authorization") != expected_v3_authorization:
        raise SystemExit("Stage-A v3 negative improperly opens an authorization")
    expected_v3_integrity = {
        "checkpoint_seed_registry_complete": True,
        "config_snapshot_hash": True,
        "consumer_compute_declared": True,
        "exclusive_kv_and_prefix_continuity": True,
        "founder_freeze_present": True,
        "held_freeze_present": True,
        "model_revisions_exact": True,
        "protected_authorizations_closed": True,
        "required_baselines_complete": True,
        "test_access_consumed_once": True,
        "test_files_registered_and_opened": True,
    }
    if (
        v3_evidence_capture.get("checks") != expected_v3_integrity
        or v3_evidence_decision.get("integrity_checks") != expected_v3_integrity
    ):
        raise SystemExit("adopted Stage-A v3 integrity check changed")
    registered_test_files = v3_evidence_access.get("registered_test_files", [])
    opened_test_files = [
        row.get("path") for row in v3_evidence_access.get("test_file_opens", [])
    ]
    if (
        v3_evidence_access.get("test_access_count") != 1
        or len(registered_test_files) != 21
        or len(opened_test_files) != 21
        or len(set(registered_test_files)) != 21
        or len(set(opened_test_files)) != 21
        or set(registered_test_files) != set(opened_test_files)
    ):
        raise SystemExit("Stage-A v3 one-access ledger changed")
    if (
        v3_evidence_workflow.get("state") != "failed"
        or v3_evidence_workflow.get("completed_stages")
        != ["prepare", "founder_fit", "freeze", "held_onboard", "evaluate"]
        or v3_evidence_workflow.get("failure", {}).get("message")
        != "independent Stage-A v3 audit failed"
    ):
        raise SystemExit("Stage-A v3 fail-closed workflow status changed")
    if (
        v3_evidence_run_summary.get("status") != "completed"
        or v3_evidence_run_summary.get("workflow_integrity_passed") is not True
    ):
        raise SystemExit("Stage-A v3 pre-audit completion record changed")
    audit_checks = v3_evidence_audit.get("checks", {})
    if (
        v3_evidence_audit.get("passed") is not False
        or {name for name, passed in audit_checks.items() if not passed}
        != {"metrics_recomputed_exactly"}
        or v3_evidence_audit.get("decision") != v3_evidence_decision
    ):
        raise SystemExit("Stage-A v3 original independent audit changed")
    if (
        v3_evidence_verification.get("schema")
        != "frank_eq_stagea_v3_adopted_verification_summary_v1"
        or v3_evidence_verification.get("overall") != "adopted_valid_negative"
        or v3_evidence_verification.get("artifact_manifest_audit", {}).get(
            "hash_matches"
        )
        != 117
        or v3_evidence_verification.get("artifact_manifest_audit", {}).get("entries")
        != 118
    ):
        raise SystemExit("Stage-A v3 adopted verification summary changed")
    terminal_mismatches = v3_evidence_verification.get(
        "artifact_manifest_audit", {}
    ).get("terminal_mismatches", [])
    if [row.get("path") for row in terminal_mismatches] != ["workflow_status.json"]:
        raise SystemExit("Stage-A v3 terminal manifest mismatch set changed")
    if (
        v3_order_diagnostic.get("schema")
        != "frank_eq_stagea_v3_verifier_order_diagnostic_v1"
        or v3_order_diagnostic.get("workflow_config_order", {}).get(
            "exact_metrics_match"
        )
        is not True
        or v3_order_diagnostic.get("workflow_config_order", {}).get("difference_count")
        != 0
        or v3_order_diagnostic.get("lexicographic_manifest_key_order", {}).get(
            "difference_count"
        )
        != 46
        or v3_order_diagnostic.get("lexicographic_manifest_key_order", {}).get(
            "maximum_absolute_difference"
        )
        != 5.551115123125783e-17
        or v3_order_diagnostic.get("result", {}).get("gate_or_decision_difference")
        is not False
    ):
        raise SystemExit("Stage-A v3 verifier-order diagnosis changed")
    forbidden_evidence_suffixes = {".pt", ".npz", ".safetensors"}
    if any(
        path.suffix in forbidden_evidence_suffixes
        for path in v3_evidence_dir.rglob("*")
        if path.is_file()
    ):
        raise SystemExit("Stage-A v3 compact evidence contains generated tensor payloads")

    stage_m_dir = ROOT / "evidence/real_stage_m_olivia_m0"
    stage_m_manifest = _validate_hash_manifest(stage_m_dir)
    stage_m_artifact_manifest = json.loads(
        (stage_m_dir / "artifact_manifest.json").read_text()
    )
    stage_m_decision = json.loads((stage_m_dir / "decision.json").read_text())
    stage_m_metrics = json.loads((stage_m_dir / "metrics.json").read_text())
    stage_m_models = json.loads((stage_m_dir / "models.json").read_text())
    stage_m_roles = json.loads((stage_m_dir / "development_splits.json").read_text())
    stage_m_run_manifest = json.loads((stage_m_dir / "run_manifest.json").read_text())
    stage_m_workflow = json.loads((stage_m_dir / "workflow_status.json").read_text())
    stage_m_verification = json.loads(
        (stage_m_dir / "verification_summary.json").read_text()
    )
    stage_m_independent = json.loads(
        (stage_m_dir / "independent_verification.json").read_text()
    )
    stage_m_runtime_diagnostic = json.loads(
        (stage_m_dir / "numpy_runtime_diagnostic.json").read_text()
    )
    if stage_m_manifest.get("schema") != "frank_eq_moment_compute_evidence_manifest_v1":
        raise SystemExit("Stage M0 evidence manifest has the wrong schema")
    expected_stage_m_files = set(stage_m_manifest.get("files", {})) | {"manifest.json"}
    observed_stage_m_files = {
        path.name for path in stage_m_dir.iterdir() if path.is_file()
    }
    if observed_stage_m_files != expected_stage_m_files:
        raise SystemExit("Stage M0 compact evidence file set changed")
    stage_m_environment = stage_m_run_manifest.get("environment", {})
    if (
        stage_m_manifest.get("source_archive_sha256")
        != stage_m_environment.get("source_sha256")
        or stage_m_manifest.get("runtime_image_sha256")
        != stage_m_environment.get("runtime_image_sha256")
        or stage_m_manifest.get("slurm_job_id")
        != stage_m_environment.get("slurm_job_id")
    ):
        raise SystemExit("Stage M0 evidence source lineage changed")
    stage_m_config = ROOT / "configs/moment_compute/real_olivia_m0.yaml"
    if sha256_file(stage_m_dir / "config.yaml") != sha256_file(stage_m_config):
        raise SystemExit("adopted Stage M0 config differs from the frozen registration")
    if stage_m_artifact_manifest.get("schema") != (
        "frank_eq_moment_compute_artifact_manifest_v1"
    ):
        raise SystemExit("Stage M0 run artifact manifest has the wrong schema")
    adopted_stage_m_hashes = stage_m_manifest.get("files", {})
    run_stage_m_hashes = stage_m_artifact_manifest.get("files", {})
    for name in (
        "calibration.json",
        "config.yaml",
        "decision.json",
        "development_splits.json",
        "direct_protocol_selection.json",
        "metrics.json",
        "models.json",
        "run_manifest.json",
        "run_summary.json",
        "workflow_status.json",
    ):
        if adopted_stage_m_hashes.get(name) != run_stage_m_hashes.get(name):
            raise SystemExit(f"adopted Stage M0 artifact differs from verified run: {name}")
    expected_stage_m_checks = {
        "atomic_retention": True,
        "event_algebra_exact": True,
        "moment_advantage_robust_by_model": False,
        "moment_over_direct_aggregate": True,
        "moment_over_marginal_aggregate": False,
        "operation_closed_events_readable": False,
    }
    if (
        stage_m_decision.get("status") != "fail"
        or stage_m_decision.get("diagnosis") != "OPERATION_CLOSED_EVENTS_NOT_READABLE"
        or stage_m_decision.get("checks") != expected_stage_m_checks
    ):
        raise SystemExit("adopted Stage M0 negative decision changed")
    if not stage_m_decision.get("authorization") or any(
        stage_m_decision["authorization"].values()
    ):
        raise SystemExit("Stage M0 negative improperly opens an authorization")
    if (
        stage_m_workflow.get("state") != "completed"
        or stage_m_workflow.get("completed_stages") != ["audit"]
        or stage_m_workflow.get("failure") is not None
        or stage_m_workflow.get("scientific_decision") != stage_m_decision
    ):
        raise SystemExit("Stage M0 workflow completion record changed")
    if (
        len(stage_m_roles.get("calibration_world_ids", [])) != 32
        or len(stage_m_roles.get("selection_world_ids", [])) != 13
        or len(stage_m_roles.get("validation_world_ids", [])) != 19
        or stage_m_roles.get("test_world_ids") != []
    ):
        raise SystemExit("Stage M0 development split changed")
    if any(
        not row.get("branch_execution", {}).get("exclusive_cache_batching")
        or row.get("branch_execution", {}).get("exact_replay_response_branches") != 0
        or row.get("prefixes") != 128
        or row.get("records") != 52992
        for row in stage_m_models
    ):
        raise SystemExit("Stage M0 causal branch record changed")
    failed_event_groups = {
        name for name, row in stage_m_metrics.get("event_groups", {}).items()
        if row.get("passed") is not True
    }
    if failed_event_groups != {
        "qwen3-4b|4|joint_outdegree|6",
        "qwen3-4b|4|two_path_intersection|4",
        "qwen3-8b|4|joint_outdegree|6",
        "qwen3-8b|4|two_path_intersection|4",
    }:
        raise SystemExit("Stage M0 failed event-group set changed")
    stage_m_composition = stage_m_metrics.get("composition", {})
    if (
        stage_m_metrics.get("executor_mismatches") != 0
        or stage_m_metrics.get("event_registry_sha256")
        != "70ce5d31d22e814b91ca0f1f6ac29567e2ef4447b472c426a979166471ca6d55"
        or stage_m_composition.get("aggregate_over_marginal", {})
        .get("brier_gain_ci", {})
        .get("upper", 0.0)
        >= 0.0
        or stage_m_composition.get("aggregate_over_crossfitted_direct", {})
        .get("brier_gain_ci", {})
        .get("lower", 0.0)
        <= 0.0
    ):
        raise SystemExit("Stage M0 registered metric conclusions changed")
    if (
        stage_m_independent.get("passed") is not True
        or not all(stage_m_independent.get("checks", {}).values())
        or stage_m_independent.get("decision") != stage_m_decision
    ):
        raise SystemExit("Stage M0 in-job independent verification changed")
    if (
        stage_m_verification.get("schema")
        != "frank_eq_moment_compute_adopted_verification_summary_v1"
        or stage_m_verification.get("overall")
        != "adopted_valid_development_negative"
        or stage_m_verification.get("repository_verifier", {}).get("overall")
        != "passed"
        or stage_m_verification.get("specialized_verifier", {}).get(
            "exact_runtime_overall"
        )
        != "passed"
    ):
        raise SystemExit("Stage M0 adopted verification summary changed")
    if (
        stage_m_runtime_diagnostic.get("schema")
        != "frank_eq_moment_compute_numpy_runtime_diagnostic_v1"
        or stage_m_runtime_diagnostic.get("exact_runtime_reverification", {}).get(
            "passed"
        )
        is not True
        or stage_m_runtime_diagnostic.get("newer_runtime", {}).get(
            "prediction_rows_with_any_difference"
        )
        != 96
        or stage_m_runtime_diagnostic.get("newer_runtime", {}).get(
            "scientific_prediction_field_differences"
        )
        != 0
        or stage_m_runtime_diagnostic.get("result", {}).get("gate_or_decision_difference")
        is not False
    ):
        raise SystemExit("Stage M0 NumPy runtime diagnostic changed")
    if any(
        path.suffix in forbidden_evidence_suffixes
        for path in stage_m_dir.rglob("*")
        if path.is_file()
    ):
        raise SystemExit("Stage M0 compact evidence contains generated tensor payloads")

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
                "stagea_v3_diagnosis": v3_evidence_decision["diagnosis"],
                "stagea_v3_evidence": v3_evidence_verification["overall"],
                "stage_m_diagnosis": stage_m_decision["diagnosis"],
                "stage_m_evidence": stage_m_verification["overall"],
                "spq0_status": spq0_registration["status"],
                "spq0_plan_sha256": spq0_plan["plan_sha256"],
                "spq0_reserved_checkpoint_access": False,
                "stageq_pair_registered": True,
                "stageq_cache_only_enforced": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
