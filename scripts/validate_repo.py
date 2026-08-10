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
    "docs/OLIVIA.md",
    "docs/LUMI.md",
    "configs/stage0/synthetic_smoke.yaml",
    "configs/stage0/synthetic_full.yaml",
    "configs/stage0/real_smoke.yaml",
    "configs/stage0/real_olivia.yaml",
    "configs/stage0/real_lumi.yaml",
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
)


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
    )
    for config_path in synthetic_configs:
        load_config(config_path)
    for config_path in real_configs:
        load_real_config(config_path)

    decision_path = ROOT / "evidence/reference_stage0/decision.json"
    metrics_path = ROOT / "evidence/reference_stage0/metrics.json"
    manifest_path = ROOT / "evidence/reference_stage0/manifest.json"
    decision = json.loads(decision_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    if decision.get("schema") != "frank_eq_stage0_decision_v1":
        raise SystemExit("reference decision has the wrong schema")
    if decision.get("status") != "pass":
        raise SystemExit("reference Stage-0 decision is not a pass")
    if decision.get("authorizes_scientific_claim") is not False:
        raise SystemExit("synthetic evidence must not authorize a scientific claim")
    if metrics.get("schema") != "frank_eq_stage0_metrics_v1":
        raise SystemExit("reference metrics have the wrong schema")

    expected = manifest.get("files", {})
    observed = {
        "metrics.json": sha256_file(metrics_path),
        "decision.json": sha256_file(decision_path),
    }
    if expected != observed:
        raise SystemExit(
            f"reference evidence hash mismatch: expected={expected}, observed={observed}"
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "required_files": len(REQUIRED),
                "synthetic_configs": len(synthetic_configs),
                "real_configs": len(real_configs),
                "reference_decision": decision["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
