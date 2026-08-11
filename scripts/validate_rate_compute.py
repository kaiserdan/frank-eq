#!/usr/bin/env python3
"""Validate the isolated RC0 rate--compute/public-basis experiment contract."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from frank_eq.data.real_panel import evaluate_operation  # noqa: E402
from frank_eq.rate_compute.config import load_rate_compute_config  # noqa: E402
from frank_eq.rate_compute.logic import edge_vector_to_matrix, execute_public_basis  # noqa: E402
from frank_eq.rate_compute.records import build_panels  # noqa: E402


REQUIRED = (
    "docs/18_RATE_COMPUTE_OPERATIONAL_BASIS.md",
    "docs/19_STAGE_R_CLUSTER_RUNBOOK.md",
    "docs/20_RC0_DECISION_RECORD.md",
    "configs/rate_compute/real_lumi_rc0.yaml",
    "configs/rate_compute/real_olivia_rc0.yaml",
    "scripts/verify_rate_compute_run.py",
    "tests/test_rate_compute_config.py",
    "tests/test_rate_compute_logic.py",
    "tests/test_rate_compute_calibration.py",
)


def _normalize(payload: dict) -> dict:
    normalized = copy.deepcopy(payload)
    normalized.pop("run_name")
    normalized.pop("output_dir")
    normalized["logging"]["wandb"]["tags"] = []
    return normalized


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"missing RC0 files: {missing}")

    lumi = load_rate_compute_config(ROOT / "configs/rate_compute/real_lumi_rc0.yaml")
    olivia = load_rate_compute_config(ROOT / "configs/rate_compute/real_olivia_rc0.yaml")
    if _normalize(lumi.as_dict()) != _normalize(olivia.as_dict()):
        raise SystemExit("LUMI and Olivia RC0 contracts differ outside run identity")
    if [model.model_id for model in lumi.models] != ["qwen3-4b", "qwen3-8b"]:
        raise SystemExit("RC0 model roster changed")
    if any(model.revision is None for model in lumi.models):
        raise SystemExit("RC0 checkpoint revisions must remain pinned")
    if lumi.capture.prompt_format != "chat_turn":
        raise SystemExit("RC0 must use the corrected chat_turn contract")
    if lumi.capture.branch_mode != "kv_reuse" or lumi.capture.allow_exact_replay_fallback:
        raise SystemExit("RC0 must use exclusive KV branching without replay fallback")
    if lumi.protocols.rationale_budget != lumi.protocols.pause_budget:
        raise SystemExit("reasoning and pause controls must have matched token budgets")
    if lumi.protocols.basis_protocol != "sequence":
        raise SystemExit("RC0 basis protocol must remain semantic sequence likelihood")

    panels = build_panels(lumi)
    exact_checks = 0
    for n_entities, panel in panels.items():
        expected_basis = n_entities * (n_entities - 1)
        for world in panel.worlds[: min(8, len(panel.worlds))]:
            vector = world.fact_vector()
            if vector.shape != (expected_basis,):
                raise SystemExit("public basis has the wrong dimension")
            matrix = edge_vector_to_matrix(vector, n_entities)
            for operation in panel.operations:
                if operation.definition.family in {"density", "reciprocity"}:
                    continue
                observed = execute_public_basis(matrix, operation.definition) >= 0.5
                expected = evaluate_operation(world, operation.definition)
                if observed != expected:
                    raise SystemExit(
                        f"public basis failed exact execution for {n_entities=} "
                        f"operation={operation.definition.to_dict()}"
                    )
                exact_checks += 1

    for quickstart in (ROOT / "lumi/quickstart.sh", ROOT / "olivia/quickstart.sh"):
        text = quickstart.read_text()
        for required in ("configs/rate_compute/", "validate-rate-compute-config", "run-rate-compute"):
            if required not in text:
                raise SystemExit(f"{quickstart} lacks RC0 dispatch marker {required!r}")

    if not np.isfinite(float(exact_checks)) or exact_checks < 1:
        raise SystemExit("RC0 exact-basis audit produced no checks")
    print(
        json.dumps(
            {
                "status": "passed",
                "models": [model.model_id for model in lumi.models],
                "entity_counts": lumi.panel.entity_counts,
                "exact_basis_checks": exact_checks,
                "development_only": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
