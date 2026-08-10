from __future__ import annotations

import json
from pathlib import Path

from frank_eq.real_config import load_real_config
from frank_eq.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_stagea_v2_supplemental_review_is_hash_verified_and_nonpromotional() -> None:
    evidence = ROOT / "evidence/real_stagea_lumi_v2"
    manifest = json.loads((evidence / "review_manifest.json").read_text())
    assert manifest["schema"] == "frank_eq_stagea_supplemental_review_manifest_v1"
    assert {
        name: sha256_file(evidence / name) for name in manifest["files"]
    } == manifest["files"]
    review = json.loads((evidence / "review.json").read_text())
    assert review["preserved_outcome"]["exact_pipeline_negative_valid"] is True
    assert review["interpretation"]["native_chat_template_falsified"] is False
    assert review["interpretation"]["prompt_surface_isolated"] is False
    assert not any(review["authorization"].values())


def test_stageq_configs_are_paired_except_for_prompt_contract_and_identity() -> None:
    legacy = load_real_config(ROOT / "configs/stageq/real_lumi_legacy_chat.yaml")
    candidate = load_real_config(ROOT / "configs/stageq/real_lumi_chat_turn.yaml")
    assert legacy.capture.prompt_format == "chat"
    assert candidate.capture.prompt_format == "chat_turn"

    legacy_dict = legacy.as_dict()
    candidate_dict = candidate.as_dict()
    for payload in (legacy_dict, candidate_dict):
        payload.pop("run_name")
        payload.pop("output_dir")
        payload["logging"]["wandb"]["tags"] = []
        payload["capture"]["prompt_format"] = "paired-placeholder"
    assert legacy_dict == candidate_dict
