from __future__ import annotations

from pathlib import Path

import pytest

from frank_eq.rate_compute.config import load_rate_compute_config


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_rate_compute_configs_load() -> None:
    lumi = load_rate_compute_config(ROOT / "configs/rate_compute/real_lumi_rc0.yaml")
    olivia = load_rate_compute_config(ROOT / "configs/rate_compute/real_olivia_rc0.yaml")

    assert [model.model_id for model in lumi.models] == ["qwen3-4b", "qwen3-8b"]
    assert lumi.panel.entity_counts == [4, 6]
    assert lumi.capture.prompt_format == "chat_turn"
    assert lumi.capture.branch_mode == "kv_reuse"
    assert lumi.capture.allow_exact_replay_fallback is False
    assert lumi.protocols.rationale_budget == lumi.protocols.pause_budget == 32
    assert lumi.protocols.basis_protocol == "sequence"

    left = lumi.as_dict()
    right = olivia.as_dict()
    for payload in (left, right):
        payload.pop("run_name")
        payload.pop("output_dir")
        payload["logging"]["wandb"]["tags"] = []
    assert left == right


def test_rate_compute_config_rejects_mixed_branch_execution(tmp_path: Path) -> None:
    source = (ROOT / "configs/rate_compute/real_lumi_rc0.yaml").read_text()
    path = tmp_path / "bad.yaml"
    path.write_text(
        source.replace(
            "allow_exact_replay_fallback: false",
            "allow_exact_replay_fallback: true",
        )
    )
    with pytest.raises(ValueError, match="forbids replay fallback"):
        load_rate_compute_config(path)
