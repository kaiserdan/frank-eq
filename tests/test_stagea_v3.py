from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from frank_eq.stagea_v3 import (
    IndependentChannelCompilers,
    StageAV3AccessController,
    TokenSlotCompiler,
    generate_v3_panel,
    load_stagea_v3_config,
    render_v3_world_prefix,
)
from frank_eq.stagea_v3.compiler import active_coordinate_indices, canonical_coordinates
from frank_eq.utils import atomic_write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/stagea_v3/real_olivia_v3.yaml"


def test_frozen_stagea_v3_config_loads_and_is_hash_bound() -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    assert [model.role for model in config.models] == ["founder", "founder", "held"]
    assert config.held_model.task_exposure == "unopened"
    assert config.section("panel")["operation_seed"] == 2026081213


def test_v3_panels_are_role_fresh_but_share_operations() -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    for entity_count in (4, 6):
        train = generate_v3_panel(config, "train", entity_count)
        validation = generate_v3_panel(config, "validation", entity_count)
        test = generate_v3_panel(config, "test", entity_count)
        repeat = generate_v3_panel(config, "train", entity_count)

        assert train.operation_registry_sha256 == validation.operation_registry_sha256
        assert train.operation_registry_sha256 == test.operation_registry_sha256
        assert train.to_dict() == repeat.to_dict()
        assert train.panel.worlds[0].edges != validation.panel.worlds[0].edges
        assert validation.panel.worlds[0].edges != test.panel.worlds[0].edges
        assert len({train.public_world_id(0), validation.public_world_id(0), test.public_world_id(0)}) == 3


def test_unseen_renderer_is_query_blind_and_coordinate_complete() -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    for entity_count, expected_coordinates in ((4, 12), (6, 30)):
        world = generate_v3_panel(config, "train", entity_count).panel.worlds[0]
        rendered = render_v3_world_prefix(world, "canonical_edge_list")
        assert rendered.count("->") == expected_coordinates
        assert "operation" not in rendered.lower().replace("no operation has been selected", "")
        assert "No operation has been selected." in rendered


def _compiler(**overrides: int | float) -> TokenSlotCompiler:
    kwargs: dict[str, int | float] = {
        "input_width": 16,
        "n_depths": 4,
        "max_entities": 6,
        "max_tokens": 8,
        "model_dim": 24,
        "attention_heads": 4,
        "attention_blocks": 2,
        "feedforward_dim": 48,
        "dropout": 0.0,
    }
    kwargs.update(overrides)
    return TokenSlotCompiler(**kwargs)


def test_compiler_uses_canonical_subgraph_slots_and_masks_padding() -> None:
    compiler = _compiler().eval()
    residuals = torch.randn(2, 4, 5, 16)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    perturbed = residuals.clone()
    perturbed[0, :, 3:] = 1_000_000.0

    with torch.no_grad():
        n4 = compiler(residuals, mask, entity_count=4)
        n4_perturbed = compiler(perturbed, mask, entity_count=4)
        n6 = compiler(residuals, mask, entity_count=6)

    assert n4.shape == (2, 12)
    assert n6.shape == (2, 30)
    assert torch.allclose(n4[0], n4_perturbed[0], atol=1e-5)
    coordinates = canonical_coordinates(6)
    selected = [coordinates[index] for index in active_coordinate_indices(4)]
    assert selected == [(source, target) for source in range(4) for target in range(4) if source != target]


def test_compiler_api_is_query_blind_and_channels_are_disjoint() -> None:
    compilers = IndependentChannelCompilers(
        seed=211,
        input_width=16,
        n_depths=4,
        max_entities=6,
        max_tokens=8,
        model_dim=24,
        attention_heads=4,
        attention_blocks=1,
        feedforward_dim=48,
        dropout=0.0,
    )
    semantic_ids = {id(parameter) for parameter in compilers.semantic.parameters()}
    behavioral_ids = {id(parameter) for parameter in compilers.behavioral.parameters()}
    assert not semantic_ids.intersection(behavioral_ids)
    assert not torch.equal(
        compilers.semantic.coordinate_queries,
        compilers.behavioral.coordinate_queries,
    )
    with pytest.raises(TypeError):
        compilers.semantic(  # type: ignore[call-arg]
            torch.randn(1, 4, 3, 16),
            torch.ones(1, 3, dtype=torch.bool),
            entity_count=4,
            operation=torch.zeros(1),
        )


def _write_bound_manifest(
    root: Path,
    filename: str,
    schema: str,
    config_sha256: str,
    artifact_name: str,
) -> None:
    artifact = root / artifact_name
    artifact.write_text(artifact_name)
    atomic_write_json(
        root / filename,
        {
            "schema": schema,
            "status": "frozen",
            "config_sha256": config_sha256,
            "artifacts": {artifact_name: sha256_file(artifact)},
        },
    )


def test_access_controller_blocks_test_until_both_freezes_and_consumes_once(
    tmp_path: Path,
) -> None:
    config_sha256 = "a" * 64
    controller = StageAV3AccessController(tmp_path, config_sha256=config_sha256)
    assert controller.initialize()["current_stage"] == "prepare"
    with pytest.raises(RuntimeError, match="held onboarding"):
        controller.assert_can_create_test(["panels/test.json"])

    controller.advance("founder_fit")
    _write_bound_manifest(
        tmp_path,
        "freeze_manifest.json",
        "frank_eq_stagea_v3_freeze_v1",
        config_sha256,
        "founder.ckpt",
    )
    controller.advance("freeze")
    controller.advance("held_onboard")
    with pytest.raises(FileNotFoundError, match="held_onboarding_manifest"):
        controller.assert_can_create_test(["panels/test.json"])

    _write_bound_manifest(
        tmp_path,
        "held_onboarding_manifest.json",
        "frank_eq_stagea_v3_held_onboarding_v1",
        config_sha256,
        "held.ckpt",
    )
    ledger = controller.assert_can_create_test(["panels/test.json"])
    assert ledger["current_stage"] == "evaluate"
    assert ledger["test_access_count"] == 1
    with pytest.raises(RuntimeError, match="held onboarding"):
        controller.assert_can_create_test(["panels/test-again.json"])

    (tmp_path / "panels").mkdir()
    (tmp_path / "panels/test.json").write_text(json.dumps({"role": "test"}))
    opened = controller.record_test_file_open("panels/test.json")
    assert len(opened["test_file_opens"]) == 1
    with pytest.raises(RuntimeError, match="unregistered"):
        controller.record_test_file_open("panels/not-registered.json")


def test_access_controller_rejects_manifest_hash_drift(tmp_path: Path) -> None:
    controller = StageAV3AccessController(tmp_path, config_sha256="b" * 64)
    controller.initialize()
    controller.advance("founder_fit")
    _write_bound_manifest(
        tmp_path,
        "freeze_manifest.json",
        "frank_eq_stagea_v3_freeze_v1",
        "b" * 64,
        "founder.ckpt",
    )
    (tmp_path / "founder.ckpt").write_text("drift")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        controller.advance("freeze")
