from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from frank_eq.data.real_panel import RealPanel, evaluate_operation
from frank_eq.rate_compute.backend import ProtocolScore
from frank_eq.stagea_v3 import (
    IndependentChannelCompilers,
    StageAV3AccessController,
    TokenSlotCompiler,
    V3CaptureShard,
    capture_panel_shard,
    generate_v3_panel,
    load_capture_shard,
    load_stagea_v3_config,
    render_v3_world_prefix,
    write_capture_shard,
)
from frank_eq.stagea_v3.baselines import (
    FinalTokenPublicMLP,
    TokenIDResampler,
    parse_v3_world_prefix,
)
from frank_eq.stagea_v3.compiler import active_coordinate_indices, canonical_coordinates
from frank_eq.stagea_v3.controls import apply_train_controls, fit_train_controls
from frank_eq.stagea_v3.evaluation import reduce_stagea_v3_decision
from frank_eq.stagea_v3.packet import (
    TypedEdgePacket,
    encode_rate_matched_text_basis,
    encode_typed_edge_packet,
    execute_typed_basis,
    panel_control_thresholds,
)
from frank_eq.stagea_v3.panel import V3Panel
from frank_eq.stagea_v3.predictions import (
    assemble_prediction_bundle,
    load_prediction_bundle,
    write_prediction_bundle_artifacts,
)
from frank_eq.stagea_v3.training import (
    load_basis_predictor,
    load_continuous_quotient,
    make_basis_predictor,
    predict_basis_logits,
    predict_continuous_logits,
    train_basis_predictor,
    train_continuous_quotient,
)
from frank_eq.stagea_v3.workflow import (
    STAGEA_V3_STAGE_ORDER,
    build_stagea_v3_plan,
    parse_stagea_v3_stages,
)
from frank_eq.utils import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/stagea_v3/real_olivia_v3.yaml"


def test_frozen_stagea_v3_config_loads_and_is_hash_bound() -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    assert [model.role for model in config.models] == ["founder", "founder", "held"]
    assert config.held_model.task_exposure == "unopened"
    assert config.section("panel")["operation_seed"] == 2026081213

    config.payload["panel"]["operation_seed"] = 1
    with pytest.raises(RuntimeError, match="in-memory"):
        config.section("panel")


def test_stagea_v3_datetime_usage_is_python310_compatible() -> None:
    for path in (ROOT / "src/frank_eq/stagea_v3").glob("*.py"):
        source = path.read_text()
        assert "from datetime import UTC" not in source
        assert "datetime.UTC" not in source


def test_v3_panels_are_role_fresh_but_share_operations() -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    for entity_count in (4, 6):
        train = generate_v3_panel(config, "train", entity_count)
        validation = generate_v3_panel(config, "validation", entity_count)
        repeat = generate_v3_panel(config, "train", entity_count)

        assert train.operation_registry_sha256 == validation.operation_registry_sha256
        assert train.to_dict() == repeat.to_dict()
        assert train.panel.worlds[0].edges != validation.panel.worlds[0].edges
        assert train.public_world_id(0) != validation.public_world_id(0)
        with pytest.raises(RuntimeError, match="access-ledger grant"):
            generate_v3_panel(config, "test", entity_count)


def test_v3_plan_is_stable_content_addressed_and_outcome_blind() -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    first = build_stagea_v3_plan(config, config_path=CONFIG_PATH)
    second = build_stagea_v3_plan(config, config_path=CONFIG_PATH.resolve())
    assert first == second
    assert first["config_path"] == "configs/stagea_v3/real_olivia_v3.yaml"
    assert first["stage_order"] == list(STAGEA_V3_STAGE_ORDER)
    assert first["held_model_task_opened"] is False
    assert first["test_panel_instantiated"] is False
    without_hash = dict(first)
    observed_hash = without_hash.pop("plan_sha256")
    assert observed_hash == sha256_bytes(canonical_json_bytes(without_hash))
    assert "olivia/stagea_v3.slurm" in first["implementation_files"]
    assert "docs/20_STAGEA_V3_PROTOCOL.md" in first["implementation_files"]
    with pytest.raises(ValueError, match="complete frozen sequence"):
        parse_stagea_v3_stages("prepare,founder_fit")


def test_unseen_renderer_is_query_blind_and_coordinate_complete() -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    for entity_count, expected_coordinates in ((4, 12), (6, 30)):
        world = generate_v3_panel(config, "train", entity_count).panel.worlds[0]
        rendered = render_v3_world_prefix(world, "canonical_edge_list")
        assert rendered.count("->") == expected_coordinates
        assert "operation" not in rendered.lower().replace("no operation has been selected", "")
        assert "No operation has been selected." in rendered


def test_text_parser_recovers_all_frozen_renderer_grammars() -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    world = generate_v3_panel(config, "train", 6).panel.worlds[0]
    for renderer in ("natural", "adjacency", "canonical_edge_list"):
        rendered = render_v3_world_prefix(world, renderer)
        assert np.array_equal(parse_v3_world_prefix(rendered, 6), world.fact_vector())


def test_typed_packet_round_trip_separates_payload_and_framing_bits() -> None:
    for entity_count in (4, 6):
        coordinates = entity_count * (entity_count - 1)
        probabilities = np.linspace(0.01, 0.99, coordinates)
        packet = encode_typed_edge_packet(
            probabilities,
            entity_count=entity_count,
            bits=4,
        )
        restored = TypedEdgePacket.from_dict(packet.to_dict())
        assert restored.payload_bits == coordinates * 4
        assert restored.framing_bits > 0
        assert restored.serialized_bits == restored.payload_bits + restored.framing_bits
        assert restored.probabilities().shape == (coordinates,)

        text_packet = encode_rate_matched_text_basis(
            np.asarray([index % 2 for index in range(coordinates)]),
            entity_count=entity_count,
        )
        assert text_packet.payload_bits == packet.payload_bits

        tampered = packet.to_dict()
        tampered["checksum_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="checksum"):
            TypedEdgePacket.from_dict(tampered)


def test_exact_public_basis_reproduces_every_panel_operation() -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    for entity_count in (4, 6):
        panel = generate_v3_panel(config, "train", entity_count).panel
        thresholds = panel_control_thresholds(panel.worlds)
        for world in panel.worlds[:4]:
            for operation in panel.operations:
                probability = execute_typed_basis(
                    world.fact_vector(),
                    operation.definition,
                    entity_count=entity_count,
                    control_thresholds=thresholds,
                )
                assert int(probability >= 0.5) == int(
                    evaluate_operation(world, operation.definition)
                )


def test_prediction_bundle_requires_every_registered_control(tmp_path: Path) -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    source_panel = generate_v3_panel(config, "validation", 4)
    worlds = source_panel.panel.worlds
    panel = V3Panel(
        role="test",
        entity_count=4,
        panel=RealPanel(
            worlds=worlds,
            operations=source_panel.panel.operations,
            oracle_signatures=source_panel.panel.oracle_signatures,
            config=source_panel.panel.config,
        ),
        operation_registry_sha256=source_panel.operation_registry_sha256,
    )
    semantic_rows: list[np.ndarray] = []
    operation_rows: list[np.ndarray] = []
    prefix_metadata: list[dict[str, object]] = []
    world_ids: list[int] = []
    renderer_ids: list[int] = []
    for world in worlds:
        for renderer_id, renderer in enumerate(("natural", "adjacency", "canonical_edge_list")):
            semantic_rows.append(world.fact_vector())
            operation_rows.append(np.asarray(source_panel.panel.oracle_signatures[world.world_id]))
            text = render_v3_world_prefix(world, renderer)
            prefix_metadata.append({"prefix_utf8_hex": text.encode().hex()})
            world_ids.append(world.world_id)
            renderer_ids.append(renderer_id)
    semantic = np.stack(semantic_rows).astype(np.float32)
    operations = np.stack(operation_rows).astype(np.float32)
    rows, coordinates = semantic.shape
    behavioral = semantic * 0.6 + 0.2
    shard = V3CaptureShard(
        model_id=config.founder_models[0].model_id,
        role="test",
        entity_count=4,
        layer_indices=(1, 2, 3, 4),
        hidden_width=8,
        residuals=torch.zeros(rows, 4, 3, 8),
        token_ids=torch.ones(rows, 3, dtype=torch.long),
        attention_mask=torch.ones(rows, 3, dtype=torch.bool),
        world_ids=torch.tensor(world_ids),
        renderer_ids=torch.tensor(renderer_ids),
        semantic_targets=torch.from_numpy(semantic),
        behavioral_targets=torch.from_numpy(behavioral),
        behavioral_log_odds=torch.logit(torch.from_numpy(behavioral)),
        operation_targets=torch.from_numpy(operations),
        operation_targets_hard=torch.from_numpy((operations >= 0.5).astype(np.int8)),
        direct_probabilities=torch.full((rows, 32, 3), 0.5),
        direct_log_odds=torch.zeros(rows, 32, 3),
        direct_generated_tokens=torch.zeros(rows, 32, 3, dtype=torch.int32),
        prefix_metadata=prefix_metadata,
        capture_summary={
            "prefix_forwards": rows,
            "operation_registry_sha256": panel.operation_registry_sha256,
        },
    )
    semantic_prediction = semantic * 0.8 + 0.1
    behavioral_prediction = behavioral * 0.8 + 0.1
    continuous = operations * 0.8 + 0.1

    def logit(value: np.ndarray) -> np.ndarray:
        return np.log(value) - np.log1p(-value)

    seeds = 3
    direct_tokens = np.zeros((rows, 32), dtype=np.int64)
    direct_tokens[:, 0] = 7
    direct_tokens[:, 1] = 32
    direct_protocols = ["sequence"] * 32
    direct_protocols[0] = "reason"
    direct_protocols[1] = "pause"
    controls: dict[str, np.ndarray | list[str]] = {
        "semantic_edge_prior": np.full((rows, coordinates), 0.5),
        "behavioral_edge_prior": np.full((rows, coordinates), 0.5),
        "operation_prior": np.full((rows, 32), 0.5),
        "interactive_basis": semantic_prediction,
        "direct_probability": continuous,
        "direct_generated_tokens": direct_tokens,
        "direct_protocols": direct_protocols,
    }
    bundle = assemble_prediction_bundle(
        config,
        shard=shard,
        panel=panel,
        semantic_primary=semantic_prediction,
        behavioral_primary=behavioral_prediction,
        token_primary=semantic_prediction,
        final_token_primary=semantic_prediction,
        continuous_primary=continuous,
        semantic_seed_logits=[logit(semantic_prediction)] * seeds,
        behavioral_seed_logits=[logit(behavioral_prediction)] * seeds,
        token_seed_logits=[logit(semantic_prediction)] * seeds,
        final_token_seed_logits=[logit(semantic_prediction)] * seeds,
        continuous_seed_logits=[logit(continuous)] * seeds,
        controls=controls,
        compiler_compute={"parameter_count": 1},
    )
    assert bundle.semantic_basis["primary_q4"].shape == (rows, coordinates)
    assert set(bundle.operations) >= {
        "historical_continuous_quotient",
        "train_selected_direct_protocol",
        "deterministic_text_parser",
    }
    assert all(
        record["payload_bits"] == 48
        for record in bundle.packet_records
        if record["bits_per_coordinate"] == 4
    )
    assert np.array_equal(
        bundle.operations["oracle_basis"] >= 0.5,
        bundle.operation_truth_hard.astype(bool),
    )
    assert bundle.compute["train_selected_direct_protocol"] == {
        "post_capture_source_queries_per_operation": 1,
        "generated_reasoning_tokens": rows * 7,
        "fixed_pause_tokens": rows * 32,
        "selected_sequence_operations": 30,
        "selected_reason_operations": 1,
        "selected_pause_operations": 1,
    }
    assert set(bundle.compute["executor_wall_seconds"]) == set(bundle.semantic_basis)
    artifacts = write_prediction_bundle_artifacts(
        tmp_path / "predictions.npz",
        tmp_path / "predictions.json",
        bundle,
        config_sha256=config.config_sha256,
    )
    restored = load_prediction_bundle(
        tmp_path / "predictions.npz",
        tmp_path / "predictions.json",
        config_sha256=config.config_sha256,
        expected_array_sha256=artifacts["array_sha256"],
        expected_metadata_sha256=artifacts["metadata_sha256"],
    )
    assert np.array_equal(restored.operations["primary_q4"], bundle.operations["primary_q4"])


def test_v3_decision_reducer_is_conjunctive_and_keeps_protected_actions_closed() -> None:
    checks = {
        "integrity": True,
        "semantic_basis": True,
        "unseen_renderer": True,
        "behavioral_basis": True,
        "activation_specificity": True,
        "composition": True,
        "public_alignment": True,
        "held_sender": True,
        "quantization": True,
        "oracle_executor": True,
    }
    passed = reduce_stagea_v3_decision(checks, {"artifacts": True})
    assert passed["diagnosis"] == "STAGEA_V3_REPRESENTATION_QUALIFIED"
    assert passed["authorization"]["receiver_protocol_draft_authorized"] is True
    assert passed["authorization"]["receiver_execution_authorized"] is False
    assert passed["authorization"]["scientific_claim_authorized"] is False

    expected = {
        "integrity": "INVALID_STAGEA_V3_RUN",
        "semantic_basis": "ONE_SHOT_PUBLIC_BASIS_NOT_QUALIFIED",
        "behavioral_basis": "BEHAVIORAL_STATE_NOT_QUALIFIED",
        "activation_specificity": "NO_ACTIVATION_SPECIFIC_ADVANTAGE",
        "composition": "NO_ONE_SHOT_COMPOSITION_ADVANTAGE",
        "held_sender": "HELD_SENDER_NOT_ESTABLISHED",
    }
    for failed_check, diagnosis in expected.items():
        failed = dict(checks)
        failed[failed_check] = False
        decision = reduce_stagea_v3_decision(failed, {"artifacts": failed_check != "integrity"})
        assert decision["diagnosis"] == diagnosis
        assert not any(decision["authorization"].values())


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
    assert selected == [
        (source, target) for source in range(4) for target in range(4) if source != target
    ]


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


def test_activation_controls_match_primary_parameter_budget() -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    primary = make_basis_predictor(config, kind="activation", input_width=16)
    token = make_basis_predictor(config, kind="token_id", input_width=16)
    final = make_basis_predictor(
        config,
        kind="final_token",
        input_width=16,
        target_parameter_count=sum(parameter.numel() for parameter in primary.parameters()),
    )
    primary_count = sum(parameter.numel() for parameter in primary.parameters())
    assert isinstance(token, TokenIDResampler)
    assert isinstance(final, FinalTokenPublicMLP)
    assert sum(parameter.numel() for parameter in token.parameters()) == primary_count
    assert (
        abs(sum(parameter.numel() for parameter in final.parameters()) - primary_count)
        / primary_count
        < 0.05
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


class _FakeTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        for word in text.split():
            start = text.index(word, cursor)
            stop = start + len(word)
            spans.append((start, stop))
            cursor = stop
        ids = torch.tensor(
            [[10 + (sum(text[start:stop].encode()) % 101) for start, stop in spans]],
            dtype=torch.long,
        )
        return {
            "input_ids": ids,
            "offset_mapping": torch.tensor([spans], dtype=torch.long),
        }


class _FakeModel:
    def __init__(self, revision: str) -> None:
        self.config = type("Config", (), {"_commit_hash": revision})()

    def __call__(self, *, input_ids: torch.Tensor, **_: object) -> object:
        tokens = int(input_ids.shape[1])
        hidden_states = tuple(
            torch.full((1, tokens, 8), float(layer), dtype=torch.float32) for layer in range(5)
        )
        return type(
            "Output",
            (),
            {"hidden_states": hidden_states, "past_key_values": ((torch.zeros(1),),)},
        )()


class _FakeCaptureAdapter:
    def __init__(self, revision: str) -> None:
        self.layer_indices = [1, 2, 3, 4]
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel(revision)
        self.device = torch.device("cpu")

    @staticmethod
    def _format_prefix(text: str) -> str:
        return text

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text)["input_ids"]

    def _query_ids(self, query: str, **_: object) -> torch.Tensor:
        length = 2 + len(query) % 3
        return torch.arange(1, length + 1, dtype=torch.long).unsqueeze(0)

    @staticmethod
    def _scores(query_ids: list[torch.Tensor], generated: int) -> list[ProtocolScore]:
        return [
            ProtocolScore(
                probability_true=0.25 + 0.1 * (int(query.shape[1]) % 3),
                log_odds_score=-0.5 + 0.25 * (int(query.shape[1]) % 3),
                false_token_count=1,
                true_token_count=1,
                generated_token_count=generated,
                generated_text="x" * generated,
            )
            for query in query_ids
        ]

    def score_sequence_batch(
        self,
        _: torch.Tensor,
        __: object,
        query_ids: list[torch.Tensor],
        ___: object,
    ) -> list[ProtocolScore]:
        return self._scores(query_ids, 0)

    def score_with_compute_batch(
        self,
        _: torch.Tensor,
        __: object,
        query_ids: list[torch.Tensor],
        ___: object,
        *,
        mode: str,
    ) -> list[ProtocolScore]:
        assert mode in {"reason", "pause"}
        return self._scores(query_ids, 32)

    @staticmethod
    def candidate_metadata(_: object) -> dict[str, object]:
        return {"false": {"text": " false"}, "true": {"text": " true"}}


def test_capture_shard_preserves_all_tokens_and_exclusive_query_accounting(
    tmp_path: Path,
) -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    full = generate_v3_panel(config, "train", 4)
    reduced_panel = RealPanel(
        worlds=full.panel.worlds[:2],
        operations=full.panel.operations,
        oracle_signatures=full.panel.oracle_signatures[:2],
        config=full.panel.config,
    )
    reduced = V3Panel(
        role="train",
        entity_count=4,
        panel=reduced_panel,
        operation_registry_sha256=full.operation_registry_sha256,
    )
    adapter = _FakeCaptureAdapter(config.founder_models[0].revision)
    shard = capture_panel_shard(
        config,
        config.founder_models[0],
        reduced,
        adapter=adapter,  # type: ignore[arg-type]
    )

    assert isinstance(shard, V3CaptureShard)
    assert shard.rows == 4
    assert shard.residuals.shape[:2] == (4, 4)
    assert shard.semantic_targets.shape == (4, 12)
    assert shard.behavioral_targets.shape == (4, 12)
    assert shard.direct_probabilities.shape == (4, 32, 3)
    assert shard.capture_summary["prefix_forwards"] == 4
    assert shard.capture_summary["logical_post_capture_source_queries"] == 4 * (12 + 96)
    assert shard.capture_summary["exact_replay_response_branches"] == 0
    assert all(row["captured_before_operation"] for row in shard.prefix_metadata)
    assert torch.all(shard.direct_generated_tokens[:, :, 0] == 0)
    assert torch.all(shard.direct_generated_tokens[:, :, 1:] == 32)

    path = tmp_path / "capture.pt"
    digest = write_capture_shard(path, shard)
    restored = load_capture_shard(path, expected_sha256=digest)
    assert torch.equal(restored.residuals, shard.residuals)
    assert restored.capture_summary == shard.capture_summary
    with pytest.raises(ValueError, match="hash mismatch"):
        load_capture_shard(path, expected_sha256="0" * 64)


class _TinyTrainingConfig:
    def __init__(self) -> None:
        self.config_sha256 = "c" * 64
        self.models = (SimpleNamespace(model_id="tiny", role="founder"),)
        self._sections = {
            "compiler": {
                "model_dim": 12,
                "attention_heads": 3,
                "attention_blocks": 1,
                "feedforward_dim": 24,
                "dropout": 0.0,
                "renderer_consistency_weight": 0.1,
                "seeds": [211],
            },
            "capture": {"normalized_depths": [0.25, 0.5, 0.75, 1.0], "compiler_max_tokens": 8},
            "panel": {"entity_counts": [4, 6]},
            "baselines": {"learned_parameter_tolerance_fraction": 0.05},
            "training": {
                "epochs": 2,
                "onboarding_epochs": 2,
                "worlds_per_batch": 2,
                "learning_rate": 0.001,
                "onboarding_learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip": 1.0,
                "patience": 2,
            },
        }

    def section(self, name: str) -> dict[str, object]:
        return deepcopy(self._sections[name])


def _tiny_shard(role: str, entity_count: int) -> V3CaptureShard:
    generator = torch.Generator().manual_seed(entity_count + (0 if role == "train" else 100))
    worlds = 4
    rows = worlds * 2
    coordinates = entity_count * (entity_count - 1)
    semantic = torch.randint(0, 2, (worlds, coordinates), generator=generator).float()
    semantic = semantic.repeat_interleave(2, dim=0)
    behavior = semantic * 0.6 + 0.2
    operation_hard = torch.randint(0, 2, (worlds, 32), generator=generator)
    operation_hard = operation_hard.repeat_interleave(2, dim=0)
    operation = operation_hard.float() * 0.96 + 0.02
    return V3CaptureShard(
        model_id="tiny",
        role=role,
        entity_count=entity_count,
        layer_indices=(1, 2, 3, 4),
        hidden_width=8,
        residuals=torch.randn(rows, 4, 5, 8, generator=generator),
        token_ids=torch.randint(1, 20, (rows, 5), generator=generator),
        attention_mask=torch.ones(rows, 5, dtype=torch.bool),
        world_ids=torch.arange(worlds).repeat_interleave(2),
        renderer_ids=torch.tensor([0, 1] * worlds),
        semantic_targets=semantic,
        behavioral_targets=behavior,
        behavioral_log_odds=torch.logit(behavior),
        operation_targets=operation,
        operation_targets_hard=operation_hard.to(torch.int8),
        direct_probabilities=torch.full((rows, 32, 3), 0.5),
        direct_log_odds=torch.zeros(rows, 32, 3),
        direct_generated_tokens=torch.zeros(rows, 32, 3, dtype=torch.int32),
        prefix_metadata=[{"row": index} for index in range(rows)],
        capture_summary={"rows": rows},
    )


def test_basis_trainer_world_groups_and_restores_frozen_checkpoint(tmp_path: Path) -> None:
    config = _TinyTrainingConfig()
    train = {entity_count: _tiny_shard("train", entity_count) for entity_count in (4, 6)}
    validation = {entity_count: _tiny_shard("validation", entity_count) for entity_count in (4, 6)}
    checkpoint = tmp_path / "semantic-211.pt"
    summary = train_basis_predictor(
        config,  # type: ignore[arg-type]
        train_shards=train,
        validation_shards=validation,
        kind="activation",
        channel="semantic",
        registered_seed=211,
        checkpoint_path=checkpoint,
        onboarding=False,
        capture_sha256={"train4": "1" * 64, "validation4": "2" * 64},
        device_name="cpu",
    )
    assert summary["best_epoch"] in {0, 1}
    assert summary["channel"] == "semantic"
    model, metadata = load_basis_predictor(
        config,
        checkpoint,
        device_name="cpu",  # type: ignore[arg-type]
    )
    logits = predict_basis_logits(
        model,
        metadata,
        validation[4],
        worlds_per_batch=2,
        device_name="cpu",
    )
    assert logits.shape == (8, 12)

    descriptors = {
        entity_count: torch.randn(32, 35, generator=torch.Generator().manual_seed(entity_count))
        for entity_count in (4, 6)
    }
    continuous_path = tmp_path / "continuous-211.pt"
    continuous_summary = train_continuous_quotient(
        config,  # type: ignore[arg-type]
        train_shards=train,
        validation_shards=validation,
        operation_descriptors=descriptors,
        registered_seed=211,
        checkpoint_path=continuous_path,
        onboarding=False,
        capture_sha256={"train4": "1" * 64},
        descriptor_sha256={"n4": "3" * 64, "n6": "4" * 64},
        device_name="cpu",
    )
    assert continuous_summary["kind"] == "historical_continuous_quotient"
    continuous, continuous_metadata = load_continuous_quotient(
        config,
        continuous_path,
        device_name="cpu",  # type: ignore[arg-type]
    )
    continuous_logits = predict_continuous_logits(
        continuous,
        continuous_metadata,
        validation[6],
        descriptors[6],
        worlds_per_batch=2,
        device_name="cpu",
    )
    assert continuous_logits.shape == (8, 32)


def test_train_controls_fit_only_train_and_apply_frozen_selection() -> None:
    config = load_stagea_v3_config(CONFIG_PATH)
    model_id = config.founder_models[0].model_id
    train_panels = {
        entity_count: generate_v3_panel(config, "train", entity_count) for entity_count in (4, 6)
    }
    train_shards: dict[int, V3CaptureShard] = {}
    for entity_count, panel in train_panels.items():
        shard = _tiny_shard("train", entity_count)
        shard.model_id = model_id
        shard.capture_summary.update(
            {
                "operation_registry_sha256": panel.operation_registry_sha256,
                "direct_protocol_order": ["sequence", "reason", "pause"],
            }
        )
        train_shards[entity_count] = shard
    artifact = fit_train_controls(
        config,
        model_id=model_id,
        train_shards=train_shards,
        panels=train_panels,
        capture_sha256={"n4": "1" * 64, "n6": "2" * 64},
    )
    assert artifact["fit_role"] == "train"
    assert artifact["validation_rows_used"] == artifact["test_rows_used"] == 0

    validation_panel = generate_v3_panel(config, "validation", 4)
    validation_shard = _tiny_shard("validation", 4)
    validation_shard.model_id = model_id
    applied = apply_train_controls(
        config,
        artifact,
        validation_shard,
        validation_panel,
    )
    assert np.asarray(applied["interactive_basis"]).shape == (8, 12)
    assert np.asarray(applied["direct_probability"]).shape == (8, 32)
    assert len(applied["direct_protocols"]) == 32
