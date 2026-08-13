from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from frank_eq.data.hf_backend import CHAT_ACKNOWLEDGEMENT
from frank_eq.shared_predictive_quotient.automaton import (
    BASIS_REGISTRY_DECIMALS,
    SELECTION_SCORE_DECIMALS,
    build_shared_predictive_basis,
    canonical_basis_registry_payload,
)
from frank_eq.shared_predictive_quotient.capture import SPQModelAdapter
from frank_eq.shared_predictive_quotient.config import load_spq0_config
from frank_eq.shared_predictive_quotient.evaluation import (
    evaluate_all_models,
    gate_decision,
)
from frank_eq.shared_predictive_quotient.panel import (
    build_panels,
    render_prefix,
    render_probability_query,
)
from frank_eq.shared_predictive_quotient.probes import (
    categorical_distribution,
    categorical_expectation,
    fit_linear_map,
    parameter_matched_token_sequence_features,
    select_categorical_temperature,
    select_target_reader,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/spq0/real_olivia_spq0.yaml"


class _SPQStubCapture:
    prompt_format = "chat_turn"
    max_length = 4096
    chat_template_kwargs = {"enable_thinking": False}


class _SPQStubSpec:
    model_id = "stub"


class _SPQTemplateTokenizer:
    chat_template = "stub"

    def __init__(self, family: str):
        self.family = family
        self.last_roles: list[str] = []

    @staticmethod
    def _encode(text: str) -> torch.Tensor:
        return torch.tensor([[ord(character) + 1 for character in text]], dtype=torch.long)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_tensors: str,
        truncation: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt" and truncation is False
        return {"input_ids": self._encode(text)}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> str:
        assert tokenize is False
        roles = [message["role"] for message in messages]
        self.last_roles = roles
        if self.family == "mistral" and any(
            role != ("user" if index % 2 == 0 else "assistant")
            for index, role in enumerate(roles)
        ):
            raise RuntimeError("Mistral roles must alternate")
        last_user = max(
            (index for index, role in enumerate(roles) if role == "user"),
            default=-1,
        )
        rendered = ""
        for index, message in enumerate(messages):
            content = message["content"]
            if self.family == "qwen" and message["role"] == "assistant":
                wrapper = "<think></think>" if index > last_user else ""
                rendered += f"<assistant>{wrapper}{content}</assistant>"
            else:
                rendered += f"<{message['role']}>{content}</{message['role']}>"
        if add_generation_prompt:
            rendered += "<assistant><think></think>" if self.family == "qwen" else "<assistant>"
        return rendered


def _spq_stub_adapter(family: str) -> SPQModelAdapter:
    adapter = object.__new__(SPQModelAdapter)
    adapter.spec = _SPQStubSpec()
    adapter.capture = _SPQStubCapture()
    adapter.tokenizer = _SPQTemplateTokenizer(family)
    adapter.device = torch.device("cpu")
    return adapter


def _contract():
    config = load_spq0_config(CONFIG)
    systems, basis = config.build_systems_and_basis()
    return config, systems, basis


@pytest.mark.parametrize("family", ["qwen", "mistral"])
def test_spq_chat_turn_is_cross_family_role_valid_and_prefix_exact(family: str) -> None:
    adapter = _spq_stub_adapter(family)
    world = "controlled stochastic history"
    query = "future-test probability bins"
    prefix = adapter._format_prefix(world)
    assert world in prefix
    assert adapter.tokenizer.last_roles == ["user"]
    prefix_ids = adapter._tokenize(prefix)
    suffix_ids = adapter._query_ids(
        query,
        world_statement=world,
        prefix_ids=prefix_ids,
    )
    assert adapter.tokenizer.last_roles == ["user", "assistant", "user"]
    combined = torch.cat([prefix_ids, suffix_ids], dim=1)
    messages = adapter._spq_prefix_messages(world)
    messages.extend(
        [
            {"role": "assistant", "content": CHAT_ACKNOWLEDGEMENT},
            {"role": "user", "content": query},
        ]
    )
    full_text = adapter._apply_chat_template(messages, add_generation_prompt=True)
    expected = adapter._tokenize(full_text, add_special_tokens=False)
    assert torch.equal(combined, expected)


def test_spq0_freezes_cross_family_active_and_unopened_reserved_models() -> None:
    config, _, _ = _contract()
    assert [(model.family, model.role) for model in config.models] == [
        ("qwen3", "founder"),
        ("mistral", "founder"),
    ]
    assert {model.family for model in config.reserved_unopened_models} == {
        "olmo2",
        "granite",
    }
    assert all(
        model.access == "reserved_unopened"
        for model in config.reserved_unopened_models
    )
    assert config.authorization.claim_bearing_test_access_authorized is False


def test_shared_future_tests_are_exact_at_rank_four_and_undercomplete_below() -> None:
    config, systems, basis = _contract()
    assert config.systems.system_seed == 2026084101
    assert basis.exact_rank == 4
    assert basis.maximum_rank == 8
    assert [test.to_dict() for test in basis.core_tests] == [
        {"actions": [0], "observation": 0},
        {"actions": [2], "observation": 0},
        {"actions": [0, 2], "observation": 0},
        {"actions": [0, 0, 2, 0], "observation": 2},
    ]
    assert SELECTION_SCORE_DECIMALS == 14
    assert BASIS_REGISTRY_DECIMALS == 10
    canonical_basis = canonical_basis_registry_payload(basis)
    assert canonical_basis["maximum_exact_executor_error"] == 0.0

    def numeric_leaves(value: object):
        if isinstance(value, float):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from numeric_leaves(item)
        elif isinstance(value, list):
            for item in value:
                yield from numeric_leaves(item)

    assert all(
        not np.signbit(value)
        for value in numeric_leaves(canonical_basis)
        if value == 0.0
    )
    assert max(basis.core_condition_numbers.values()) <= 5.0
    assert basis.maximum_target_l1 <= 4.0
    rng = np.random.default_rng(7)
    for system in systems:
        rank_three_errors = []
        for _ in range(40):
            belief = rng.dirichlet(np.ones(4))
            target = belief @ basis.target_matrices[system.system_id]
            for rank in (4, 6, 8):
                packet = basis.public_probabilities(system.system_id, belief, rank=rank)
                observed = basis.execute_targets(
                    system.system_id, packet, rank=rank, clip=False
                )
                assert np.allclose(observed, target, atol=1e-10, rtol=0.0)
            rank_three = basis.public_probabilities(system.system_id, belief, rank=3)
            approximate = basis.execute_targets(
                system.system_id, rank_three, rank=3, clip=False
            )
            rank_three_errors.append(float(np.max(np.abs(approximate - target))))
        assert max(rank_three_errors) > 1e-4


def test_validation_only_system_cannot_select_public_or_target_tests() -> None:
    config, systems, basis = _contract()
    held = systems[-1]
    replacement = replace(
        held,
        transitions=systems[1].transitions.copy(),
        emissions=systems[1].emissions.copy(),
    )
    altered = build_shared_predictive_basis(
        (*systems[:-1], replacement),
        horizons=config.systems.future_horizons,
        exact_rank=config.systems.predictive_rank,
        maximum_rank=max(config.semantic_encoder.rank_grid),
        n_target_tests=config.systems.target_tests,
        target_seed=config.systems.core_selection_seed,
        max_core_condition_number=config.systems.core_condition_number_max,
        max_target_l1=config.systems.target_executor_l1_max,
    )
    assert altered.public_tests == basis.public_tests
    assert altered.target_tests == basis.target_tests
    assert not np.array_equal(held.transitions, systems[0].transitions)
    assert not np.array_equal(held.emissions, systems[0].emissions)


def test_panels_have_three_disjoint_roles_and_validation_only_transfer() -> None:
    config, systems, basis = _contract()
    panels = build_panels(config, systems, basis)
    assert {role: len(panel.histories) for role, panel in panels.items()} == {
        "calibration": 384,
        "selection": 192,
        "validation": 576,
    }
    role_ids = [
        {history.history_id for history in panels[role].histories}
        for role in ("calibration", "selection", "validation")
    ]
    assert not role_ids[0] & role_ids[1]
    assert not role_ids[0] & role_ids[2]
    assert not role_ids[1] & role_ids[2]
    assert all(
        history.system_role == "fit"
        for role in ("calibration", "selection")
        for history in panels[role].histories
    )
    assert any(
        history.system_role == "validation_only"
        for history in panels["validation"].histories
    )
    assert 32 not in panels["calibration"].lengths
    assert 32 not in panels["selection"].lengths
    assert 32 in panels["validation"].lengths


def test_renderers_are_paired_and_probability_query_is_categorical() -> None:
    config, systems, basis = _contract()
    panel = build_panels(config, systems, basis)["calibration"]
    history = panel.histories[0]
    system = next(item for item in systems if item.system_id == history.system_id)
    rendered = [render_prefix(system, history, name) for name in ("narrative", "table", "symbolic")]
    assert len({item.text for item in rendered}) == 3
    assert all(len(item.event_end_markers) == history.length for item in rendered)
    assert all("future" in item.text.lower() and "unselected" in item.text.lower() for item in rendered)
    query = render_probability_query(
        system,
        basis.target_tests[0],
        bins=config.probability_protocol.bins,
        candidate_labels=config.probability_protocol.candidate_labels,
    )
    assert "probability bin" in query
    assert "A=0.05" in query and "J=0.95" in query
    assert "true or false" not in query.lower()


def test_categorical_temperature_selection_uses_probability_expectation() -> None:
    bins = np.asarray([0.05 + 0.1 * index for index in range(10)])
    logits = np.full((20, 3, 10), -8.0)
    truth = np.empty((20, 3), dtype=np.float64)
    for row in range(20):
        for test in range(3):
            selected = (row + test) % 10
            logits[row, test, selected] = 3.0
            truth[row, test] = bins[selected]
    result = select_categorical_temperature(logits, truth, bins, [0.5, 1.0, 2.0])
    assert result["selected_temperature"] == 0.5
    distribution = categorical_distribution(logits, result["selected_temperature"])
    expected = categorical_expectation(distribution, bins)
    assert float(np.mean((expected - truth) ** 2)) < 1e-6


def test_reduced_rank_map_and_parameter_matched_token_surface_are_deterministic() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(40, 32))
    targets = features[:, :3] @ rng.normal(size=(3, 6))
    fitted = fit_linear_map(
        features,
        targets,
        ridge=0.1,
        method="reduced_rank_regression",
        maximum_coefficient_rank=3,
    )
    assert fitted.coefficient_rank == 3
    assert np.mean((fitted.predict(features, clip=False) - targets) ** 2) < 0.01

    tokens = np.asarray([[1, 2, 3, 4, 0], [4, 3, 2, 1, 0]], dtype=np.int64)
    mask = tokens != 0
    boundaries = np.asarray([[1, 3], [1, 3]], dtype=np.int64)
    first = parameter_matched_token_sequence_features(
        tokens,
        mask,
        boundaries,
        width=32,
        decay_grid=[0.5, 0.75, 0.9, 0.97],
    )
    second = parameter_matched_token_sequence_features(
        tokens,
        mask,
        boundaries,
        width=32,
        decay_grid=[0.5, 0.75, 0.9, 0.97],
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], first[1])


def test_target_reader_is_fitted_from_oracle_cores_without_pair_parameters() -> None:
    _, _, basis = _contract()
    rng = np.random.default_rng(23)
    rows = 80
    cores = rng.uniform(0.05, 0.95, size=(rows, 4))
    semantic = rng.uniform(0.05, 0.95, size=(rows, len(basis.target_tests)))
    bins = np.asarray([0.05 + 0.1 * index for index in range(10)])
    distance = np.abs(semantic[:, :, None] - bins[None, None, :])
    signatures = np.exp(-20.0 * distance)
    signatures /= signatures.sum(axis=2, keepdims=True)
    calibration = np.arange(rows) < 50
    selection = (np.arange(rows) >= 50) & (np.arange(rows) < 70)
    reader, artifact = select_target_reader(
        cores,
        semantic,
        signatures,
        calibration,
        selection,
        basis.target_tests,
        [0.01, 0.1, 1.0],
    )
    prediction = reader.predict(cores[70:], semantic[70:], basis.target_tests)
    assert prediction.shape == (10, len(basis.target_tests), 10)
    assert np.allclose(prediction.sum(axis=2), 1.0)
    assert reader.metadata()["pair_specific_parameters"] is False
    assert artifact["refit_roles"] == ["calibration", "selection"]


def _synthetic_capture(model_id: str, family: str, width: int):
    config, systems, basis = _contract()
    rng = np.random.default_rng(100 + width)
    data_rng = np.random.default_rng(77)
    rows: list[tuple[int, int, int, int, int]] = []
    history_id = 1
    for role_id, count in ((0, 16), (1, 12)):
        for index in range(count):
            rows.append((history_id, role_id, index % 2, index % 2, 8 + 8 * (index % 2)))
            history_id += 1
    validation_conditions = [
        (0, 0, 8),
        (0, 2, 8),
        (2, 0, 8),
        (0, 0, 32),
        (2, 2, 32),
    ]
    for system_id, renderer_id, length in validation_conditions:
        for _ in range(5):
            rows.append((history_id, 2, system_id, renderer_id, length))
            history_id += 1
    n_rows = len(rows)
    beliefs = data_rng.dirichlet(np.ones(4), size=n_rows)
    public = np.empty((n_rows, 8))
    targets = np.empty((n_rows, len(basis.target_tests)))
    for row, (_, _, system_id, _, _) in enumerate(rows):
        system_name = f"system-{system_id:02d}"
        public[row] = beliefs[row] @ basis.public_matrices[system_name]
        targets[row] = beliefs[row] @ basis.target_matrices[system_name]
    projection = rng.normal(size=(4, width))
    base = public[:, :4] @ projection
    final = np.stack(
        [base + rng.normal(scale=0.03 + 0.01 * layer, size=base.shape) for layer in range(4)],
        axis=1,
    )
    event = np.concatenate([final, final**2], axis=2)
    all_token = np.concatenate([final, np.tanh(final)], axis=2)
    bins = np.asarray(config.probability_protocol.bins)
    semantic_all = np.concatenate([public, targets], axis=1)
    logits = -((semantic_all[:, :, None] - bins[None, None, :]) / 0.12) ** 2
    logits += rng.normal(scale=0.03, size=logits.shape)
    token_ids = rng.integers(1, 500, size=(n_rows, 40), dtype=np.int64)
    attention = np.ones_like(token_ids, dtype=np.bool_)
    boundaries = np.tile(np.asarray([7, 15, 23, 31]), (n_rows, 1))
    arrays = {
        "final_token_residual": final.astype(np.float32),
        "event_boundary_summary": event.astype(np.float32),
        "all_token_summary": all_token.astype(np.float32),
        "mean_input_embedding": (
            base + rng.normal(scale=0.2, size=base.shape)
        ).astype(np.float32),
        "token_ids": token_ids,
        "attention_mask": attention,
        "event_token_indices": boundaries,
        "history_ids": np.asarray([row[0] for row in rows], dtype=np.int64),
        "role_ids": np.asarray([row[1] for row in rows], dtype=np.int8),
        "system_ids": np.asarray([row[2] for row in rows], dtype=np.int8),
        "system_role_ids": np.asarray(
            [0 if row[2] < 2 else 1 for row in rows], dtype=np.int8
        ),
        "renderer_ids": np.asarray([row[3] for row in rows], dtype=np.int8),
        "lengths": np.asarray([row[4] for row in rows], dtype=np.int16),
        "last_observations": data_rng.integers(0, 3, size=n_rows, dtype=np.int8),
        "observation_frequencies": data_rng.dirichlet(np.ones(3), size=n_rows),
        "semantic_public": public,
        "semantic_core": public[:, :4],
        "semantic_targets": targets,
        "categorical_log_likelihoods": logits,
    }
    metadata = {"model_id": model_id, "family": family}
    return arrays, metadata


def test_complete_reducer_evaluates_both_ordered_pairs_without_pair_mapper() -> None:
    config, systems, basis = _contract()
    # Keep this reducer test quick while preserving grouped computation.
    config.evaluation.bootstrap_replicates = 50
    captures = {
        "mistral-7b-v03": _synthetic_capture("mistral-7b-v03", "mistral", 14),
        "qwen3-4b": _synthetic_capture("qwen3-4b", "qwen3", 12),
    }
    metrics, training, predictions, checkpoints = evaluate_all_models(
        config, systems, basis, captures
    )
    assert set(metrics["cross_family_composition"]) == {
        "mistral-7b-v03__to__qwen3-4b",
        "qwen3-4b__to__mistral-7b-v03",
    }
    assert all(
        row["pair_specific_mapper"] is False
        and row["target_reader_frozen_before_source_evaluation"] is True
        for row in metrics["cross_family_composition"].values()
    )
    assert metrics["behavioral_residual_census"]["promotional"] is False
    assert metrics["behavioral_residual_census"]["source_local_residual_encoders"] is True
    assert training["pair_specific_mapper_count"] == 0
    assert "cross_family" in predictions
    assert set(checkpoints) == {"mistral-7b-v03", "qwen3-4b"}
    for model in metrics["models"].values():
        baselines = model["semantic_core"]["joint_ood"]["baselines"]
        assert {"wrong_history", "shuffled_history", "renderer_shuffled"} <= set(
            baselines
        )
    decision = gate_decision(config, metrics)
    assert decision["authorization"]["spq1_execution_authorized"] is False
    assert decision["authorization"]["receiver_execution_authorized"] is False
