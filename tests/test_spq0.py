from __future__ import annotations

from pathlib import Path

import numpy as np

from frank_eq.shared_predictive_quotient.config import load_spq0_config
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


def _contract():
    config = load_spq0_config(CONFIG)
    systems, basis = config.build_systems_and_basis()
    return config, systems, basis


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
    _, systems, basis = _contract()
    assert basis.exact_rank == 4
    assert basis.maximum_rank == 8
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
