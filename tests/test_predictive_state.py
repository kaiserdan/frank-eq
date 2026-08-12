from __future__ import annotations

from pathlib import Path

import numpy as np

from frank_eq.predictive_state.automaton import PredictiveAutomaton
from frank_eq.predictive_state.config import load_predictive_state_config
from frank_eq.predictive_state.panel import (
    generate_predictive_panel,
    render_future_test_query,
    render_predictive_prefix,
)
from frank_eq.predictive_state.probes import (
    choose_ridge_and_layer,
    deterministic_token_hash_features,
    fit_ridge_probe,
    paired_brier_gain_interval,
    wrong_history_margin_interval,
)
from frank_eq.predictive_state.workflow import build_predictive_state_plan

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/predictive_state/real_olivia_psr0.yaml"


def _automaton() -> PredictiveAutomaton:
    return load_predictive_state_config(CONFIG).build_automaton()


def _basis():
    config = load_predictive_state_config(CONFIG)
    model = config.build_automaton()
    return model.build_basis(
        horizons=config.automaton.candidate_horizons,
        n_target_tests=config.automaton.n_target_tests,
        target_seed=config.automaton.target_seed,
        max_condition_number=config.automaton.max_core_condition_number,
        max_target_l1=config.automaton.max_target_executor_l1,
    )


def test_frozen_config_and_inspected_plan_are_consistent() -> None:
    config = load_predictive_state_config(CONFIG)
    plan = build_predictive_state_plan(
        config,
        config_path=Path("configs/predictive_state/real_olivia_psr0.yaml"),
    )
    import json

    stored = json.loads(
        (ROOT / "configs/predictive_state/inspected_plan.json").read_text()
    )
    assert plan == stored
    assert plan["public_basis"]["rank"] == 4
    assert plan["compute"]["prefixes_per_model"] == 1088
    assert plan["compute"]["response_branches_per_model"] == 23936
    assert plan["access"] == {
        "claim_bearing_test_role": False,
        "held_sender": False,
        "receiver": False,
        "future_operation_revealed_before_capture": False,
    }


def test_rank_selected_core_tests_exactly_factor_target_tests() -> None:
    model = _automaton()
    basis = _basis()
    assert basis.rank == 4
    assert basis.condition_number < 4.0
    assert len({len(test.actions) for test in basis.core_tests}) >= 2
    rng = np.random.default_rng(7)
    for _ in range(100):
        belief = rng.dirichlet(np.ones(4))
        core = belief @ basis.core_matrix
        expected = belief @ basis.target_matrix
        assert np.allclose(basis.execute(core, clip=False), expected, atol=1e-10)


def test_history_panels_preserve_base_history_across_renderers() -> None:
    config = load_predictive_state_config(CONFIG)
    model = config.build_automaton()
    basis = _basis()
    panel = generate_predictive_panel(
        model,
        basis,
        role="train",
        lengths=(8, 16),
        histories_per_length=16,
        seed=11,
        min_entropy=0.05,
        max_entropy=1.38,
        min_core_variance=1e-5,
        max_attempt_multiplier=100,
    )
    history = panel.histories[0]
    texts = [
        render_predictive_prefix(model, history, renderer)
        for renderer in panel.renderers
    ]
    assert len(set(texts)) == 3
    for action, observation in zip(
        history.actions, history.observations, strict=True
    ):
        assert model.action_names[action] in texts[0]
        assert model.observation_names[observation] in texts[0]
    query = render_future_test_query(
        model,
        basis.core_tests[0],
        false_display=" false",
        true_display=" true",
        sequence_cue="\nAnswer:",
    )
    assert "Registered future test" in query
    assert "false" in query and "true" in query


def test_train_only_layer_selection_recovers_predictive_features() -> None:
    rng = np.random.default_rng(17)
    histories = np.repeat(np.arange(60), 2)
    latent = rng.normal(size=(60, 3))
    targets = 1.0 / (1.0 + np.exp(-latent))
    row_targets = np.repeat(targets, 2, axis=0)
    features = rng.normal(size=(120, 3, 12))
    features[:, 1, :3] = np.repeat(latent, 2, axis=0) + rng.normal(
        scale=0.01, size=(120, 3)
    )
    selected = choose_ridge_and_layer(
        features,
        row_targets,
        histories,
        ridge_grid=[0.1, 1.0, 10.0],
        selection_fraction=0.2,
        selection_seed=3,
    )
    assert selected["layer"] == 1
    prediction = selected["probe"].predict(features[:, 1])
    baseline = np.repeat(
        row_targets.mean(axis=0, keepdims=True), len(row_targets), axis=0
    )
    interval = paired_brier_gain_interval(
        row_targets,
        prediction,
        baseline,
        histories,
        replicates=200,
        seed=5,
    )
    assert interval["lower"] > 0.0


def test_high_dimensional_ridge_uses_a_finite_dual_solution() -> None:
    rng = np.random.default_rng(29)
    features = rng.normal(size=(24, 512))
    target = 1.0 / (1.0 + np.exp(-(features[:, :3] @ rng.normal(size=(3, 4)))))
    probe = fit_ridge_probe(features, target, ridge=1.0)
    prediction = probe.predict(features)
    assert probe.weights.shape == (512, 4)
    assert np.all(np.isfinite(prediction))
    assert np.mean((prediction - target) ** 2) < 0.02


def test_token_hash_is_deterministic_order_sensitive_and_width_matched() -> None:
    token_ids = np.asarray([[1, 2, 3, 0], [3, 2, 1, 0]], dtype=np.int64)
    mask = token_ids != 0
    first = deterministic_token_hash_features(
        token_ids, mask, width=16, position_period=8
    )
    second = deterministic_token_hash_features(
        token_ids, mask, width=16, position_period=8
    )
    assert first.shape == (2, 16)
    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], first[1])


def test_wrong_history_margin_detects_history_specific_predictions() -> None:
    rng = np.random.default_rng(41)
    histories = np.repeat(np.arange(30), 2)
    lengths = np.repeat(np.where(np.arange(30) < 15, 8, 16), 2)
    truth_by_history = rng.uniform(0.1, 0.9, size=(30, 4))
    truth = np.repeat(truth_by_history, 2, axis=0)
    prediction = np.clip(truth + rng.normal(scale=0.01, size=truth.shape), 0.0, 1.0)
    interval = wrong_history_margin_interval(
        truth,
        prediction,
        histories,
        lengths,
        replicates=200,
        seed=9,
    )
    assert interval["lower"] > 0.0
