"""Finite predictive-state environment and exact public core-test executor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True, slots=True, order=True)
class PredictiveTest:
    """One future action sequence followed by one terminal observation event."""

    actions: tuple[int, ...]
    observation: int

    def to_dict(self) -> dict[str, Any]:
        return {"actions": list(self.actions), "observation": self.observation}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PredictiveTest:
        return cls(
            actions=tuple(int(value) for value in payload["actions"]),
            observation=int(payload["observation"]),
        )


@dataclass(frozen=True, slots=True)
class PredictiveBasis:
    """A rank-complete public test basis and exact linear target executor."""

    core_tests: tuple[PredictiveTest, ...]
    target_tests: tuple[PredictiveTest, ...]
    core_matrix: np.ndarray
    target_matrix: np.ndarray
    executor: np.ndarray
    rank: int
    condition_number: float
    maximum_target_l1: float

    def validate(self, *, tolerance: float = 1e-10) -> None:
        if self.core_matrix.ndim != 2 or self.target_matrix.ndim != 2:
            raise ValueError("predictive basis matrices must be rank two")
        n_states, n_core = self.core_matrix.shape
        if n_core != len(self.core_tests) or n_states != n_core:
            raise ValueError("PSR0 uses a square state-rank core basis")
        if self.target_matrix.shape != (n_states, len(self.target_tests)):
            raise ValueError("target matrix shape differs from the target-test registry")
        if self.executor.shape != (n_core, len(self.target_tests)):
            raise ValueError("executor shape differs from the public test registries")
        observed_rank = int(np.linalg.matrix_rank(self.core_matrix, tol=tolerance))
        if self.rank != observed_rank or observed_rank != n_states:
            raise ValueError("public core tests do not separate latent belief states")
        if not np.isfinite(self.condition_number) or self.condition_number <= 0:
            raise ValueError("public core-test condition number is invalid")
        reconstructed = self.core_matrix @ self.executor
        if not np.allclose(reconstructed, self.target_matrix, atol=tolerance, rtol=0.0):
            raise ValueError("public executor does not exactly factor the target tests")
        if abs(self.maximum_target_l1 - float(np.max(np.sum(np.abs(self.executor), axis=0)))) > 1e-10:
            raise ValueError("stored public-executor sensitivity is inconsistent")

    def execute(self, core_probabilities: np.ndarray, *, clip: bool = True) -> np.ndarray:
        values = np.asarray(core_probabilities, dtype=np.float64)
        if values.shape[-1] != len(self.core_tests):
            raise ValueError("core probability vector has the wrong dimension")
        result = values @ self.executor
        return np.clip(result, 0.0, 1.0) if clip else result

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "frank_eq_predictive_basis_v1",
            "core_tests": [test.to_dict() for test in self.core_tests],
            "target_tests": [test.to_dict() for test in self.target_tests],
            "core_matrix": self.core_matrix.tolist(),
            "target_matrix": self.target_matrix.tolist(),
            "executor": self.executor.tolist(),
            "rank": self.rank,
            "condition_number": self.condition_number,
            "maximum_target_l1": self.maximum_target_l1,
        }


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    """One action-observation history with its exact posterior predictive state."""

    history_id: int
    role: str
    actions: tuple[int, ...]
    observations: tuple[int, ...]
    posterior: tuple[float, ...]
    core_probabilities: tuple[float, ...]
    target_probabilities: tuple[float, ...]
    entropy: float

    @property
    def length(self) -> int:
        return len(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HistoryRecord:
        return cls(
            history_id=int(payload["history_id"]),
            role=str(payload["role"]),
            actions=tuple(int(value) for value in payload["actions"]),
            observations=tuple(int(value) for value in payload["observations"]),
            posterior=tuple(float(value) for value in payload["posterior"]),
            core_probabilities=tuple(float(value) for value in payload["core_probabilities"]),
            target_probabilities=tuple(float(value) for value in payload["target_probabilities"]),
            entropy=float(payload["entropy"]),
        )


class PredictiveAutomaton:
    """Known controlled HMM used to define an observable predictive-state quotient."""

    def __init__(
        self,
        *,
        state_names: Iterable[str],
        action_names: Iterable[str],
        observation_names: Iterable[str],
        transition_matrices: np.ndarray,
        emission_matrix: np.ndarray,
        initial_belief: np.ndarray,
    ) -> None:
        self.state_names = tuple(str(value) for value in state_names)
        self.action_names = tuple(str(value) for value in action_names)
        self.observation_names = tuple(str(value) for value in observation_names)
        self.transitions = np.asarray(transition_matrices, dtype=np.float64)
        self.emissions = np.asarray(emission_matrix, dtype=np.float64)
        self.initial_belief = np.asarray(initial_belief, dtype=np.float64)
        self.validate()

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    @property
    def n_actions(self) -> int:
        return len(self.action_names)

    @property
    def n_observations(self) -> int:
        return len(self.observation_names)

    def validate(self) -> None:
        if len(set(self.state_names)) != len(self.state_names):
            raise ValueError("state names must be unique")
        if len(set(self.action_names)) != len(self.action_names):
            raise ValueError("action names must be unique")
        if len(set(self.observation_names)) != len(self.observation_names):
            raise ValueError("observation names must be unique")
        expected_transition = (self.n_actions, self.n_states, self.n_states)
        if self.transitions.shape != expected_transition:
            raise ValueError(f"transition matrices must have shape {expected_transition}")
        if self.emissions.shape != (self.n_states, self.n_observations):
            raise ValueError("emission matrix has the wrong shape")
        if self.initial_belief.shape != (self.n_states,):
            raise ValueError("initial belief has the wrong shape")
        for name, values in (
            ("transition", self.transitions),
            ("emission", self.emissions),
            ("initial belief", self.initial_belief),
        ):
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"{name} probabilities must be finite and non-negative")
        if not np.allclose(self.transitions.sum(axis=2), 1.0, atol=1e-10):
            raise ValueError("every transition row must sum to one")
        if not np.allclose(self.emissions.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("every emission row must sum to one")
        if not np.isclose(self.initial_belief.sum(), 1.0, atol=1e-10):
            raise ValueError("initial belief must sum to one")
        if np.any(self.initial_belief <= 0):
            raise ValueError("PSR0 requires a full-support initial belief")

    def posterior_update(
        self,
        belief: np.ndarray,
        action: int,
        observation: int,
    ) -> np.ndarray:
        prior = np.asarray(belief, dtype=np.float64)
        if prior.shape != (self.n_states,):
            raise ValueError("belief has the wrong shape")
        if not 0 <= action < self.n_actions or not 0 <= observation < self.n_observations:
            raise IndexError("action or observation is outside the public registry")
        predicted = prior @ self.transitions[action]
        weighted = predicted * self.emissions[:, observation]
        normalizer = float(weighted.sum())
        if normalizer <= 1e-15:
            raise RuntimeError("history has zero probability under the frozen automaton")
        return weighted / normalizer

    def posterior_for_history(
        self,
        actions: Iterable[int],
        observations: Iterable[int],
    ) -> np.ndarray:
        action_tuple = tuple(int(value) for value in actions)
        observation_tuple = tuple(int(value) for value in observations)
        if len(action_tuple) != len(observation_tuple):
            raise ValueError("history actions and observations must have equal length")
        belief = self.initial_belief.copy()
        for action, observation in zip(action_tuple, observation_tuple, strict=True):
            belief = self.posterior_update(belief, action, observation)
        return belief

    def test_vector(self, test: PredictiveTest) -> np.ndarray:
        if not test.actions:
            raise ValueError("future tests must contain at least one action")
        if not 0 <= test.observation < self.n_observations:
            raise IndexError("future-test observation is outside the registry")
        matrix = np.eye(self.n_states, dtype=np.float64)
        for action in test.actions:
            if not 0 <= action < self.n_actions:
                raise IndexError("future-test action is outside the registry")
            matrix = matrix @ self.transitions[action]
        return matrix @ self.emissions[:, test.observation]

    def test_probability(self, belief: np.ndarray, test: PredictiveTest) -> float:
        return float(np.asarray(belief, dtype=np.float64) @ self.test_vector(test))

    def candidate_tests(self, horizons: Iterable[int]) -> tuple[PredictiveTest, ...]:
        registry: list[PredictiveTest] = []
        for horizon in sorted(set(int(value) for value in horizons)):
            if horizon < 1:
                raise ValueError("candidate test horizons must be positive")
            for actions in product(range(self.n_actions), repeat=horizon):
                for observation in range(self.n_observations):
                    registry.append(PredictiveTest(actions=tuple(actions), observation=observation))
        return tuple(registry)

    def build_basis(
        self,
        *,
        horizons: Iterable[int],
        n_target_tests: int,
        target_seed: int,
        max_condition_number: float,
        max_target_l1: float,
    ) -> PredictiveBasis:
        candidates = self.candidate_tests(horizons)
        vectors = np.stack([self.test_vector(test) for test in candidates], axis=1)
        if int(np.linalg.matrix_rank(vectors)) < self.n_states:
            raise ValueError("candidate future tests do not span the latent predictive rank")

        selected: list[int] = []
        current = np.empty((self.n_states, 0), dtype=np.float64)
        for target_rank in range(1, self.n_states + 1):
            best: tuple[tuple[float, float, float, float], int, np.ndarray] | None = None
            for index, test in enumerate(candidates):
                if index in selected:
                    continue
                proposed = np.column_stack([current, vectors[:, index]])
                if int(np.linalg.matrix_rank(proposed, tol=1e-11)) != target_rank:
                    continue
                singular = np.linalg.svd(proposed, compute_uv=False)
                diversity = float(
                    len({candidates[item].observation for item in (*selected, index)})
                )
                horizon_diversity = float(
                    len({len(candidates[item].actions) for item in (*selected, index)})
                )
                score = (
                    float(singular[-1] / singular[0]),
                    diversity,
                    horizon_diversity,
                    -float(len(test.actions)),
                )
                if best is None or score > best[0]:
                    best = (score, index, proposed)
            if best is None:
                raise RuntimeError("greedy core-test selection could not reach full rank")
            _, index, current = best
            selected.append(index)

        core_matrix = current
        condition = float(np.linalg.cond(core_matrix))
        if condition > max_condition_number:
            raise ValueError(
                f"selected public basis condition {condition:.6f} exceeds {max_condition_number}"
            )

        remaining: list[tuple[int, float]] = []
        for index, _test in enumerate(candidates):
            if index in selected:
                continue
            coefficient = np.linalg.solve(core_matrix, vectors[:, index])
            sensitivity = float(np.sum(np.abs(coefficient)))
            if sensitivity <= max_target_l1:
                remaining.append((index, sensitivity))
        if len(remaining) < n_target_tests:
            raise ValueError("too few stable non-core tests remain for the target bank")

        rng = np.random.default_rng(target_seed)
        chosen_targets: list[int] = []
        strata: dict[tuple[int, int], list[int]] = {}
        for index, _ in remaining:
            test = candidates[index]
            strata.setdefault((len(test.actions), test.observation), []).append(index)
        stratum_keys = sorted(strata)
        for values in strata.values():
            rng.shuffle(values)
        while len(chosen_targets) < n_target_tests:
            progressed = False
            for key in stratum_keys:
                values = strata[key]
                if values and len(chosen_targets) < n_target_tests:
                    chosen_targets.append(values.pop())
                    progressed = True
            if not progressed:
                break
        if len(chosen_targets) != n_target_tests:
            raise RuntimeError("balanced target-test selection exhausted unexpectedly")

        target_matrix = vectors[:, chosen_targets]
        executor = np.linalg.solve(core_matrix, target_matrix)
        basis = PredictiveBasis(
            core_tests=tuple(candidates[index] for index in selected),
            target_tests=tuple(candidates[index] for index in chosen_targets),
            core_matrix=core_matrix,
            target_matrix=target_matrix,
            executor=executor,
            rank=int(np.linalg.matrix_rank(core_matrix)),
            condition_number=condition,
            maximum_target_l1=float(np.max(np.sum(np.abs(executor), axis=0))),
        )
        basis.validate()
        return basis

    def sample_history(
        self,
        *,
        length: int,
        rng: np.random.Generator,
    ) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
        if length < 1:
            raise ValueError("history length must be positive")
        latent = int(rng.choice(self.n_states, p=self.initial_belief))
        belief = self.initial_belief.copy()
        actions: list[int] = []
        observations: list[int] = []
        for _ in range(length):
            action = int(rng.integers(0, self.n_actions))
            latent = int(rng.choice(self.n_states, p=self.transitions[action, latent]))
            observation = int(rng.choice(self.n_observations, p=self.emissions[latent]))
            belief = self.posterior_update(belief, action, observation)
            actions.append(action)
            observations.append(observation)
        return tuple(actions), tuple(observations), belief

    @staticmethod
    def entropy(belief: np.ndarray) -> float:
        values = np.clip(np.asarray(belief, dtype=np.float64), 1e-15, 1.0)
        return float(-np.sum(values * np.log(values)))

    def make_history_record(
        self,
        *,
        history_id: int,
        role: str,
        actions: tuple[int, ...],
        observations: tuple[int, ...],
        basis: PredictiveBasis,
    ) -> HistoryRecord:
        posterior = self.posterior_for_history(actions, observations)
        core = posterior @ basis.core_matrix
        targets = posterior @ basis.target_matrix
        compiled = basis.execute(core, clip=False)
        if not np.allclose(compiled, targets, atol=1e-10, rtol=0.0):
            raise RuntimeError("oracle predictive state does not exactly execute target tests")
        return HistoryRecord(
            history_id=int(history_id),
            role=str(role),
            actions=actions,
            observations=observations,
            posterior=tuple(float(value) for value in posterior),
            core_probabilities=tuple(float(value) for value in core),
            target_probabilities=tuple(float(value) for value in targets),
            entropy=self.entropy(posterior),
        )
