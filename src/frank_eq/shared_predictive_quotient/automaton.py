"""Controlled stochastic systems and the exact shared predictive quotient."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable, Mapping

import numpy as np

SELECTION_SCORE_DECIMALS = 14
BASIS_REGISTRY_DECIMALS = 10


def _selection_score(value: float) -> float:
    """Remove backend-level LAPACK noise before registry comparisons.

    Candidate registries contain exact or near-exact symmetry classes.  Raw
    singular values can differ in their final bits across Accelerate, OpenBLAS,
    and MKL, which must not silently change a frozen public coordinate.  The
    scientific conditioning margin is many orders of magnitude wider than this
    comparison precision; registry order provides the deterministic tie break.
    """

    return round(float(value), SELECTION_SCORE_DECIMALS)


def _canonicalize_basis_registry_value(value: Any) -> Any:
    """Canonicalize numerical basis evidence for cross-platform hashing.

    Runtime arrays retain float64 precision.  The plan registry deliberately
    hashes a 10-decimal representation because LAPACK backends can disagree in
    the final bits of SVD/pseudoinverse outputs.  This representation is used
    only for registry identity; executed arrays retain full float64 precision.
    """

    if isinstance(value, float):
        rounded = round(value, BASIS_REGISTRY_DECIMALS)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, list):
        return [_canonicalize_basis_registry_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonicalize_basis_registry_value(item) for key, item in value.items()}
    return value


def canonical_basis_registry_payload(basis: SharedPredictiveBasis) -> dict[str, Any]:
    """Return the platform-stable numerical payload bound by the dry-run plan."""

    return _canonicalize_basis_registry_value(basis.to_dict())


@dataclass(frozen=True, slots=True, order=True)
class PredictiveTest:
    """A public future action sequence and terminal observation event."""

    actions: tuple[int, ...]
    observation: int

    def to_dict(self) -> dict[str, Any]:
        return {"actions": list(self.actions), "observation": self.observation}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PredictiveTest:
        return cls(
            actions=tuple(int(value) for value in payload["actions"]),
            observation=int(payload["observation"]),
        )


@dataclass(frozen=True, slots=True)
class ControlledSystem:
    """One known full-support controlled HMM in the SPQ0 system family."""

    system_id: str
    role: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    transitions: np.ndarray
    emissions: np.ndarray
    initial_belief: np.ndarray

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    @property
    def n_actions(self) -> int:
        return len(self.action_names)

    @property
    def n_observations(self) -> int:
        return len(self.observation_names)

    def validate(self, *, minimum_probability: float = 0.0) -> None:
        if self.role not in {"fit", "validation_only"}:
            raise ValueError("controlled-system role must be fit or validation_only")
        if not self.system_id or not all(
            len(set(values)) == len(values)
            for values in (self.state_names, self.action_names, self.observation_names)
        ):
            raise ValueError("system ID and public symbol registries must be non-empty and unique")
        if self.transitions.shape != (self.n_actions, self.n_states, self.n_states):
            raise ValueError("transition tensor has the wrong shape")
        if self.emissions.shape != (self.n_states, self.n_observations):
            raise ValueError("emission matrix has the wrong shape")
        if self.initial_belief.shape != (self.n_states,):
            raise ValueError("initial belief has the wrong shape")
        for label, values in (
            ("transition", self.transitions),
            ("emission", self.emissions),
            ("initial belief", self.initial_belief),
        ):
            if not np.all(np.isfinite(values)) or np.any(values < minimum_probability - 1e-12):
                raise ValueError(f"{label} probabilities violate the full-support contract")
        if not np.allclose(self.transitions.sum(axis=2), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("transition rows must sum to one")
        if not np.allclose(self.emissions.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("emission rows must sum to one")
        if not np.isclose(self.initial_belief.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("initial belief must sum to one")

    def posterior_update(self, belief: np.ndarray, action: int, observation: int) -> np.ndarray:
        prior = np.asarray(belief, dtype=np.float64)
        if prior.shape != (self.n_states,):
            raise ValueError("belief has the wrong shape")
        if not 0 <= action < self.n_actions or not 0 <= observation < self.n_observations:
            raise IndexError("action or observation is outside the public registry")
        predicted = prior @ self.transitions[action]
        weighted = predicted * self.emissions[:, observation]
        normalizer = float(weighted.sum())
        if normalizer <= 0.0:
            raise RuntimeError("history has zero probability under a full-support system")
        return weighted / normalizer

    def posterior_for_history(
        self,
        actions: Iterable[int],
        observations: Iterable[int],
    ) -> np.ndarray:
        action_values = tuple(int(value) for value in actions)
        observation_values = tuple(int(value) for value in observations)
        if len(action_values) != len(observation_values):
            raise ValueError("history actions and observations must have equal length")
        belief = self.initial_belief.copy()
        for action, observation in zip(action_values, observation_values, strict=True):
            belief = self.posterior_update(belief, action, observation)
        return belief

    def test_vector(self, test: PredictiveTest) -> np.ndarray:
        if not test.actions:
            raise ValueError("future tests require at least one action")
        if not 0 <= test.observation < self.n_observations:
            raise IndexError("future-test observation is outside the registry")
        matrix = np.eye(self.n_states, dtype=np.float64)
        for action in test.actions:
            if not 0 <= action < self.n_actions:
                raise IndexError("future-test action is outside the registry")
            matrix = matrix @ self.transitions[action]
        return matrix @ self.emissions[:, test.observation]

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
            action = int(rng.integers(self.n_actions))
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

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "system_id": self.system_id,
            "role": self.role,
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "observation_names": list(self.observation_names),
            "transitions": self.transitions.tolist(),
            "emissions": self.emissions.tolist(),
            "initial_belief": self.initial_belief.tolist(),
        }


@dataclass(frozen=True, slots=True)
class SharedPredictiveBasis:
    """One typed test registry with linear and normalization-aware executors.

    ``exact_rank`` is the homogeneous linear PSR rank.  Because every belief is
    normalized, an executor also knows the null-test probability ``1`` without
    receiving it.  The first ``normalization_aware_dimension`` public tests,
    augmented by that constant, therefore form the rate-minimal affine core.
    Keeping both conventions prevents a redundant linear coordinate from being
    mistaken for an intrinsically necessary message dimension.
    """

    exact_rank: int
    normalization_aware_dimension: int
    public_tests: tuple[PredictiveTest, ...]
    target_tests: tuple[PredictiveTest, ...]
    public_matrices: Mapping[str, np.ndarray]
    target_matrices: Mapping[str, np.ndarray]
    executors: Mapping[int, Mapping[str, np.ndarray]]
    normalization_aware_executors: Mapping[int, Mapping[str, np.ndarray]]
    core_condition_numbers: Mapping[str, float]
    normalization_aware_condition_numbers: Mapping[str, float]
    maximum_target_l1: float
    maximum_normalization_aware_target_l1: float
    maximum_exact_executor_error: float
    maximum_normalization_aware_executor_error: float

    @property
    def core_tests(self) -> tuple[PredictiveTest, ...]:
        return self.public_tests[: self.exact_rank]

    @property
    def maximum_rank(self) -> int:
        return len(self.public_tests)

    def validate(self, *, tolerance: float = 1e-10) -> None:
        if self.exact_rank < 1 or self.maximum_rank < self.exact_rank:
            raise ValueError("shared predictive basis rank registry is invalid")
        if self.normalization_aware_dimension != self.exact_rank - 1:
            raise ValueError("normalization-aware dimension must remove exactly one constant")
        system_ids = set(self.public_matrices)
        if not system_ids or set(self.target_matrices) != system_ids:
            raise ValueError("basis matrices do not cover one common system registry")
        if set(self.executors) != set(range(1, self.maximum_rank + 1)):
            raise ValueError("basis lacks a rank-conditioned executor")
        if set(self.normalization_aware_executors) != set(
            range(1, self.normalization_aware_dimension + 1)
        ):
            raise ValueError("basis lacks a normalization-aware rank-conditioned executor")
        observed_max_l1 = 0.0
        observed_max_error = 0.0
        observed_affine_max_l1 = 0.0
        observed_affine_max_error = 0.0
        for system_id in sorted(system_ids):
            public = np.asarray(self.public_matrices[system_id], dtype=np.float64)
            target = np.asarray(self.target_matrices[system_id], dtype=np.float64)
            if public.shape != (self.exact_rank, self.maximum_rank):
                raise ValueError("public test matrix has the wrong shape")
            if target.shape != (self.exact_rank, len(self.target_tests)):
                raise ValueError("target test matrix has the wrong shape")
            core = public[:, : self.exact_rank]
            if int(np.linalg.matrix_rank(core, tol=tolerance)) != self.exact_rank:
                raise ValueError(f"core tests do not attain exact rank for {system_id}")
            condition = float(np.linalg.cond(core))
            if not np.isclose(
                condition,
                float(self.core_condition_numbers[system_id]),
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ValueError("stored core condition number is inconsistent")
            for rank, by_system in self.executors.items():
                if set(by_system) != system_ids:
                    raise ValueError("rank-conditioned executor system registry changed")
                executor = np.asarray(by_system[system_id], dtype=np.float64)
                if executor.shape != (rank, len(self.target_tests)):
                    raise ValueError("rank-conditioned target executor has the wrong shape")
                if rank >= self.exact_rank:
                    difference = public[:, :rank] @ executor - target
                    observed_max_error = max(
                        observed_max_error,
                        float(np.max(np.abs(difference))),
                    )
            exact_executor = np.asarray(
                self.executors[self.exact_rank][system_id], dtype=np.float64
            )
            observed_max_l1 = max(
                observed_max_l1,
                float(np.max(np.sum(np.abs(exact_executor), axis=0))),
            )
            for rank, by_system in self.normalization_aware_executors.items():
                if set(by_system) != system_ids:
                    raise ValueError("normalization-aware executor system registry changed")
                executor = np.asarray(by_system[system_id], dtype=np.float64)
                if executor.shape != (rank + 1, len(self.target_tests)):
                    raise ValueError("normalization-aware target executor has the wrong shape")
            affine_public = np.column_stack(
                (
                    public[:, : self.normalization_aware_dimension],
                    np.ones(self.exact_rank, dtype=np.float64),
                )
            )
            if int(np.linalg.matrix_rank(affine_public, tol=tolerance)) != self.exact_rank:
                raise ValueError(f"normalization-aware core is not sufficient for {system_id}")
            affine_condition = float(np.linalg.cond(affine_public))
            if not np.isclose(
                affine_condition,
                float(self.normalization_aware_condition_numbers[system_id]),
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ValueError("stored normalization-aware condition number is inconsistent")
            affine_executor = np.asarray(
                self.normalization_aware_executors[self.normalization_aware_dimension][system_id],
                dtype=np.float64,
            )
            affine_difference = affine_public @ affine_executor - target
            observed_affine_max_error = max(
                observed_affine_max_error,
                float(np.max(np.abs(affine_difference))),
            )
            observed_affine_max_l1 = max(
                observed_affine_max_l1,
                float(np.max(np.sum(np.abs(affine_executor), axis=0))),
            )
        if observed_max_error > tolerance:
            raise ValueError("rank-complete public packets do not exactly execute targets")
        if not np.isclose(observed_max_error, self.maximum_exact_executor_error, atol=1e-14):
            raise ValueError("stored exact-executor error is inconsistent")
        if not np.isclose(observed_max_l1, self.maximum_target_l1, atol=1e-12):
            raise ValueError("stored target-executor sensitivity is inconsistent")
        if observed_affine_max_error > tolerance:
            raise ValueError("normalization-aware public packets do not exactly execute targets")
        if not np.isclose(
            observed_affine_max_error,
            self.maximum_normalization_aware_executor_error,
            atol=1e-14,
        ):
            raise ValueError("stored normalization-aware executor error is inconsistent")
        if not np.isclose(
            observed_affine_max_l1,
            self.maximum_normalization_aware_target_l1,
            atol=1e-12,
        ):
            raise ValueError("stored normalization-aware executor sensitivity is inconsistent")

    def public_probabilities(
        self,
        system_id: str,
        belief: np.ndarray,
        *,
        rank: int | None = None,
    ) -> np.ndarray:
        selected_rank = self.maximum_rank if rank is None else int(rank)
        if selected_rank not in self.executors:
            raise ValueError("packet rank is outside the frozen sweep")
        return (
            np.asarray(belief, dtype=np.float64)
            @ self.public_matrices[system_id][:, :selected_rank]
        )

    def decode_core(self, system_id: str, packet: np.ndarray, *, rank: int) -> np.ndarray:
        values = np.asarray(packet, dtype=np.float64)
        if values.shape[-1] != rank:
            raise ValueError("public packet has the wrong rank")
        public = self.public_matrices[system_id][:, :rank]
        core = self.public_matrices[system_id][:, : self.exact_rank]
        return values @ np.linalg.pinv(public, rcond=1e-12) @ core

    def decode_core_normalization_aware(
        self,
        system_id: str,
        packet: np.ndarray,
        *,
        rank: int,
    ) -> np.ndarray:
        """Decode with the known null-test probability as a zero-bit intercept."""

        values = np.asarray(packet, dtype=np.float64)
        if rank not in self.normalization_aware_executors or values.shape[-1] != rank:
            raise ValueError("normalization-aware packet has the wrong rank")
        augmented_values = np.concatenate(
            (values, np.ones((*values.shape[:-1], 1), dtype=np.float64)),
            axis=-1,
        )
        public = np.column_stack(
            (
                self.public_matrices[system_id][:, :rank],
                np.ones(self.exact_rank, dtype=np.float64),
            )
        )
        core = self.public_matrices[system_id][:, : self.exact_rank]
        return augmented_values @ np.linalg.pinv(public, rcond=1e-12) @ core

    def execute_targets(
        self,
        system_id: str,
        packet: np.ndarray,
        *,
        rank: int,
        clip: bool = True,
    ) -> np.ndarray:
        values = np.asarray(packet, dtype=np.float64)
        if values.shape[-1] != rank:
            raise ValueError("public packet has the wrong rank")
        result = values @ self.executors[rank][system_id]
        return np.clip(result, 0.0, 1.0) if clip else result

    def execute_targets_normalization_aware(
        self,
        system_id: str,
        packet: np.ndarray,
        *,
        rank: int,
        clip: bool = True,
    ) -> np.ndarray:
        values = np.asarray(packet, dtype=np.float64)
        if rank not in self.normalization_aware_executors or values.shape[-1] != rank:
            raise ValueError("normalization-aware packet has the wrong rank")
        augmented_values = np.concatenate(
            (values, np.ones((*values.shape[:-1], 1), dtype=np.float64)),
            axis=-1,
        )
        result = augmented_values @ self.normalization_aware_executors[rank][system_id]
        return np.clip(result, 0.0, 1.0) if clip else result

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "frank_eq_shared_predictive_basis_v2",
            "exact_rank": self.exact_rank,
            "normalization_aware_dimension": self.normalization_aware_dimension,
            "public_tests": [test.to_dict() for test in self.public_tests],
            "core_tests": [test.to_dict() for test in self.core_tests],
            "target_tests": [test.to_dict() for test in self.target_tests],
            "public_matrices": {
                key: self.public_matrices[key].tolist() for key in sorted(self.public_matrices)
            },
            "target_matrices": {
                key: self.target_matrices[key].tolist() for key in sorted(self.target_matrices)
            },
            "executors": {
                str(rank): {
                    key: self.executors[rank][key].tolist() for key in sorted(self.executors[rank])
                }
                for rank in sorted(self.executors)
            },
            "normalization_aware_executors": {
                str(rank): {
                    key: self.normalization_aware_executors[rank][key].tolist()
                    for key in sorted(self.normalization_aware_executors[rank])
                }
                for rank in sorted(self.normalization_aware_executors)
            },
            "core_condition_numbers": dict(sorted(self.core_condition_numbers.items())),
            "normalization_aware_condition_numbers": dict(
                sorted(self.normalization_aware_condition_numbers.items())
            ),
            "maximum_target_l1": self.maximum_target_l1,
            "maximum_normalization_aware_target_l1": (
                self.maximum_normalization_aware_target_l1
            ),
            "maximum_exact_executor_error": self.maximum_exact_executor_error,
            "maximum_normalization_aware_executor_error": (
                self.maximum_normalization_aware_executor_error
            ),
        }


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    """One role-bound history and its exact externally defined predictive state."""

    history_id: int
    system_id: str
    system_role: str
    role: str
    actions: tuple[int, ...]
    observations: tuple[int, ...]
    posterior: tuple[float, ...]
    public_probabilities: tuple[float, ...]
    core_probabilities: tuple[float, ...]
    target_probabilities: tuple[float, ...]
    entropy: float

    @property
    def length(self) -> int:
        return len(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HistoryRecord:
        return cls(
            history_id=int(payload["history_id"]),
            system_id=str(payload["system_id"]),
            system_role=str(payload["system_role"]),
            role=str(payload["role"]),
            actions=tuple(int(value) for value in payload["actions"]),
            observations=tuple(int(value) for value in payload["observations"]),
            posterior=tuple(float(value) for value in payload["posterior"]),
            public_probabilities=tuple(float(value) for value in payload["public_probabilities"]),
            core_probabilities=tuple(float(value) for value in payload["core_probabilities"]),
            target_probabilities=tuple(float(value) for value in payload["target_probabilities"]),
            entropy=float(payload["entropy"]),
        )


def candidate_tests(
    *,
    n_actions: int,
    n_observations: int,
    horizons: Iterable[int],
) -> tuple[PredictiveTest, ...]:
    registry: list[PredictiveTest] = []
    for horizon in sorted(set(int(value) for value in horizons)):
        if horizon < 1:
            raise ValueError("candidate future-test horizons must be positive")
        for actions in product(range(n_actions), repeat=horizon):
            for observation in range(n_observations):
                registry.append(PredictiveTest(tuple(actions), observation))
    return tuple(registry)


def _full_support_rows(
    rng: np.random.Generator,
    shape: tuple[int, int],
    minimum_probability: float,
) -> np.ndarray:
    columns = shape[1]
    if not 0.0 < minimum_probability < 1.0 / columns:
        raise ValueError("minimum probability is incompatible with row width")
    # Sparse Dirichlet draws produce observably distinct states while the
    # affine floor preserves full support. The draw is part of the frozen
    # system generator, not fitted from any model response.
    raw = rng.dirichlet(np.full(columns, 0.07), size=shape[0])
    return minimum_probability + (1.0 - columns * minimum_probability) * raw


def generate_system_family(
    *,
    latent_states: int,
    actions: int,
    observations: int,
    fit_systems: int,
    validation_only_systems: int,
    validation_parent_weight: float,
    minimum_probability: float,
    seed: int,
) -> tuple[ControlledSystem, ...]:
    """Generate the frozen transition/emission family without reading any model."""

    if (latent_states, actions, observations) != (4, 3, 3):
        raise ValueError("SPQ0 is frozen to four states, three actions, and three observations")
    if fit_systems < 2 or validation_only_systems < 1:
        raise ValueError("SPQ0 needs two fit systems and a validation-only system")
    if not 0.0 < validation_parent_weight < 1.0:
        raise ValueError("validation parent weight must be strictly between zero and one")
    rng = np.random.default_rng(seed)
    raw_systems: list[tuple[np.ndarray, np.ndarray]] = []
    total = fit_systems + validation_only_systems
    for _index in range(total):
        transition_rows: list[np.ndarray] = []
        for _ in range(actions):
            permutation = rng.permutation(latent_states)
            matrix = np.full(
                (latent_states, latent_states),
                minimum_probability,
                dtype=np.float64,
            )
            matrix[np.arange(latent_states), permutation] = (
                1.0 - (latent_states - 1) * minimum_probability
            )
            transition_rows.append(matrix)
        transitions = np.stack(transition_rows)
        emissions = _full_support_rows(
            rng,
            (latent_states, observations),
            minimum_probability,
        )
        raw_systems.append((transitions, emissions))

    # A validation system is a prospectively frozen 10% perturbation of a fit
    # system by an independent full-support draw.  This gives a genuinely new
    # transition/emission law while guaranteeing that the shared public test
    # registry remains numerically meaningful without inspecting any model
    # response or using validation rows for basis selection.
    systems: list[ControlledSystem] = []
    for index, (transitions, emissions) in enumerate(raw_systems):
        if index >= fit_systems:
            parent_transitions, parent_emissions = raw_systems[(index - fit_systems) % fit_systems]
            transitions = (
                validation_parent_weight * parent_transitions
                + (1.0 - validation_parent_weight) * transitions
            )
            emissions = (
                validation_parent_weight * parent_emissions
                + (1.0 - validation_parent_weight) * emissions
            )
        system = ControlledSystem(
            system_id=f"system-{index:02d}",
            role="fit" if index < fit_systems else "validation_only",
            state_names=tuple(f"S{state}" for state in range(latent_states)),
            action_names=("orbit", "fold", "shift"),
            observation_names=("amber", "blue", "coral"),
            transitions=transitions,
            emissions=emissions,
            initial_belief=np.full(latent_states, 1.0 / latent_states),
        )
        system.validate(minimum_probability=minimum_probability)
        systems.append(system)
    return tuple(systems)


def validation_shift_summary(
    systems: tuple[ControlledSystem, ...],
    *,
    parent_weight: float,
) -> dict[str, Any]:
    """Quantify the frozen held-law perturbation without model responses."""

    fit_systems = tuple(system for system in systems if system.role == "fit")
    validation_systems = tuple(
        system for system in systems if system.role == "validation_only"
    )
    if not fit_systems or not validation_systems:
        raise ValueError("validation-shift summary requires fit and validation systems")
    result: dict[str, Any] = {}
    for offset, system in enumerate(validation_systems):
        parent = fit_systems[offset % len(fit_systems)]
        result[system.system_id] = {
            "parent_system_id": parent.system_id,
            "parent_weight": float(parent_weight),
            "independent_draw_weight": float(round(1.0 - parent_weight, 15)),
            "transition_mean_row_total_variation": float(
                np.mean(np.sum(np.abs(system.transitions - parent.transitions), axis=2) / 2.0)
            ),
            "emission_mean_row_total_variation": float(
                np.mean(np.sum(np.abs(system.emissions - parent.emissions), axis=1) / 2.0)
            ),
        }
    return result


def _select_shared_public_tests(
    systems: tuple[ControlledSystem, ...],
    candidates: tuple[PredictiveTest, ...],
    vectors: Mapping[str, np.ndarray],
    *,
    exact_rank: int,
    maximum_rank: int,
) -> list[int]:
    selected: list[int] = []
    for target_rank in range(1, exact_rank + 1):
        best: tuple[tuple[float, float, float, float], int] | None = None
        for index, test in enumerate(candidates):
            if index in selected:
                continue
            proposed_indices = [*selected, index]
            singular_ratios: list[float] = []
            valid = True
            for system in systems:
                proposed = vectors[system.system_id][:, proposed_indices]
                if int(np.linalg.matrix_rank(proposed, tol=1e-11)) != target_rank:
                    valid = False
                    break
                singular = np.linalg.svd(proposed, compute_uv=False)
                singular_ratios.append(float(singular[-1] / singular[0]))
            if not valid:
                continue
            registry = [candidates[item] for item in proposed_indices]
            score = (
                _selection_score(min(singular_ratios)),
                float(len({item.observation for item in registry})),
                float(len({len(item.actions) for item in registry})),
                -float(len(test.actions)),
            )
            if best is None or score > best[0]:
                best = (score, index)
        if best is None:
            raise RuntimeError("shared public-test selection could not attain exact rank")
        selected.append(best[1])

    while len(selected) < maximum_rank:
        best_extra: tuple[tuple[float, float, float], int] | None = None
        for index, test in enumerate(candidates):
            if index in selected:
                continue
            proposed_indices = [*selected, index]
            worst_condition = max(
                float(np.linalg.cond(vectors[system.system_id][:, proposed_indices]))
                for system in systems
            )
            registry = [candidates[item] for item in proposed_indices]
            score = (
                _selection_score(-worst_condition),
                float(len({(len(item.actions), item.observation) for item in registry})),
                -float(len(test.actions)),
            )
            if best_extra is None or score > best_extra[0]:
                best_extra = (score, index)
        if best_extra is None:
            raise RuntimeError("overcomplete public-test selection exhausted unexpectedly")
        selected.append(best_extra[1])
    return selected


def build_shared_predictive_basis(
    systems: tuple[ControlledSystem, ...],
    *,
    horizons: Iterable[int],
    exact_rank: int,
    maximum_rank: int,
    n_target_tests: int,
    target_seed: int,
    max_core_condition_number: float,
    max_target_l1: float,
    normalization_aware_dimension: int | None = None,
    max_normalization_aware_condition_number: float | None = None,
    max_normalization_aware_target_l1: float | None = None,
) -> SharedPredictiveBasis:
    """Select one rank-conditioned typed basis that is exact for every system."""

    if not systems or len({system.system_id for system in systems}) != len(systems):
        raise ValueError("system family must be non-empty with unique IDs")
    reference = systems[0]
    if any(
        (system.n_states, system.n_actions, system.n_observations)
        != (reference.n_states, reference.n_actions, reference.n_observations)
        for system in systems
    ):
        raise ValueError("all SPQ0 systems must share the public symbol registry")
    if exact_rank != reference.n_states or maximum_rank < exact_rank:
        raise ValueError("exact and maximum packet ranks are inconsistent with the systems")
    affine_dimension = (
        exact_rank - 1
        if normalization_aware_dimension is None
        else int(normalization_aware_dimension)
    )
    if affine_dimension != exact_rank - 1:
        raise ValueError("normalization-aware packet must use rank minus one coordinates")
    candidates = candidate_tests(
        n_actions=reference.n_actions,
        n_observations=reference.n_observations,
        horizons=horizons,
    )
    vectors = {
        system.system_id: np.stack([system.test_vector(test) for test in candidates], axis=1)
        for system in systems
    }
    if any(int(np.linalg.matrix_rank(matrix)) < exact_rank for matrix in vectors.values()):
        raise ValueError("candidate future tests do not span every system's predictive rank")
    selection_systems = tuple(system for system in systems if system.role == "fit")
    if not selection_systems:
        raise ValueError("shared public tests require at least one fit-role system")
    public_indices = _select_shared_public_tests(
        selection_systems,
        candidates,
        vectors,
        exact_rank=exact_rank,
        maximum_rank=maximum_rank,
    )
    condition_numbers = {
        system.system_id: float(
            np.linalg.cond(vectors[system.system_id][:, public_indices[:exact_rank]])
        )
        for system in systems
    }
    if max(condition_numbers.values()) > max_core_condition_number:
        raise ValueError("shared exact core exceeds the frozen worst-system condition-number bound")

    stable_candidates: list[int] = []
    for index in range(len(candidates)):
        if index in public_indices:
            continue
        stable = True
        for system in selection_systems:
            core = vectors[system.system_id][:, public_indices[:exact_rank]]
            coefficient = np.linalg.solve(core, vectors[system.system_id][:, index])
            if _selection_score(float(np.sum(np.abs(coefficient)))) > max_target_l1:
                stable = False
                break
        if stable:
            stable_candidates.append(index)
    if len(stable_candidates) < n_target_tests:
        raise ValueError("too few cross-system stable target tests remain")

    rng = np.random.default_rng(target_seed)
    strata: dict[tuple[int, int], list[int]] = {}
    for index in stable_candidates:
        test = candidates[index]
        strata.setdefault((len(test.actions), test.observation), []).append(index)
    for values in strata.values():
        rng.shuffle(values)
    target_indices: list[int] = []
    stratum_keys = sorted(strata)
    while len(target_indices) < n_target_tests:
        progressed = False
        for key in stratum_keys:
            if strata[key] and len(target_indices) < n_target_tests:
                target_indices.append(strata[key].pop())
                progressed = True
        if not progressed:
            break
    if len(target_indices) != n_target_tests:
        raise RuntimeError("balanced cross-system target selection exhausted unexpectedly")

    public_matrices = {
        system.system_id: vectors[system.system_id][:, public_indices] for system in systems
    }
    target_matrices = {
        system.system_id: vectors[system.system_id][:, target_indices] for system in systems
    }
    executors: dict[int, dict[str, np.ndarray]] = {}
    maximum_error = 0.0
    for rank in range(1, maximum_rank + 1):
        executors[rank] = {}
        for system in systems:
            public = public_matrices[system.system_id][:, :rank]
            target = target_matrices[system.system_id]
            executor = np.linalg.pinv(public, rcond=1e-12) @ target
            executors[rank][system.system_id] = executor
            if rank >= exact_rank:
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(public @ executor - target))),
                )
    maximum_l1 = max(
        float(np.max(np.sum(np.abs(executors[exact_rank][system.system_id]), axis=0)))
        for system in systems
    )
    normalization_aware_executors: dict[int, dict[str, np.ndarray]] = {}
    for rank in range(1, affine_dimension + 1):
        normalization_aware_executors[rank] = {}
        for system in systems:
            public = np.column_stack(
                (
                    public_matrices[system.system_id][:, :rank],
                    np.ones(exact_rank, dtype=np.float64),
                )
            )
            normalization_aware_executors[rank][system.system_id] = (
                np.linalg.pinv(public, rcond=1e-12) @ target_matrices[system.system_id]
            )
    normalization_aware_conditions = {
        system.system_id: float(
            np.linalg.cond(
                np.column_stack(
                    (
                        public_matrices[system.system_id][:, :affine_dimension],
                        np.ones(exact_rank, dtype=np.float64),
                    )
                )
            )
        )
        for system in systems
    }
    if (
        max_normalization_aware_condition_number is not None
        and max(normalization_aware_conditions.values())
        > max_normalization_aware_condition_number
    ):
        raise ValueError("normalization-aware core exceeds its condition-number bound")
    normalization_aware_maximum_l1 = max(
        float(
            np.max(
                np.sum(
                    np.abs(
                        normalization_aware_executors[affine_dimension][system.system_id]
                    ),
                    axis=0,
                )
            )
        )
        for system in systems
    )
    if (
        max_normalization_aware_target_l1 is not None
        and normalization_aware_maximum_l1 > max_normalization_aware_target_l1
    ):
        raise ValueError("normalization-aware executor exceeds its L1 bound")
    normalization_aware_maximum_error = max(
        float(
            np.max(
                np.abs(
                    np.column_stack(
                        (
                            public_matrices[system.system_id][:, :affine_dimension],
                            np.ones(exact_rank, dtype=np.float64),
                        )
                    )
                    @ normalization_aware_executors[affine_dimension][system.system_id]
                    - target_matrices[system.system_id]
                )
            )
        )
        for system in systems
    )
    basis = SharedPredictiveBasis(
        exact_rank=exact_rank,
        normalization_aware_dimension=affine_dimension,
        public_tests=tuple(candidates[index] for index in public_indices),
        target_tests=tuple(candidates[index] for index in target_indices),
        public_matrices=public_matrices,
        target_matrices=target_matrices,
        executors=executors,
        normalization_aware_executors=normalization_aware_executors,
        core_condition_numbers=condition_numbers,
        normalization_aware_condition_numbers=normalization_aware_conditions,
        maximum_target_l1=maximum_l1,
        maximum_normalization_aware_target_l1=normalization_aware_maximum_l1,
        maximum_exact_executor_error=maximum_error,
        maximum_normalization_aware_executor_error=normalization_aware_maximum_error,
    )
    basis.validate()
    return basis


def make_history_record(
    system: ControlledSystem,
    basis: SharedPredictiveBasis,
    *,
    history_id: int,
    role: str,
    actions: tuple[int, ...],
    observations: tuple[int, ...],
) -> HistoryRecord:
    posterior = system.posterior_for_history(actions, observations)
    public = basis.public_probabilities(system.system_id, posterior)
    core = public[: basis.exact_rank]
    targets = posterior @ basis.target_matrices[system.system_id]
    compiled = basis.execute_targets(
        system.system_id,
        core,
        rank=basis.exact_rank,
        clip=False,
    )
    if not np.allclose(compiled, targets, atol=1e-10, rtol=0.0):
        raise RuntimeError("exact public core does not reproduce target probabilities")
    return HistoryRecord(
        history_id=int(history_id),
        system_id=system.system_id,
        system_role=system.role,
        role=role,
        actions=actions,
        observations=observations,
        posterior=tuple(float(value) for value in posterior),
        public_probabilities=tuple(float(value) for value in public),
        core_probabilities=tuple(float(value) for value in core),
        target_probabilities=tuple(float(value) for value in targets),
        entropy=system.entropy(posterior),
    )
