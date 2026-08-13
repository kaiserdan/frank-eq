"""Deterministic SPQ0 encoders, readers, controls, and grouped uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .automaton import PredictiveTest


@dataclass(frozen=True, slots=True)
class LinearMap:
    """A centered ridge map with an optional truncated-ridge coefficient matrix."""

    weights: np.ndarray
    feature_mean: np.ndarray
    target_mean: np.ndarray
    ridge: float
    method: str
    coefficient_rank: int

    def predict(self, features: np.ndarray, *, clip: bool = True) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.weights.shape[0]:
            raise ValueError("linear-map features have the wrong shape")
        prediction = (values - self.feature_mean) @ self.weights + self.target_mean
        return np.clip(prediction, 1e-7, 1.0 - 1e-7) if clip else prediction

    @property
    def learned_parameter_count(self) -> int:
        return int(self.weights.size + self.target_mean.size)

    def metadata(self) -> dict[str, Any]:
        return {
            "input_dim": int(self.weights.shape[0]),
            "output_dim": int(self.weights.shape[1]),
            "ridge": self.ridge,
            "method": self.method,
            "coefficient_rank": self.coefficient_rank,
            "learned_parameter_count": self.learned_parameter_count,
            "weight_norm": float(np.linalg.norm(self.weights)),
            "feature_mean_norm": float(np.linalg.norm(self.feature_mean)),
            "target_mean_norm": float(np.linalg.norm(self.target_mean)),
        }

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}__weights": self.weights,
            f"{prefix}__feature_mean": self.feature_mean,
            f"{prefix}__target_mean": self.target_mean,
            f"{prefix}__ridge": np.asarray([self.ridge], dtype=np.float64),
            f"{prefix}__coefficient_rank": np.asarray([self.coefficient_rank], dtype=np.int64),
        }


def fit_linear_map(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
    method: str = "ridge",
    maximum_coefficient_rank: int | None = None,
) -> LinearMap:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("linear fitting requires paired rank-two arrays")
    if ridge <= 0 or method not in {"ridge", "truncated_ridge"}:
        raise ValueError("linear fitting has an invalid ridge or method")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("linear fitting inputs must be finite")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    if x.shape[1] <= x.shape[0]:
        gram = xc.T @ xc
        gram.flat[:: gram.shape[0] + 1] += ridge
        right = xc.T @ yc
        try:
            weights = np.linalg.solve(gram, right)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(gram, rcond=1e-12) @ right
    else:
        kernel = xc @ xc.T
        kernel.flat[:: kernel.shape[0] + 1] += ridge
        try:
            dual = np.linalg.solve(kernel, yc)
        except np.linalg.LinAlgError:
            dual = np.linalg.pinv(kernel, rcond=1e-12) @ yc
        weights = xc.T @ dual
    coefficient_rank = int(np.linalg.matrix_rank(weights, tol=1e-12))
    if method == "truncated_ridge":
        bound = maximum_coefficient_rank or min(x.shape[1], y.shape[1])
        bound = max(1, min(int(bound), min(weights.shape)))
        left, singular, right = np.linalg.svd(weights, full_matrices=False)
        weights = (left[:, :bound] * singular[:bound]) @ right[:bound]
        coefficient_rank = int(bound)
    return LinearMap(
        weights=np.asarray(weights, dtype=np.float64),
        feature_mean=x_mean,
        target_mean=y_mean,
        ridge=float(ridge),
        method=method,
        coefficient_rank=coefficient_rank,
    )


def brier_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    truth = np.asarray(targets, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    if truth.shape != estimate.shape:
        raise ValueError("Brier arrays must have identical shapes")
    return float(np.mean((truth - estimate) ** 2))


def row_brier(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    truth = np.asarray(targets, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim < 2:
        raise ValueError("row Brier arrays must be matching rank-two-or-higher arrays")
    axes = tuple(range(1, truth.ndim))
    return np.mean((truth - estimate) ** 2, axis=axes)


def r2_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    truth = np.asarray(targets, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    if truth.shape != estimate.shape:
        raise ValueError("R2 arrays must have identical shapes")
    residual = float(np.sum((truth - estimate) ** 2))
    centered = float(np.sum((truth - truth.mean(axis=0, keepdims=True)) ** 2))
    return 0.0 if centered <= 1e-15 else float(1.0 - residual / centered)


def aggregate_by_group(
    values: np.ndarray,
    group_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    identities = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    if array.shape[0] != identities.shape[0]:
        raise ValueError("values and group IDs have different row counts")
    unique = np.unique(identities)
    grouped = np.asarray([array[identities == value].mean(axis=0) for value in unique])
    return unique, grouped


def bootstrap_interval(
    grouped_values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(grouped_values, dtype=np.float64).reshape(-1)
    if values.size < 2 or replicates < 1:
        raise ValueError("bootstrap interval requires at least two units")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    samples = values[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "replicates": int(replicates),
        "units": int(values.size),
    }


def paired_brier_gain_interval(
    targets: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    group_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    gains = row_brier(targets, baseline) - row_brier(targets, candidate)
    _, grouped = aggregate_by_group(gains, group_ids)
    return bootstrap_interval(grouped, replicates=replicates, seed=seed)


def paired_value_interval(
    candidate_values: np.ndarray,
    baseline_values: np.ndarray,
    group_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    difference = np.asarray(candidate_values, dtype=np.float64) - np.asarray(
        baseline_values, dtype=np.float64
    )
    _, grouped = aggregate_by_group(difference, group_ids)
    return bootstrap_interval(grouped, replicates=replicates, seed=seed)


def softmax(values: np.ndarray, *, axis: int = -1) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    centered = scores - np.max(scores, axis=axis, keepdims=True)
    exponent = np.exp(np.clip(centered, -700.0, 0.0))
    return exponent / np.sum(exponent, axis=axis, keepdims=True)


def categorical_distribution(log_likelihoods: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("categorical temperature must be positive")
    return softmax(np.asarray(log_likelihoods, dtype=np.float64) / temperature)


def categorical_expectation(distribution: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values = np.asarray(distribution, dtype=np.float64)
    bin_values = np.asarray(bins, dtype=np.float64)
    if values.shape[-1] != bin_values.size:
        raise ValueError("categorical distribution and bin registry differ")
    return values @ bin_values


def select_categorical_temperature(
    log_likelihoods: np.ndarray,
    semantic_probabilities: np.ndarray,
    bins: np.ndarray,
    temperature_grid: Iterable[float],
) -> dict[str, Any]:
    logits = np.asarray(log_likelihoods, dtype=np.float64)
    truth = np.asarray(semantic_probabilities, dtype=np.float64)
    if logits.shape[:-1] != truth.shape:
        raise ValueError("temperature fitting logits and semantic targets differ")
    candidates: list[dict[str, float]] = []
    for temperature in temperature_grid:
        distribution = categorical_distribution(logits, float(temperature))
        expectation = categorical_expectation(distribution, bins)
        candidates.append(
            {
                "temperature": float(temperature),
                "calibration_brier": float(np.mean((expectation - truth) ** 2)),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (row["calibration_brier"], row["temperature"]),
    )
    return {"selected_temperature": selected["temperature"], "candidates": candidates}


def deterministic_token_hash_features(
    token_ids: np.ndarray,
    attention_mask: np.ndarray,
    *,
    width: int,
    position_period: int,
) -> np.ndarray:
    """A fixed order-sensitive transcript control with no model activation."""

    tokens = np.asarray(token_ids, dtype=np.int64)
    mask = np.asarray(attention_mask, dtype=bool)
    if tokens.ndim != 2 or mask.shape != tokens.shape:
        raise ValueError("token hash expects matching token and mask matrices")
    if width < 1 or position_period < 1:
        raise ValueError("token hash width and position period must be positive")
    rows = np.zeros((tokens.shape[0], width), dtype=np.float64)
    for row_index in range(tokens.shape[0]):
        positions = np.flatnonzero(mask[row_index])
        if positions.size == 0:
            raise ValueError("token hash cannot encode an empty prefix")
        length = float(positions.size)
        for order, position in enumerate(positions):
            token = int(tokens[row_index, position])
            bucket = int(
                (
                    token * 1_315_423_911
                    + (order % position_period) * 2_654_435_761
                    + (order // position_period) * 97_531
                )
                % width
            )
            sign = -1.0 if ((token * 31 + order * 17) & 1) else 1.0
            recency = 0.5 + 0.5 * (order + 1) / length
            rows[row_index, bucket] += sign * recency
        rows[row_index] /= np.sqrt(length)
    return rows


def parameter_matched_token_sequence_features(
    token_ids: np.ndarray,
    attention_mask: np.ndarray,
    event_token_indices: np.ndarray,
    *,
    width: int,
    decay_grid: Iterable[float],
) -> np.ndarray:
    """Fixed causal sequence features whose fitted ridge map matches activation parameters.

    The feature map retains order through multiple causal decay traces and
    explicit event-boundary channels. Its only learned parameters are the
    downstream ridge coefficients, exactly ``width * packet_rank`` just like a
    width-matched activation ridge encoder.
    """

    tokens = np.asarray(token_ids, dtype=np.int64)
    mask = np.asarray(attention_mask, dtype=bool)
    boundaries = np.asarray(event_token_indices, dtype=np.int64)
    decays = tuple(float(value) for value in decay_grid)
    if tokens.ndim != 2 or mask.shape != tokens.shape or boundaries.ndim != 2:
        raise ValueError("parameter-matched token sequence arrays have invalid ranks")
    if tokens.shape[0] != boundaries.shape[0] or width < len(decays) or not decays:
        raise ValueError("parameter-matched token sequence dimensions are invalid")
    if any(not 0.0 < value < 1.0 for value in decays):
        raise ValueError("causal token-sequence decays must lie in (0,1)")
    result = np.zeros((tokens.shape[0], width), dtype=np.float64)
    channel_width = width // len(decays)
    if channel_width < 1:
        raise ValueError("token-sequence width is too small for its decay registry")
    for row in range(tokens.shape[0]):
        positions = np.flatnonzero(mask[row])
        if positions.size == 0:
            raise ValueError("token sequence cannot encode an empty prefix")
        boundary_set = set(int(value) for value in boundaries[row] if value >= 0)
        last = int(positions[-1])
        for channel, decay in enumerate(decays):
            base = channel * channel_width
            limit = width if channel == len(decays) - 1 else base + channel_width
            local_width = limit - base
            for order, position in enumerate(positions):
                token = int(tokens[row, position])
                distance = last - int(position)
                boundary = int(position) in boundary_set
                bucket = base + int(
                    (token * 2_654_435_761 + order * 1_315_423_911 + int(boundary) * 805_459_861)
                    % local_width
                )
                sign = -1.0 if ((token * 13 + order * 29 + channel) & 1) else 1.0
                weight = decay**distance
                if boundary:
                    weight *= 2.0
                result[row, bucket] += sign * weight
            norm = float(np.linalg.norm(result[row, base:limit]))
            if norm > 0:
                result[row, base:limit] /= norm
    return result


def quantize_probabilities(values: np.ndarray, bits: int) -> np.ndarray:
    if not 1 <= bits <= 16:
        raise ValueError("probability quantization bits must lie in [1,16]")
    levels = (1 << bits) - 1
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.round(clipped * levels) / levels


def wrong_history_margin_interval(
    targets: np.ndarray,
    predictions: np.ndarray,
    group_ids: np.ndarray,
    strata: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    truth = np.asarray(targets, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    identities = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    strata_values = np.asarray(strata, dtype=np.int64).reshape(-1)
    if truth.shape != estimate.shape or truth.shape[0] != identities.size:
        raise ValueError("wrong-history inputs have inconsistent row counts")
    unique, grouped_truth = aggregate_by_group(truth, identities)
    _, grouped_prediction = aggregate_by_group(estimate, identities)
    grouped_strata = np.asarray(
        [int(np.unique(strata_values[identities == identity])[0]) for identity in unique]
    )
    wrong_truth = np.empty_like(grouped_truth)
    for stratum in np.unique(grouped_strata):
        positions = np.flatnonzero(grouped_strata == stratum)
        if positions.size < 2:
            raise ValueError("wrong-history specificity needs two units per stratum")
        wrong_truth[positions] = grouped_truth[np.roll(positions, 1)]
    correct = row_brier(grouped_truth, grouped_prediction)
    wrong = row_brier(wrong_truth, grouped_prediction)
    return bootstrap_interval(wrong - correct, replicates=replicates, seed=seed)


def project_simplex(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-9, None)
    return clipped / clipped.sum(axis=-1, keepdims=True)


def target_reader_features(
    cores: np.ndarray,
    executed_targets: np.ndarray,
    target_tests: tuple[PredictiveTest, ...],
) -> np.ndarray:
    """Build reader features from public state and deterministic public execution.

    ``executed_targets`` must be the output of the frozen system-local public
    executor. During fitting it is exact because the core is oracle; during
    source evaluation it is recomputed from the learned source packet. It is
    therefore a deterministic feature of the public state, not an oracle label
    supplied to a transferred packet.
    """

    core = np.asarray(cores, dtype=np.float64)
    executed = np.asarray(executed_targets, dtype=np.float64)
    if core.ndim != 2 or executed.ndim != 2 or core.shape[0] != executed.shape[0]:
        raise ValueError("target reader requires paired core and target matrices")
    if executed.shape[1] != len(target_tests):
        raise ValueError("target reader semantic target registry changed")
    horizons = sorted({len(test.actions) for test in target_tests})
    observations = sorted({test.observation for test in target_tests})
    rows: list[np.ndarray] = []
    for test_index, test in enumerate(target_tests):
        probability = executed[:, test_index : test_index + 1]
        horizon = np.zeros((len(core), len(horizons)), dtype=np.float64)
        horizon[:, horizons.index(len(test.actions))] = 1.0
        observation = np.zeros((len(core), len(observations)), dtype=np.float64)
        observation[:, observations.index(test.observation)] = 1.0
        test_identity = np.zeros((len(core), len(target_tests)), dtype=np.float64)
        test_identity[:, test_index] = 1.0
        rows.append(
            np.concatenate(
                [core, probability, probability**2, horizon, observation, test_identity],
                axis=1,
            )
        )
    return np.stack(rows, axis=1)


@dataclass(frozen=True, slots=True)
class TargetLocalReader:
    """One target-model reader fitted only from oracle public cores."""

    linear_map: LinearMap
    n_tests: int
    n_bins: int

    def predict(
        self,
        cores: np.ndarray,
        executed_targets: np.ndarray,
        target_tests: tuple[PredictiveTest, ...],
    ) -> np.ndarray:
        design = target_reader_features(cores, executed_targets, target_tests)
        flat = design.reshape(-1, design.shape[-1])
        prediction = self.linear_map.predict(flat, clip=False)
        return project_simplex(prediction.reshape(len(cores), self.n_tests, self.n_bins))

    def metadata(self) -> dict[str, Any]:
        return {
            "n_tests": self.n_tests,
            "n_bins": self.n_bins,
            "linear_map": self.linear_map.metadata(),
            "fit_input": ("oracle_exact_public_core_frozen_public_executor_and_test_descriptor"),
            "source_evaluation_input": (
                "learned_public_core_frozen_public_executor_and_test_descriptor"
            ),
            "pair_specific_parameters": False,
        }

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        payload = self.linear_map.arrays(prefix)
        payload[f"{prefix}__shape"] = np.asarray([self.n_tests, self.n_bins], dtype=np.int64)
        return payload


def fit_target_local_reader(
    cores: np.ndarray,
    executed_targets: np.ndarray,
    signatures: np.ndarray,
    target_tests: tuple[PredictiveTest, ...],
    *,
    ridge: float,
) -> TargetLocalReader:
    behavior = np.asarray(signatures, dtype=np.float64)
    design = target_reader_features(cores, executed_targets, target_tests)
    if behavior.shape[:2] != design.shape[:2] or behavior.ndim != 3:
        raise ValueError("target-reader behavior signatures have the wrong shape")
    linear_map = fit_linear_map(
        design.reshape(-1, design.shape[-1]),
        behavior.reshape(-1, behavior.shape[-1]),
        ridge=ridge,
        method="ridge",
    )
    return TargetLocalReader(
        linear_map=linear_map,
        n_tests=behavior.shape[1],
        n_bins=behavior.shape[2],
    )


def select_target_reader(
    cores: np.ndarray,
    executed_targets: np.ndarray,
    signatures: np.ndarray,
    calibration_mask: np.ndarray,
    selection_mask: np.ndarray,
    target_tests: tuple[PredictiveTest, ...],
    ridge_grid: Iterable[float],
) -> tuple[TargetLocalReader, dict[str, Any]]:
    candidates: list[dict[str, float]] = []
    for ridge in ridge_grid:
        reader = fit_target_local_reader(
            cores[calibration_mask],
            executed_targets[calibration_mask],
            signatures[calibration_mask],
            target_tests,
            ridge=float(ridge),
        )
        prediction = reader.predict(
            cores[selection_mask],
            executed_targets[selection_mask],
            target_tests,
        )
        candidates.append(
            {
                "ridge": float(ridge),
                "selection_brier": brier_score(signatures[selection_mask], prediction),
            }
        )
    selected = min(candidates, key=lambda row: (row["selection_brier"], row["ridge"]))
    fit_mask = np.asarray(calibration_mask, dtype=bool) | np.asarray(selection_mask, dtype=bool)
    reader = fit_target_local_reader(
        cores[fit_mask],
        executed_targets[fit_mask],
        signatures[fit_mask],
        target_tests,
        ridge=selected["ridge"],
    )
    return reader, {
        "selected": selected,
        "candidates": candidates,
        "refit_roles": ["calibration", "selection"],
    }


def nearest_centroid_accuracy(
    train_packets: np.ndarray,
    train_labels: np.ndarray,
    validation_packets: np.ndarray,
    validation_labels: np.ndarray,
) -> float:
    train = np.asarray(train_packets, dtype=np.float64)
    validation = np.asarray(validation_packets, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    expected = np.asarray(validation_labels, dtype=np.int64)
    classes = np.unique(labels)
    centroids = np.stack([train[labels == value].mean(axis=0) for value in classes])
    scale = np.std(train, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    distances = np.mean(
        ((validation[:, None, :] - centroids[None, :, :]) / scale[None, None, :]) ** 2,
        axis=2,
    )
    predicted = classes[np.argmin(distances, axis=1)]
    return float(np.mean(predicted == expected))
