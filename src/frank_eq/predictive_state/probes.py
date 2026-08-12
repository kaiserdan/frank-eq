"""Train-only predictive-state probes, controls, and grouped uncertainty."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RidgeProbe:
    weights: np.ndarray
    intercept: np.ndarray
    ridge: float
    feature_mean: np.ndarray
    target_mean: np.ndarray

    def predict(self, features: np.ndarray, *, clip: bool = True) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.weights.shape[0]:
            raise ValueError("probe features have the wrong shape")
        prediction = (values - self.feature_mean) @ self.weights + self.target_mean
        prediction += self.intercept
        return np.clip(prediction, 1e-7, 1.0 - 1e-7) if clip else prediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "ridge": self.ridge,
            "input_dim": int(self.weights.shape[0]),
            "output_dim": int(self.weights.shape[1]),
            "intercept_norm": float(np.linalg.norm(self.intercept)),
            "weight_norm": float(np.linalg.norm(self.weights)),
        }


def fit_ridge_probe(features: np.ndarray, targets: np.ndarray, *, ridge: float) -> RidgeProbe:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("ridge probe requires paired rank-two arrays with at least two rows")
    if ridge <= 0:
        raise ValueError("ridge penalty must be positive")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("ridge probe inputs must be finite")
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    if xc.shape[1] <= xc.shape[0]:
        gram = xc.T @ xc
        gram.flat[:: gram.shape[0] + 1] += ridge
        right = xc.T @ yc
        try:
            weights = np.linalg.solve(gram, right)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(gram) @ right
    else:
        # High-dimensional LLM residuals have far more coordinates than
        # development histories.  Solve the mathematically equivalent dual
        # system to avoid allocating a hidden_width x hidden_width matrix.
        kernel = xc @ xc.T
        kernel.flat[:: kernel.shape[0] + 1] += ridge
        try:
            dual = np.linalg.solve(kernel, yc)
        except np.linalg.LinAlgError:
            dual = np.linalg.pinv(kernel) @ yc
        weights = xc.T @ dual
    return RidgeProbe(
        weights=weights,
        intercept=np.zeros(y.shape[1], dtype=np.float64),
        ridge=float(ridge),
        feature_mean=x_mean[0],
        target_mean=y_mean[0],
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
    if truth.shape != estimate.shape or truth.ndim != 2:
        raise ValueError("row Brier arrays must be matching matrices")
    return np.mean((truth - estimate) ** 2, axis=1)


def aggregate_by_history(values: np.ndarray, history_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    identities = np.asarray(history_ids, dtype=np.int64).reshape(-1)
    if array.shape[0] != identities.shape[0]:
        raise ValueError("values and history IDs have different row counts")
    unique = np.unique(identities)
    return unique, np.asarray([array[identities == value].mean(axis=0) for value in unique])


def bootstrap_interval(
    history_values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(history_values, dtype=np.float64).reshape(-1)
    if values.size < 2 or replicates < 1:
        raise ValueError("bootstrap interval requires at least two values and one replicate")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    samples = values[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "point": float(values.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "replicates": int(replicates),
        "units": int(values.size),
    }


def paired_brier_gain_interval(
    targets: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    history_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    candidate_loss = row_brier(targets, candidate)
    baseline_loss = row_brier(targets, baseline)
    _, gains = aggregate_by_history(baseline_loss - candidate_loss, history_ids)
    return bootstrap_interval(gains, replicates=replicates, seed=seed)


def deterministic_token_hash_features(
    token_ids: np.ndarray,
    attention_mask: np.ndarray,
    *,
    width: int,
    position_period: int,
) -> np.ndarray:
    """Parameter-free order-sensitive token control with the activation width."""

    tokens = np.asarray(token_ids, dtype=np.int64)
    mask = np.asarray(attention_mask, dtype=bool)
    if tokens.ndim != 2 or mask.shape != tokens.shape:
        raise ValueError("token hash expects token and mask matrices with equal shape")
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


def split_fit_selection(
    history_ids: np.ndarray,
    *,
    selection_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    identities = np.asarray(history_ids, dtype=np.int64).reshape(-1)
    unique = np.unique(identities)
    if unique.size < 10:
        raise ValueError("probe selection requires at least ten distinct histories")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    selection_count = max(2, int(round(selection_fraction * unique.size)))
    selection_set = set(int(value) for value in shuffled[:selection_count])
    selection = np.asarray([int(value) in selection_set for value in identities], dtype=bool)
    fit = ~selection
    if not np.any(fit) or not np.any(selection):
        raise RuntimeError("fit/selection split is empty")
    return fit, selection


def choose_ridge_and_layer(
    features_by_layer: np.ndarray,
    targets: np.ndarray,
    history_ids: np.ndarray,
    *,
    ridge_grid: list[float],
    selection_fraction: float,
    selection_seed: int,
) -> dict[str, Any]:
    """Choose one contextual layer and ridge on a train-only history split."""

    features = np.asarray(features_by_layer, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if features.ndim != 3 or truth.ndim != 2 or features.shape[0] != truth.shape[0]:
        raise ValueError("layer selection requires [rows,layers,width] and [rows,targets]")
    fit_mask, selection_mask = split_fit_selection(
        history_ids,
        selection_fraction=selection_fraction,
        seed=selection_seed,
    )
    candidates: list[dict[str, Any]] = []
    for layer in range(features.shape[1]):
        for ridge in ridge_grid:
            probe = fit_ridge_probe(features[fit_mask, layer], truth[fit_mask], ridge=float(ridge))
            prediction = probe.predict(features[selection_mask, layer])
            score = brier_score(truth[selection_mask], prediction)
            candidates.append({"layer": layer, "ridge": float(ridge), "selection_brier": score})
    selected = min(candidates, key=lambda row: (row["selection_brier"], row["layer"], row["ridge"]))
    final_probe = fit_ridge_probe(
        features[:, int(selected["layer"])],
        truth,
        ridge=float(selected["ridge"]),
    )
    return {
        "layer": int(selected["layer"]),
        "ridge": float(selected["ridge"]),
        "selection_brier": float(selected["selection_brier"]),
        "candidates": candidates,
        "probe": final_probe,
    }


def quantize_probabilities(values: np.ndarray, bits: int) -> np.ndarray:
    if bits < 1 or bits > 16:
        raise ValueError("probability quantization bits must lie in [1,16]")
    levels = (1 << bits) - 1
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.round(clipped * levels) / levels


def wrong_history_margin_interval(
    targets: np.ndarray,
    predictions: np.ndarray,
    history_ids: np.ndarray,
    lengths: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Compare each prediction with its correct history and a length-matched wrong one."""

    truth = np.asarray(targets, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    identities = np.asarray(history_ids, dtype=np.int64).reshape(-1)
    history_lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if truth.shape != estimate.shape or truth.shape[0] != identities.size:
        raise ValueError("wrong-history margin inputs have inconsistent row counts")
    if history_lengths.size != identities.size:
        raise ValueError("wrong-history lengths have the wrong row count")

    unique, grouped_truth = aggregate_by_history(truth, identities)
    _, grouped_prediction = aggregate_by_history(estimate, identities)
    grouped_lengths = np.asarray(
        [int(np.unique(history_lengths[identities == value])[0]) for value in unique],
        dtype=np.int64,
    )
    wrong_truth = np.empty_like(grouped_truth)
    for length in np.unique(grouped_lengths):
        positions = np.flatnonzero(grouped_lengths == length)
        if positions.size < 2:
            raise ValueError("wrong-history specificity needs two histories per length")
        wrong_truth[positions] = grouped_truth[np.roll(positions, 1)]
    correct_loss = np.mean((grouped_truth - grouped_prediction) ** 2, axis=1)
    wrong_loss = np.mean((wrong_truth - grouped_prediction) ** 2, axis=1)
    return bootstrap_interval(
        wrong_loss - correct_loss,
        replicates=replicates,
        seed=seed,
    )
