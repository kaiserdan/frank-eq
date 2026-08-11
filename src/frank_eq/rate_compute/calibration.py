"""Train-only score calibration and paired evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    positive = array >= 0
    result = np.empty_like(array)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_values = np.exp(array[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """One-dimensional affine log-odds calibration map."""

    alpha: float
    beta: float
    l2: float
    steps: int
    converged: bool

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return sigmoid(self.alpha * np.asarray(scores, dtype=np.float64) + self.beta)

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def fit_platt_calibrator(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    l2: float = 1e-3,
    max_steps: int = 100,
    tolerance: float = 1e-9,
) -> PlattCalibrator:
    """Fit soft-label logistic calibration with damped Newton updates.

    Targets may lie strictly inside ``[0, 1]`` because the graph oracle uses
    label smoothing. Only the slope is regularized; the intercept remains free.
    """

    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("calibration requires matching score/target arrays with at least two rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("calibration inputs must be finite")
    if np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError("calibration targets must lie in [0, 1]")
    if l2 < 0 or max_steps < 1:
        raise ValueError("invalid calibration optimization settings")

    design = np.column_stack([x, np.ones_like(x)])
    weights = np.asarray([1.0, float(np.log((y.mean() + 1e-4) / (1.0 - y.mean() + 1e-4)))])
    converged = False
    completed = 0
    for step in range(max_steps):
        logits = design @ weights
        probabilities = sigmoid(logits)
        gradient = design.T @ (probabilities - y)
        gradient[0] += l2 * weights[0]
        curvature = np.clip(probabilities * (1.0 - probabilities), 1e-6, None)
        hessian = design.T @ (design * curvature[:, None])
        hessian[0, 0] += l2
        hessian += np.eye(2) * 1e-8
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.pinv(hessian) @ gradient
        # A bounded step avoids numerical explosions for nearly separable scores.
        norm = float(np.linalg.norm(update))
        if norm > 10.0:
            update *= 10.0 / norm
        weights -= update
        completed = step + 1
        if float(np.linalg.norm(update)) <= tolerance:
            converged = True
            break
    return PlattCalibrator(
        alpha=float(weights[0]),
        beta=float(weights[1]),
        l2=float(l2),
        steps=completed,
        converged=converged,
    )


def brier_score(targets: np.ndarray, probabilities: np.ndarray) -> float:
    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(probabilities, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError("Brier inputs must have identical shapes")
    return float(np.mean((target - prediction) ** 2))


def balanced_accuracy(targets: np.ndarray, probabilities: np.ndarray) -> float:
    target = np.asarray(targets, dtype=np.float64).reshape(-1) >= 0.5
    predicted = np.asarray(probabilities, dtype=np.float64).reshape(-1) >= 0.5
    recalls: list[float] = []
    for label in (False, True):
        mask = target == label
        if np.any(mask):
            recalls.append(float(np.mean(predicted[mask] == label)))
    return float(np.mean(recalls)) if recalls else 0.0


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int,
) -> float:
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    prediction = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if bins < 2:
        raise ValueError("ECE requires at least two bins")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index + 1 == bins:
            selection = (prediction >= boundaries[index]) & (prediction <= boundaries[index + 1])
        else:
            selection = (prediction >= boundaries[index]) & (prediction < boundaries[index + 1])
        if not np.any(selection):
            continue
        confidence = float(np.mean(prediction[selection]))
        frequency = float(np.mean(target[selection]))
        error += float(np.mean(selection)) * abs(confidence - frequency)
    return float(error)


def aggregate_by_world(values: np.ndarray, world_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average all renderer/operation rows within the paired world unit."""

    array = np.asarray(values, dtype=np.float64)
    worlds = np.asarray(world_ids, dtype=np.int64).reshape(-1)
    if array.shape[0] != worlds.shape[0]:
        raise ValueError("world IDs and values have different row counts")
    unique = np.unique(worlds)
    return unique, np.asarray([array[worlds == world].mean(axis=0) for world in unique])
