"""Train-only calibration and grouped metric helpers for RC0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frank_eq.evaluation.bootstrap import bootstrap_statistic


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    positive = array >= 0
    result = np.empty_like(array)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def logit(probabilities: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """Affine log-odds calibration fitted only on training worlds."""

    alpha: float
    beta: float
    steps: int
    converged: bool

    def predict_from_score(self, scores: np.ndarray) -> np.ndarray:
        return sigmoid(self.alpha * np.asarray(scores, dtype=np.float64) + self.beta)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.predict_from_score(scores)

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "steps": self.steps,
            "converged": self.converged,
        }


def _calibration_objective(
    scores: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    beta: float,
    l2: float,
) -> float:
    prediction = sigmoid(alpha * scores + beta)
    epsilon = 1e-12
    cross_entropy = -np.mean(
        targets * np.log(np.clip(prediction, epsilon, 1.0))
        + (1.0 - targets) * np.log(np.clip(1.0 - prediction, epsilon, 1.0))
    )
    return float(cross_entropy + 0.5 * l2 * alpha * alpha)


def fit_platt_calibrator(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    l2: float = 1e-3,
    max_steps: int = 100,
    tolerance: float = 1e-9,
) -> PlattCalibrator:
    """Fit a stable two-parameter logistic calibrator by damped Newton steps.

    Targets may be smoothed binary probabilities. The slope is deliberately
    unconstrained: a negative value identifies a stable answer-channel inversion
    that a model-local interface is allowed to correct.
    """

    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("calibration requires paired score/target vectors with at least two rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("calibration inputs must be finite")
    if np.any((y <= 0.0) | (y >= 1.0)):
        raise ValueError("calibration targets must lie strictly inside (0,1)")

    alpha = 1.0
    mean = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
    beta = float(np.log(mean) - np.log1p(-mean))
    previous = _calibration_objective(x, y, alpha, beta, l2)
    converged = False
    steps = 0

    for step_index in range(max_steps):
        steps = step_index + 1
        prediction = sigmoid(alpha * x + beta)
        error = prediction - y
        curvature = np.clip(prediction * (1.0 - prediction), 1e-8, None)
        gradient = np.asarray(
            [
                np.mean(error * x) + l2 * alpha,
                np.mean(error),
            ],
            dtype=np.float64,
        )
        hessian = np.asarray(
            [
                [np.mean(curvature * x * x) + l2, np.mean(curvature * x)],
                [np.mean(curvature * x), np.mean(curvature) + 1e-10],
            ],
            dtype=np.float64,
        )
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.pinv(hessian) @ gradient

        norm = float(np.linalg.norm(update))
        if norm > 25.0:
            update *= 25.0 / norm

        step_size = 1.0
        accepted = False
        candidate_alpha = alpha
        candidate_beta = beta
        candidate_objective = previous
        while step_size >= 1e-8:
            proposed_alpha = float(alpha - step_size * update[0])
            proposed_beta = float(beta - step_size * update[1])
            objective = _calibration_objective(
                x,
                y,
                proposed_alpha,
                proposed_beta,
                l2,
            )
            if np.isfinite(objective) and objective <= previous + 1e-14:
                candidate_alpha = proposed_alpha
                candidate_beta = proposed_beta
                candidate_objective = objective
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            break
        alpha = candidate_alpha
        beta = candidate_beta
        change = float(np.linalg.norm(step_size * update))
        improvement = previous - candidate_objective
        previous = candidate_objective
        if change <= tolerance or improvement <= tolerance:
            converged = True
            break

    return PlattCalibrator(
        alpha=float(alpha),
        beta=float(beta),
        steps=steps,
        converged=converged,
    )


def brier_score(targets: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        np.mean(
            (
                np.asarray(targets, dtype=np.float64)
                - np.asarray(probabilities, dtype=np.float64)
            )
            ** 2
        )
    )


def balanced_accuracy(targets: np.ndarray, probabilities: np.ndarray) -> float:
    truth = np.asarray(targets, dtype=np.float64) >= 0.5
    prediction = np.asarray(probabilities, dtype=np.float64) >= 0.5
    rates: list[float] = []
    for label in (False, True):
        selection = truth == label
        if np.any(selection):
            rates.append(float(np.mean(prediction[selection] == label)))
    return float(np.mean(rates)) if rates else 0.0


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    truth = np.asarray(targets, dtype=np.float64) >= 0.5
    prediction = np.asarray(probabilities, dtype=np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selection = (prediction >= lower) & (
            prediction <= upper if index == bins - 1 else prediction < upper
        )
        if not np.any(selection):
            continue
        confidence = float(prediction[selection].mean())
        accuracy = float(truth[selection].mean())
        value += float(selection.mean()) * abs(confidence - accuracy)
    return value


def aggregate_by_world(values: np.ndarray, world_ids: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    worlds = np.asarray(world_ids, dtype=np.int64)
    return np.asarray(
        [array[worlds == world].mean(axis=0) for world in np.unique(worlds)],
        dtype=np.float64,
    )


def interval(
    world_values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    return bootstrap_statistic(
        np.asarray(world_values, dtype=np.float64),
        replicates=replicates,
        seed=seed,
    ).to_dict()
