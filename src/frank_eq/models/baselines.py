"""Classical and architectural baselines for Stage-0 evaluation."""

from __future__ import annotations

import numpy as np


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    ridge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an affine ridge map using a numerically stable augmented system."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge inputs must be aligned two-dimensional arrays")
    means_x = x.mean(axis=0, keepdims=True)
    means_y = y.mean(axis=0, keepdims=True)
    centered_x = x - means_x
    centered_y = y - means_y
    gram = centered_x.T @ centered_x
    gram.flat[:: gram.shape[0] + 1] += ridge
    weights = np.linalg.solve(gram, centered_x.T @ centered_y)
    bias = (means_y - means_x @ weights).reshape(-1)
    return weights.astype(np.float32), bias.astype(np.float32)


def apply_affine(features: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float32) @ weights + bias


def r2_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    residual = np.sum((targets - predictions) ** 2)
    total = np.sum((targets - targets.mean(axis=0, keepdims=True)) ** 2)
    if total <= 1e-12:
        return float("nan")
    return float(1.0 - residual / total)
