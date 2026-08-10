"""Grouped nonparametric bootstrap utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    replicates: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "replicates": self.replicates,
        }


def bootstrap_statistic(
    values: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> Interval:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.shape[0] < 2:
        estimate = statistic(array)
        return Interval(estimate, estimate, estimate, replicates)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample_ids = rng.integers(0, array.shape[0], size=array.shape[0])
        estimates[index] = statistic(array[sample_ids])
    alpha = (1.0 - confidence) / 2.0
    return Interval(
        estimate=statistic(array),
        lower=float(np.quantile(estimates, alpha)),
        upper=float(np.quantile(estimates, 1.0 - alpha)),
        replicates=replicates,
    )
