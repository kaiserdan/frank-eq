"""Deterministic probability/logit quantization."""

from __future__ import annotations

import numpy as np


def probabilities_to_logits(probabilities: np.ndarray, limit: float = 8.0) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(values) - np.log1p(-values)
    return np.clip(logits, -limit, limit).astype(np.float32)


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def quantize_logits(logits: np.ndarray, bits: int, limit: float = 8.0) -> np.ndarray:
    if not 1 <= bits <= 16:
        raise ValueError("bits must be between 1 and 16")
    levels = 2**bits - 1
    values = np.clip(np.asarray(logits, dtype=np.float64), -limit, limit)
    normalized = (values + limit) / (2.0 * limit)
    return np.rint(normalized * levels).astype(np.int32)


def dequantize_logits(values: np.ndarray, bits: int, limit: float = 8.0) -> np.ndarray:
    if not 1 <= bits <= 16:
        raise ValueError("bits must be between 1 and 16")
    levels = 2**bits - 1
    quantized = np.asarray(values, dtype=np.float64)
    return (quantized / levels * (2.0 * limit) - limit).astype(np.float32)
