"""Train/validation-only localization for failed real Stage-A runs.

This module deliberately does not produce a promotion decision. It asks which
part of the failed pipeline is limiting:

1. Are formal facts linearly readable from the frozen capture?
2. Is the model's own future branch signature readable from the capture?
3. Is the model natively competent on the registered operation panel?
4. Does readability survive a renderer swap?

All probe fits use training worlds and all reported probe outcomes use
validation worlds. Test worlds are never selected by this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from frank_eq.data.real import RealBundle
from frank_eq.utils import atomic_write_json

_EPS = 1e-6


def _clip_probability(value: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=np.float64), _EPS, 1.0 - _EPS)


def _brier(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.mean((target - prediction) ** 2))


def _binary_cross_entropy(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    probability = _clip_probability(prediction)
    return float(-np.mean(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability)))


def _accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.mean((np.asarray(target) >= 0.5) == (np.asarray(prediction) >= 0.5))
    )


def _balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float | None:
    truth = np.asarray(target) >= 0.5
    guessed = np.asarray(prediction) >= 0.5
    if truth.ndim == 1:
        truth = truth[:, None]
        guessed = guessed[:, None]
    scores: list[float] = []
    for column in range(truth.shape[1]):
        positive = truth[:, column]
        negative = ~positive
        if not positive.any() or not negative.any():
            continue
        true_positive = float(np.mean(guessed[positive, column]))
        true_negative = float(np.mean(~guessed[negative, column]))
        scores.append(0.5 * (true_positive + true_negative))
    return None if not scores else float(np.mean(scores))


def _r2(target: np.ndarray, prediction: np.ndarray) -> float | None:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = float(np.sum((target - prediction) ** 2))
    total = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    return None if total <= 1e-12 else float(1.0 - residual / total)


def _standardize(
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    validation = np.asarray(validation, dtype=np.float64)
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (train - mean) / scale, (validation - mean) / scale


def _dual_ridge_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    """Fit affine ridge in sample space, avoiding a hidden-width square solve."""

    x_train, x_validation = _standardize(train_features, validation_features)
    y_train = np.asarray(train_targets, dtype=np.float64)
    target_mean = y_train.mean(axis=0, keepdims=True)
    centered_target = y_train - target_mean
    kernel = x_train @ x_train.T
    kernel.flat[:: kernel.shape[0] + 1] += float(ridge)
    dual = np.linalg.solve(kernel, centered_target)
    return (x_validation @ x_train.T @ dual + target_mean).astype(np.float64)


def _probability_report(
    train_target: np.ndarray,
    validation_target: np.ndarray,
    validation_prediction: np.ndarray,
) -> dict[str, Any]:
    train_target = np.asarray(train_target, dtype=np.float64)
    validation_target = np.asarray(validation_target, dtype=np.float64)
    prediction = _clip_probability(validation_prediction)
    coordinate_prior = _clip_probability(train_target.mean(axis=0, keepdims=True))
    prior_prediction = np.repeat(coordinate_prior, len(validation_target), axis=0)
    global_prior = float(np.mean(train_target))
    global_prediction = np.full_like(validation_target, global_prior, dtype=np.float64)
    return {
        "brier": _brier(validation_target, prediction),
        "bce": _binary_cross_entropy(validation_target, prediction),
        "accuracy": _accuracy(validation_target, prediction),
        "balanced_accuracy": _balanced_accuracy(validation_target, prediction),
        "coordinate_prior_brier": _brier(validation_target, prior_prediction),
        "coordinate_prior_bce": _binary_cross_entropy(validation_target, prior_prediction),
        "coordinate_prior_accuracy": _accuracy(validation_target, prior_prediction),
        "global_prior_brier": _brier(validation_target, global_prediction),
        "global_prior_accuracy": _accuracy(validation_target, global_prediction),
        "brier_gain_over_coordinate_prior": (
            _brier(validation_target, prior_prediction)
            - _brier(validation_target, prediction)
        ),
    }


def _continuous_report(
    train_target: np.ndarray,
    validation_target: np.ndarray,
    validation_prediction: np.ndarray,
) -> dict[str, Any]:
    train_target = np.asarray(train_target, dtype=np.float64)
    validation_target = np.asarray(validation_target, dtype=np.float64)
    prediction = np.asarray(validation_prediction, dtype=np.float64)
    prior = np.repeat(train_target.mean(axis=0, keepdims=True), len(validation_target), axis=0)
    return {
        "mse": _brier(validation_target, prediction),
        "prior_mse": _brier(validation_target, prior),
        "mse_gain_over_prior": _brier(validation_target, prior) - _brier(validation_target, prediction),
        "r2": _r2(validation_target, prediction),
    }


def _model_indices(
    bundle: RealBundle,
    *,
    model_id: int,
    world_ids: tuple[int, ...],
    renderer_id: int | None = None,
) -> np.ndarray:
    mask = (bundle.model_ids == model_id) & np.isin(
        bundle.world_ids, np.asarray(world_ids, dtype=np.int64)
    )
    if renderer_id is not None:
        mask &= bundle.renderer_ids == renderer_id
    return np.flatnonzero(mask)


def _features(
    bundle: RealBundle,
    indices: np.ndarray,
    *,
    model_id: int,
    layer: int | None,
) -> np.ndarray:
    width = int(bundle.model_hidden_dims[model_id])
    hidden = np.asarray(bundle.hidden[indices, :, :width], dtype=np.float32)
    return hidden.reshape(len(indices), -1) if layer is None else hidden[:, layer, :]


def _fit_target(
    bundle: RealBundle,
    *,
    model_id: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    target_name: str,
    ridge: float,
) -> dict[str, Any]:
    target = {
        "facts": bundle.facts,
        "oracle_signature": bundle.signatures,
        "self_signature": bundle.model_signatures,
        "residual": bundle.residual,
    }[target_name]
    feature_reports: dict[str, Any] = {}
    for layer in range(bundle.n_layers):
        name = f"layer_{layer}"
        prediction = _dual_ridge_predict(
            _features(bundle, train_indices, model_id=model_id, layer=layer),
            target[train_indices],
            _features(bundle, validation_indices, model_id=model_id, layer=layer),
            ridge=ridge,
        )
        feature_reports[name] = (
            _continuous_report(target[train_indices], target[validation_indices], prediction)
            if target_name == "residual"
            else _probability_report(target[train_indices], target[validation_indices], prediction)
        )
    prediction = _dual_ridge_predict(
        _features(bundle, train_indices, model_id=model_id, layer=None),
        target[train_indices],
        _features(bundle, validation_indices, model_id=model_id, layer=None),
        ridge=ridge,
    )
    feature_reports["concatenated"] = (
        _continuous_report(target[train_indices], target[validation_indices], prediction)
        if target_name == "residual"
        else _probability_report(target[train_indices], target[validation_indices], prediction)
    )
    score_name = "mse" if target_name == "residual" else "brier"
    best_name = min(feature_reports, key=lambda name: feature_reports[name][score_name])
    return {
        "target": target_name,
        "ridge": ridge,
        "features": feature_reports,
        "best_feature": best_name,
        "best": feature_reports[best_name],
    }


def _renderer_transfer(
    bundle: RealBundle,
    *,
    model_id: int,
    ridge: float,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for train_renderer, validation_renderer in ((0, 1), (1, 0)):
        train_indices = _model_indices(
            bundle,
            model_id=model_id,
            world_ids=bundle.split.train_world_ids,
            renderer_id=train_renderer,
        )
        validation_indices = _model_indices(
            bundle,
            model_id=model_id,
            world_ids=bundle.split.validation_world_ids,
            renderer_id=validation_renderer,
        )
        prediction = _dual_ridge_predict(
            _features(bundle, train_indices, model_id=model_id, layer=None),
            bundle.facts[train_indices],
            _features(bundle, validation_indices, model_id=model_id, layer=None),
            ridge=ridge,
        )
        reports[f"{train_renderer}_to_{validation_renderer}"] = _probability_report(
            bundle.facts[train_indices],
            bundle.facts[validation_indices],
            prediction,
        )
    return reports


def _native_competence(
    bundle: RealBundle,
    *,
    model_id: int,
) -> dict[str, Any]:
    train_indices = _model_indices(
        bundle, model_id=model_id, world_ids=bundle.split.train_world_ids
    )
    validation_indices = _model_indices(
        bundle, model_id=model_id, world_ids=bundle.split.validation_world_ids
    )
    overall = _probability_report(
        bundle.signatures[train_indices],
        bundle.signatures[validation_indices],
        bundle.model_signatures[validation_indices],
    )
    by_family: dict[str, Any] = {}
    for family in sorted({operation.family for operation in bundle.operations}):
        operation_ids = np.asarray(
            [operation.operation_id for operation in bundle.operations if operation.family == family],
            dtype=np.int64,
        )
        by_family[family] = _probability_report(
            bundle.signatures[train_indices][:, operation_ids],
            bundle.signatures[validation_indices][:, operation_ids],
            bundle.model_signatures[validation_indices][:, operation_ids],
        )
    return {"overall": overall, "by_family": by_family}


def _recommendation(models: list[dict[str, Any]], founder_ids: tuple[int, ...]) -> dict[str, Any]:
    founders = [models[index] for index in founder_ids]
    native_gain = float(
        np.mean(
            [
                model["native_competence"]["overall"]["brier_gain_over_coordinate_prior"]
                for model in founders
            ]
        )
    )
    self_gain = float(
        min(
            model["readability"]["self_signature"]["best"][
                "brier_gain_over_coordinate_prior"
            ]
            for model in founders
        )
    )
    fact_gain = float(
        min(
            model["readability"]["facts"]["best"]["brier_gain_over_coordinate_prior"]
            for model in founders
        )
    )
    fact_balanced = [
        model["readability"]["facts"]["best"]["balanced_accuracy"] for model in founders
    ]
    minimum_balanced = float(
        min(value for value in fact_balanced if value is not None)
    ) if any(value is not None for value in fact_balanced) else None

    if native_gain <= 0.0:
        code = "FIX_NATIVE_COMPETENCE_BEFORE_LATENT_REVISION"
        reason = (
            "On validation worlds the frozen models' own branches do not beat an "
            "operation-wise training prior. Prompt/runtime competence is therefore "
            "a prerequisite before attributing failure to hidden-state capture."
        )
    elif self_gain <= 0.0:
        code = "EXPAND_CAUSAL_STATE_CAPTURE"
        reason = (
            "The current last-token residual capture does not linearly predict the "
            "model's own future branch signature. Test token-sequence or selected-KV "
            "state summaries before changing the public quotient."
        )
    elif fact_gain <= 0.0 or (minimum_balanced is not None and minimum_balanced < 0.55):
        code = "SEPARATE_OPERATIONAL_STATE_FROM_SEMANTIC_GROUNDING"
        reason = (
            "Own-future state is readable but external graph facts are not. Preserve "
            "a behavioral operational code and treat fact grounding as a separate "
            "model-local calibration problem."
        )
    else:
        code = "REVISE_COMPILER_GAUGE_AND_OBJECTIVE"
        reason = (
            "Raw captures support the required targets, so the failed joint quotient "
            "is more consistent with shared-head gauge or multi-objective optimization "
            "than with absent source information."
        )
    return {
        "code": code,
        "reason": reason,
        "founder_native_brier_gain_over_prior_mean": native_gain,
        "founder_self_signature_brier_gain_min": self_gain,
        "founder_fact_brier_gain_min": fact_gain,
        "founder_fact_balanced_accuracy_min": minimum_balanced,
        "promotional": False,
    }


def diagnose_real_cache(
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    ridge: float = 10.0,
) -> dict[str, Any]:
    """Run a non-promotional diagnostic on training and validation worlds only."""

    if ridge <= 0:
        raise ValueError("ridge must be positive")
    bundle = RealBundle.load(cache_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models: list[dict[str, Any]] = []
    for model_id, model_name in enumerate(bundle.model_names):
        train_indices = _model_indices(
            bundle, model_id=model_id, world_ids=bundle.split.train_world_ids
        )
        validation_indices = _model_indices(
            bundle, model_id=model_id, world_ids=bundle.split.validation_world_ids
        )
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise RuntimeError(f"model {model_name} has incomplete train/validation rows")
        models.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "role": (
                    "held" if bundle.split.held_model_id == model_id else "founder"
                ),
                "train_views": int(len(train_indices)),
                "validation_views": int(len(validation_indices)),
                "readability": {
                    target: _fit_target(
                        bundle,
                        model_id=model_id,
                        train_indices=train_indices,
                        validation_indices=validation_indices,
                        target_name=target,
                        ridge=ridge,
                    )
                    for target in (
                        "facts",
                        "oracle_signature",
                        "self_signature",
                        "residual",
                    )
                },
                "renderer_transfer_fact_readability": _renderer_transfer(
                    bundle, model_id=model_id, ridge=ridge
                ),
                "native_competence": _native_competence(bundle, model_id=model_id),
            }
        )

    report = {
        "schema": "frank_eq_stagea_localization_v1",
        "scope": "post-outcome train-validation-only localization",
        "cache_dir": str(Path(cache_dir)),
        "ridge": ridge,
        "data_usage": {
            "train_worlds": len(bundle.split.train_world_ids),
            "validation_worlds": len(bundle.split.validation_world_ids),
            "test_worlds_used": 0,
            "test_labels_consumed": False,
            "heldout_operation_gate_reused": False,
        },
        "models": models,
        "recommendation": _recommendation(models, bundle.split.founder_model_ids),
        "authorizes_test_access": False,
        "authorizes_new_outcome_run": False,
        "authorizes_receiver_execution": False,
        "authorizes_scientific_claim": False,
    }
    atomic_write_json(output / "localization.json", report)
    return report
