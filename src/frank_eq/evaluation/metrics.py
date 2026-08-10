"""Metric implementations for operational equivalence Stage 0."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from frank_eq.models.baselines import apply_affine, fit_ridge, r2_score


def brier_score(target: np.ndarray, prediction: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    return np.mean((np.asarray(target) - np.asarray(prediction)) ** 2, axis=axis)


def binary_accuracy(target: np.ndarray, prediction: np.ndarray, axis: int | None = None) -> np.ndarray:
    return np.mean((np.asarray(target) >= 0.5) == (np.asarray(prediction) >= 0.5), axis=axis)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return numerator / np.clip(denominator, 1e-12, None)


def aggregate_views(
    values: np.ndarray,
    world_ids: np.ndarray,
    model_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (world, model) in enumerate(zip(world_ids, model_ids, strict=True)):
        groups[(int(world), int(model))].append(index)
    ordered = sorted(groups)
    aggregated = np.stack([values[groups[key]].mean(axis=0) for key in ordered])
    worlds = np.asarray([key[0] for key in ordered], dtype=np.int64)
    models = np.asarray([key[1] for key in ordered], dtype=np.int64)
    return aggregated, worlds, models


def renderer_invariance_cosine(
    code: np.ndarray,
    world_ids: np.ndarray,
    model_ids: np.ndarray,
) -> np.ndarray:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(world_ids, model_ids, strict=True)):
        groups[(int(key[0]), int(key[1]))].append(index)
    scores: list[float] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        anchor = code[indices[0]]
        for other in indices[1:]:
            scores.append(float(cosine_similarity(anchor[None], code[other][None])[0]))
    return np.asarray(scores, dtype=np.float64)


def cross_model_retrieval(
    code: np.ndarray,
    world_ids: np.ndarray,
    model_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aggregated, worlds, models = aggregate_views(code, world_ids, model_ids)
    unique_models = sorted(int(v) for v in np.unique(models))
    outcomes: list[float] = []
    margins: list[float] = []
    outcome_worlds: list[int] = []
    for source_model in unique_models:
        source_mask = models == source_model
        source_code = aggregated[source_mask]
        source_worlds = worlds[source_mask]
        source_code = source_code / np.clip(np.linalg.norm(source_code, axis=1, keepdims=True), 1e-12, None)
        for target_model in unique_models:
            if target_model == source_model:
                continue
            target_mask = models == target_model
            target_code = aggregated[target_mask]
            target_worlds = worlds[target_mask]
            target_code = target_code / np.clip(np.linalg.norm(target_code, axis=1, keepdims=True), 1e-12, None)
            similarities = source_code @ target_code.T
            for row_index, world in enumerate(source_worlds):
                matching = np.flatnonzero(target_worlds == world)
                if len(matching) != 1:
                    continue
                correct_index = int(matching[0])
                predicted_index = int(np.argmax(similarities[row_index]))
                outcomes.append(float(predicted_index == correct_index))
                outcome_worlds.append(int(world))
                wrong = np.delete(similarities[row_index], correct_index)
                hardest_wrong = float(np.max(wrong)) if wrong.size else -1.0
                margins.append(float(similarities[row_index, correct_index] - hardest_wrong))
    return (
        np.asarray(outcomes, dtype=np.float64),
        np.asarray(margins, dtype=np.float64),
        np.asarray(outcome_worlds, dtype=np.int64),
    )


def model_identity_probe(
    train_code: np.ndarray,
    train_model_ids: np.ndarray,
    test_code: np.ndarray,
    test_model_ids: np.ndarray,
    ridge: float = 1.0,
) -> float:
    unique = np.unique(np.concatenate([train_model_ids, test_model_ids]))
    lookup = {int(model_id): index for index, model_id in enumerate(unique)}
    train_one_hot = np.zeros((len(train_model_ids), len(unique)), dtype=np.float32)
    for index, model_id in enumerate(train_model_ids):
        train_one_hot[index, lookup[int(model_id)]] = 1.0
    weights, bias = fit_ridge(train_code, train_one_hot, ridge=ridge)
    scores = apply_affine(test_code, weights, bias)
    predictions = np.argmax(scores, axis=1)
    targets = np.asarray([lookup[int(model_id)] for model_id in test_model_ids])
    return float(np.mean(predictions == targets))


def pairwise_hidden_ridge_r2(
    train_hidden: np.ndarray,
    train_world_ids: np.ndarray,
    train_model_ids: np.ndarray,
    test_hidden: np.ndarray,
    test_world_ids: np.ndarray,
    test_model_ids: np.ndarray,
    ridge: float = 10.0,
) -> dict[str, float]:
    train_agg, train_worlds, train_models = aggregate_views(
        train_hidden.reshape(train_hidden.shape[0], -1),
        train_world_ids,
        train_model_ids,
    )
    test_agg, test_worlds, test_models = aggregate_views(
        test_hidden.reshape(test_hidden.shape[0], -1),
        test_world_ids,
        test_model_ids,
    )
    results: dict[str, float] = {}
    unique_models = sorted(int(v) for v in np.unique(train_models))
    for source_model in unique_models:
        for target_model in unique_models:
            if source_model == target_model:
                continue
            train_source_mask = train_models == source_model
            train_target_mask = train_models == target_model
            test_source_mask = test_models == source_model
            test_target_mask = test_models == target_model
            source_train = train_agg[train_source_mask]
            source_train_worlds = train_worlds[train_source_mask]
            target_train = train_agg[train_target_mask]
            target_train_worlds = train_worlds[train_target_mask]
            source_test = test_agg[test_source_mask]
            source_test_worlds = test_worlds[test_source_mask]
            target_test = test_agg[test_target_mask]
            target_test_worlds = test_worlds[test_target_mask]
            common_train = sorted(set(source_train_worlds) & set(target_train_worlds))
            common_test = sorted(set(source_test_worlds) & set(target_test_worlds))
            if not common_train or not common_test:
                continue
            x_train = np.stack([source_train[np.flatnonzero(source_train_worlds == w)[0]] for w in common_train])
            y_train = np.stack([target_train[np.flatnonzero(target_train_worlds == w)[0]] for w in common_train])
            x_test = np.stack([source_test[np.flatnonzero(source_test_worlds == w)[0]] for w in common_test])
            y_test = np.stack([target_test[np.flatnonzero(target_test_worlds == w)[0]] for w in common_test])
            # Widths differ. Ridge is fit from source to the observed target width;
            # padded dimensions remain part of the explicitly reported baseline.
            weights, bias = fit_ridge(x_train, y_train, ridge=ridge)
            results[f"{source_model}->{target_model}"] = r2_score(
                y_test,
                apply_affine(x_test, weights, bias),
            )
    return results
