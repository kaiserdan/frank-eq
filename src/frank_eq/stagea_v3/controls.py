"""Train-only priors, calibration, and direct-protocol selection for Stage-A v3."""

from __future__ import annotations

from typing import Any

import numpy as np

from frank_eq.rate_compute.calibration import (
    brier_score,
    fit_platt_calibrator,
    sigmoid,
)

from .capture import V3CaptureShard
from .config import StageAV3Config
from .panel import V3Panel

_CALIBRATION_L2 = 1e-3
_CALIBRATION_MAX_STEPS = 100


def _validate_fit_inputs(
    config: StageAV3Config,
    model_id: str,
    train_shards: dict[int, V3CaptureShard],
    panels: dict[int, V3Panel],
) -> None:
    expected = set(config.section("panel")["entity_counts"])
    if set(train_shards) != expected or set(panels) != expected:
        raise ValueError("train controls require every registered complexity")
    if model_id not in {model.model_id for model in config.models}:
        raise ValueError("train controls name an unregistered model")
    for entity_count in expected:
        shard = train_shards[entity_count]
        panel = panels[entity_count]
        shard.validate()
        panel.validate()
        if shard.role != "train" or panel.role != "train":
            raise ValueError("train controls may use training roles only")
        if shard.model_id != model_id:
            raise ValueError("train control shard belongs to another model")
        if panel.operation_registry_sha256 != shard.capture_summary.get(
            "operation_registry_sha256"
        ):
            raise ValueError("train control panel/capture operation registries differ")


def fit_train_controls(
    config: StageAV3Config,
    *,
    model_id: str,
    train_shards: dict[int, V3CaptureShard],
    panels: dict[int, V3Panel],
    capture_sha256: dict[str, str],
) -> dict[str, Any]:
    """Fit every permitted calibrator and selector using train worlds only."""

    _validate_fit_inputs(config, model_id, train_shards, panels)
    protocol_order = list(config.section("teacher_protocol")["direct_protocols"])
    smoothing = float(config.section("panel")["oracle_smoothing"])
    artifact: dict[str, Any] = {
        "schema": "frank_eq_stagea_v3_train_controls_v1",
        "config_sha256": config.config_sha256,
        "model_id": model_id,
        "fit_role": "train",
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "calibration": {
            "method": "platt_affine_log_odds",
            "l2": _CALIBRATION_L2,
            "max_steps": _CALIBRATION_MAX_STEPS,
            "negative_slope_allowed": True,
        },
        "capture_sha256": dict(sorted(capture_sha256.items())),
        "complexities": {},
    }
    for entity_count in sorted(train_shards):
        shard = train_shards[entity_count]
        panel = panels[entity_count]
        observed_protocols = shard.capture_summary.get("direct_protocol_order")
        if observed_protocols is not None and observed_protocols != protocol_order:
            raise ValueError("direct protocol order differs between config and capture")
        semantic_prior = shard.semantic_targets.mean(dim=0).numpy().astype(np.float64)
        behavioral_prior = shard.behavioral_targets.mean(dim=0).numpy().astype(np.float64)
        operation_prior = shard.operation_targets.mean(dim=0).numpy().astype(np.float64)

        interactive: list[dict[str, Any]] = []
        smoothed_edges = (
            shard.semantic_targets.numpy().astype(np.float64) * (1.0 - 2.0 * smoothing)
            + smoothing
        )
        behavioral_scores = shard.behavioral_log_odds.numpy().astype(np.float64)
        for coordinate in range(shard.coordinate_count):
            calibrator = fit_platt_calibrator(
                behavioral_scores[:, coordinate],
                smoothed_edges[:, coordinate],
                l2=_CALIBRATION_L2,
                max_steps=_CALIBRATION_MAX_STEPS,
            )
            interactive.append(calibrator.to_dict())

        direct_scores = shard.direct_log_odds.numpy().astype(np.float64)
        direct_targets = shard.operation_targets.numpy().astype(np.float64)
        direct_calibration: dict[str, dict[str, dict[str, Any]]] = {}
        direct_selection: dict[str, dict[str, Any]] = {}
        families = sorted({operation.definition.family for operation in panel.panel.operations})
        for family in families:
            operation_ids = [
                operation.definition.operation_id
                for operation in panel.panel.operations
                if operation.definition.family == family
            ]
            if not operation_ids:
                raise RuntimeError(f"training panel has no {family} operation instances")
            direct_calibration[family] = {}
            candidates: list[tuple[float, int, str]] = []
            for protocol_index, protocol in enumerate(protocol_order):
                scores = direct_scores[:, operation_ids, protocol_index].reshape(-1)
                targets = direct_targets[:, operation_ids].reshape(-1)
                calibrator = fit_platt_calibrator(
                    scores,
                    targets,
                    l2=_CALIBRATION_L2,
                    max_steps=_CALIBRATION_MAX_STEPS,
                )
                predictions = calibrator.predict(scores)
                train_brier = brier_score(targets, predictions)
                direct_calibration[family][protocol] = {
                    "calibrator": calibrator.to_dict(),
                    "train_brier": train_brier,
                    "rows": int(scores.size),
                }
                candidates.append((train_brier, protocol_index, protocol))
            train_brier, protocol_index, selected_protocol = min(
                candidates, key=lambda row: (row[0], row[1])
            )
            direct_selection[family] = {
                "protocol": selected_protocol,
                "protocol_index": protocol_index,
                "train_brier": train_brier,
                "all_train_briers": {
                    protocol: score for score, _, protocol in candidates
                },
            }

        artifact["complexities"][str(entity_count)] = {
            "operation_registry_sha256": panel.operation_registry_sha256,
            "semantic_edge_prior": semantic_prior.tolist(),
            "behavioral_edge_prior": behavioral_prior.tolist(),
            "operation_prior": operation_prior.tolist(),
            "interactive_basis_calibration": interactive,
            "direct_protocol_order": protocol_order,
            "direct_calibration": direct_calibration,
            "direct_selection": direct_selection,
        }
    return artifact


def _calibrator_predict(payload: dict[str, Any], scores: np.ndarray) -> np.ndarray:
    alpha = float(payload["alpha"])
    beta = float(payload["beta"])
    return sigmoid(alpha * np.asarray(scores, dtype=np.float64) + beta)


def apply_train_controls(
    config: StageAV3Config,
    artifact: dict[str, Any],
    shard: V3CaptureShard,
    panel: V3Panel,
) -> dict[str, np.ndarray | list[str]]:
    """Apply frozen train-only priors/calibrators/selectors to any later role."""

    if artifact.get("schema") != "frank_eq_stagea_v3_train_controls_v1":
        raise ValueError("unsupported Stage-A v3 train-control schema")
    if artifact.get("config_sha256") != config.config_sha256:
        raise ValueError("train controls belong to a different frozen config")
    if artifact.get("model_id") != shard.model_id:
        raise ValueError("train controls and capture shard model IDs differ")
    if artifact.get("fit_role") != "train" or artifact.get("test_rows_used") != 0:
        raise ValueError("train controls have an invalid fit/access scope")
    shard.validate()
    panel.validate()
    if shard.entity_count != panel.entity_count or shard.role != panel.role:
        raise ValueError("train-control panel and capture shard roles differ")
    row = artifact["complexities"][str(shard.entity_count)]
    if row["operation_registry_sha256"] != panel.operation_registry_sha256:
        raise ValueError("train controls and evaluation operation registries differ")

    semantic_prior = np.asarray(row["semantic_edge_prior"], dtype=np.float64)
    behavioral_prior = np.asarray(row["behavioral_edge_prior"], dtype=np.float64)
    operation_prior = np.asarray(row["operation_prior"], dtype=np.float64)
    if semantic_prior.shape != (shard.coordinate_count,):
        raise ValueError("semantic edge prior has the wrong coordinate count")
    interactive = np.empty((shard.rows, shard.coordinate_count), dtype=np.float64)
    scores = shard.behavioral_log_odds.numpy().astype(np.float64)
    calibration_rows = row["interactive_basis_calibration"]
    if len(calibration_rows) != shard.coordinate_count:
        raise ValueError("interactive basis calibration is incomplete")
    for coordinate, calibrator in enumerate(calibration_rows):
        interactive[:, coordinate] = _calibrator_predict(
            calibrator, scores[:, coordinate]
        )

    protocol_order = list(row["direct_protocol_order"])
    if protocol_order != config.section("teacher_protocol")["direct_protocols"]:
        raise ValueError("direct protocol registry changed after training")
    direct_probability = np.empty_like(shard.operation_targets.numpy(), dtype=np.float64)
    direct_generated_tokens = np.empty_like(
        shard.operation_targets.numpy(), dtype=np.int64
    )
    direct_protocols: list[str] = [""] * shard.operation_targets.shape[1]
    direct_scores = shard.direct_log_odds.numpy().astype(np.float64)
    generated = shard.direct_generated_tokens.numpy().astype(np.int64)
    for operation in panel.panel.operations:
        operation_id = operation.definition.operation_id
        family = operation.definition.family
        selection = row["direct_selection"][family]
        protocol = str(selection["protocol"])
        protocol_index = int(selection["protocol_index"])
        if protocol_order[protocol_index] != protocol:
            raise ValueError("direct protocol index/name selection differs")
        calibrator = row["direct_calibration"][family][protocol]["calibrator"]
        direct_probability[:, operation_id] = _calibrator_predict(
            calibrator,
            direct_scores[:, operation_id, protocol_index],
        )
        direct_generated_tokens[:, operation_id] = generated[
            :, operation_id, protocol_index
        ]
        direct_protocols[operation_id] = protocol
    return {
        "semantic_edge_prior": np.broadcast_to(
            semantic_prior, (shard.rows, shard.coordinate_count)
        ).copy(),
        "behavioral_edge_prior": np.broadcast_to(
            behavioral_prior, (shard.rows, shard.coordinate_count)
        ).copy(),
        "operation_prior": np.broadcast_to(
            operation_prior, tuple(shard.operation_targets.shape)
        ).copy(),
        "interactive_basis": interactive,
        "direct_probability": direct_probability,
        "direct_generated_tokens": direct_generated_tokens,
        "direct_protocols": direct_protocols,
    }
