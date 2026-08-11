"""World-balanced fitting for Stage-A v3 compilers and activation controls."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from frank_eq.utils import parameter_count, resolve_device, seed_everything

from .baselines import FinalTokenPublicMLP, HistoricalContinuousQuotient, TokenIDResampler
from .capture import V3CaptureShard
from .compiler import TokenSlotCompiler
from .config import StageAV3Config

_BASIS_KINDS = {"activation", "token_id", "final_token"}
_CHANNELS = {"semantic", "behavioral"}


def _compiler_kwargs(config: StageAV3Config, input_width: int) -> dict[str, int | float]:
    compiler = config.section("compiler")
    capture = config.section("capture")
    return {
        "input_width": input_width,
        "n_depths": len(capture["normalized_depths"]),
        "max_entities": max(config.section("panel")["entity_counts"]),
        "max_tokens": int(capture["compiler_max_tokens"]),
        "model_dim": int(compiler["model_dim"]),
        "attention_heads": int(compiler["attention_heads"]),
        "attention_blocks": int(compiler["attention_blocks"]),
        "feedforward_dim": int(compiler["feedforward_dim"]),
        "dropout": float(compiler["dropout"]),
    }


def make_basis_predictor(
    config: StageAV3Config,
    *,
    kind: str,
    input_width: int,
    target_parameter_count: int | None = None,
) -> nn.Module:
    if kind not in _BASIS_KINDS:
        raise ValueError(f"unsupported Stage-A v3 basis predictor: {kind}")
    kwargs = _compiler_kwargs(config, input_width)
    if kind == "activation":
        return TokenSlotCompiler(**kwargs)
    if kind == "token_id":
        return TokenIDResampler(**kwargs)
    if target_parameter_count is None:
        reference = TokenSlotCompiler(**kwargs)
        target_parameter_count = parameter_count(reference)
    return FinalTokenPublicMLP(
        input_width=input_width,
        n_depths=int(kwargs["n_depths"]),
        target_parameter_count=target_parameter_count,
        max_entities=int(kwargs["max_entities"]),
        dropout=float(kwargs["dropout"]),
    )


def _validate_shard_pair(
    train_shards: dict[int, V3CaptureShard],
    validation_shards: dict[int, V3CaptureShard],
    config: StageAV3Config,
) -> tuple[str, int, int]:
    expected = set(config.section("panel")["entity_counts"])
    if set(train_shards) != expected or set(validation_shards) != expected:
        raise ValueError("basis fitting requires every registered entity count")
    all_shards = [*train_shards.values(), *validation_shards.values()]
    if any(shard.role != "train" for shard in train_shards.values()):
        raise ValueError("basis fitting train shards have a non-train role")
    if any(shard.role != "validation" for shard in validation_shards.values()):
        raise ValueError("basis fitting validation shards have a non-validation role")
    model_ids = {shard.model_id for shard in all_shards}
    hidden_widths = {shard.hidden_width for shard in all_shards}
    depth_counts = {len(shard.layer_indices) for shard in all_shards}
    if len(model_ids) != 1 or len(hidden_widths) != 1 or len(depth_counts) != 1:
        raise ValueError("basis fitting shards disagree on model or residual dimensions")
    for shard in all_shards:
        shard.validate()
        grouped: dict[int, set[int]] = {}
        for world_id, renderer_id in zip(
            shard.world_ids.tolist(), shard.renderer_ids.tolist(), strict=True
        ):
            grouped.setdefault(int(world_id), set()).add(int(renderer_id))
        expected_renderers = {0, 1}
        if any(renderers != expected_renderers for renderers in grouped.values()):
            raise ValueError("fit shards must pair both frozen renderer views within every world")
    return next(iter(model_ids)), next(iter(hidden_widths)), next(iter(depth_counts))


def _world_batches(
    shard: V3CaptureShard,
    *,
    worlds_per_batch: int,
    seed: int,
    shuffle: bool,
) -> Iterator[torch.Tensor]:
    worlds = np.asarray(sorted(set(int(value) for value in shard.world_ids.tolist())))
    if shuffle:
        worlds = np.random.default_rng(seed).permutation(worlds)
    for offset in range(0, len(worlds), worlds_per_batch):
        selected = torch.tensor(worlds[offset : offset + worlds_per_batch], dtype=torch.long)
        mask = torch.isin(shard.world_ids, selected)
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() < 1:
            raise RuntimeError("world-balanced batch unexpectedly contains no rows")
        yield indices


def _epoch_batches(
    shards: dict[int, V3CaptureShard],
    *,
    worlds_per_batch: int,
    seed: int,
    shuffle: bool,
) -> list[tuple[int, torch.Tensor]]:
    batches = [
        (entity_count, indices)
        for entity_count, shard in sorted(shards.items())
        for indices in _world_batches(
            shard,
            worlds_per_batch=worlds_per_batch,
            seed=seed + entity_count,
            shuffle=shuffle,
        )
    ]
    if shuffle:
        order = np.random.default_rng(seed + 99_991).permutation(len(batches))
        batches = [batches[int(index)] for index in order]
    return batches


def _renderer_variance(probabilities: torch.Tensor, world_ids: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for world_id in torch.unique(world_ids):
        group = probabilities[world_ids == world_id]
        if group.shape[0] != 2:
            raise RuntimeError("renderer consistency requires exactly two fit views per world")
        losses.append(((group - group.mean(dim=0, keepdim=True)) ** 2).mean())
    if not losses:
        return probabilities.new_tensor(0.0)
    return torch.stack(losses).mean()


def _predict_logits(
    model: nn.Module,
    kind: str,
    shard: V3CaptureShard,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention_mask = shard.attention_mask[indices].to(device)
    world_ids = shard.world_ids[indices].to(device)
    if kind == "token_id":
        if not isinstance(model, TokenIDResampler):
            raise TypeError("token-ID predictor has the wrong module type")
        logits = model(
            shard.token_ids[indices].to(device),
            attention_mask,
            entity_count=shard.entity_count,
        )
    else:
        residuals = shard.residuals[indices].to(device)
        logits = model(  # type: ignore[call-arg]
            residuals,
            attention_mask,
            entity_count=shard.entity_count,
        )
    return logits, attention_mask, world_ids


def _targets(
    shard: V3CaptureShard,
    indices: torch.Tensor,
    channel: str,
    device: torch.device,
) -> torch.Tensor:
    source = shard.semantic_targets if channel == "semantic" else shard.behavioral_targets
    return source[indices].to(device)


def _basis_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    world_ids: torch.Tensor,
    *,
    channel: str,
    consistency_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    probabilities = torch.sigmoid(logits)
    if channel == "semantic":
        prediction_loss = F.binary_cross_entropy_with_logits(logits, targets)
    else:
        prediction_loss = F.mse_loss(probabilities, targets)
    renderer_variance = _renderer_variance(probabilities, world_ids)
    total = prediction_loss + consistency_weight * renderer_variance
    return total, {
        "prediction_loss": float(prediction_loss.detach().cpu()),
        "renderer_variance": float(renderer_variance.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def _validation_metric(
    model: nn.Module,
    kind: str,
    channel: str,
    shards: dict[int, V3CaptureShard],
    *,
    worlds_per_batch: int,
    device: torch.device,
    consistency_weight: float,
) -> dict[str, float]:
    model.eval()
    squared_errors: list[torch.Tensor] = []
    variances: list[float] = []
    with torch.inference_mode():
        for entity_count, indices in _epoch_batches(
            shards,
            worlds_per_batch=worlds_per_batch,
            seed=0,
            shuffle=False,
        ):
            shard = shards[entity_count]
            logits, _, world_ids = _predict_logits(
                model, kind, shard, indices, device=device
            )
            probabilities = torch.sigmoid(logits)
            targets = _targets(shard, indices, channel, device)
            squared_errors.append(((probabilities - targets) ** 2).detach().cpu().reshape(-1))
            variances.append(float(_renderer_variance(probabilities, world_ids).cpu()))
    brier = float(torch.cat(squared_errors).mean())
    renderer_variance = float(np.mean(variances))
    return {
        "brier": brier,
        "renderer_variance": renderer_variance,
        "selection_metric": brier + consistency_weight * renderer_variance,
    }


def _initialization_seed(registered_seed: int, channel: str, kind: str) -> int:
    channel_offset = {"semantic": 1, "behavioral": 2}[channel]
    kind_offset = {"activation": 0, "token_id": 10, "final_token": 20}[kind]
    return registered_seed * 100 + kind_offset + channel_offset


def train_basis_predictor(
    config: StageAV3Config,
    *,
    train_shards: dict[int, V3CaptureShard],
    validation_shards: dict[int, V3CaptureShard],
    kind: str,
    channel: str,
    registered_seed: int,
    checkpoint_path: str | Path,
    onboarding: bool,
    capture_sha256: dict[str, str],
    device_name: str = "auto",
) -> dict[str, Any]:
    """Fit one independent seed/channel module and freeze its best validation state."""

    if kind not in _BASIS_KINDS or channel not in _CHANNELS:
        raise ValueError("unsupported basis predictor kind or channel")
    if kind != "activation" and channel != "semantic":
        raise ValueError("activation controls are registered only for the semantic channel")
    registered_seeds = config.section("compiler")["seeds"]
    if registered_seed not in registered_seeds:
        raise ValueError("compiler seed is not part of the frozen ensemble")
    model_id, input_width, depth_count = _validate_shard_pair(
        train_shards, validation_shards, config
    )
    model_role = next(model.role for model in config.models if model.model_id == model_id)
    if onboarding != (model_role == "held"):
        raise ValueError("onboarding flag disagrees with the frozen model role")

    initialization_seed = _initialization_seed(registered_seed, channel, kind)
    seed_everything(initialization_seed)
    if kind == "activation":
        model = make_basis_predictor(config, kind=kind, input_width=input_width)
        primary_parameters = parameter_count(model)
    else:
        reference = make_basis_predictor(config, kind="activation", input_width=input_width)
        primary_parameters = parameter_count(reference)
        del reference
        seed_everything(initialization_seed)
        model = make_basis_predictor(
            config,
            kind=kind,
            input_width=input_width,
            target_parameter_count=primary_parameters,
        )
    observed_parameters = parameter_count(model)
    tolerance = float(config.section("baselines")["learned_parameter_tolerance_fraction"])
    if kind in {"token_id", "final_token"}:
        relative_error = abs(observed_parameters - primary_parameters) / primary_parameters
        if relative_error > tolerance:
            raise RuntimeError("activation-control parameter count exceeds frozen tolerance")

    training = config.section("training")
    compiler = config.section("compiler")
    epochs = int(training["onboarding_epochs"] if onboarding else training["epochs"])
    learning_rate = float(
        training["onboarding_learning_rate"] if onboarding else training["learning_rate"]
    )
    patience = int(training["patience"])
    worlds_per_batch = int(training["worlds_per_batch"])
    consistency_weight = float(compiler["renderer_consistency_weight"])
    device = resolve_device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training["weight_decay"]),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_metric = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(epochs):
        model.train()
        batch_metrics: list[dict[str, float]] = []
        for entity_count, indices in _epoch_batches(
            train_shards,
            worlds_per_batch=worlds_per_batch,
            seed=initialization_seed + epoch * 10_007,
            shuffle=True,
        ):
            shard = train_shards[entity_count]
            optimizer.zero_grad(set_to_none=True)
            logits, _, world_ids = _predict_logits(
                model, kind, shard, indices, device=device
            )
            target = _targets(shard, indices, channel, device)
            loss, metrics = _basis_loss(
                logits,
                target,
                world_ids,
                channel=channel,
                consistency_weight=consistency_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            batch_metrics.append(metrics)

        validation = _validation_metric(
            model,
            kind,
            channel,
            validation_shards,
            worlds_per_batch=worlds_per_batch,
            device=device,
            consistency_weight=consistency_weight,
        )
        epoch_row = {
            "epoch": epoch,
            "train": {
                key: float(np.mean([row[key] for row in batch_metrics]))
                for key in ("prediction_loss", "renderer_variance", "total")
            },
            "validation": validation,
        }
        history.append(epoch_row)
        if validation["selection_metric"] < best_metric - 1e-8:
            best_metric = validation["selection_metric"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("basis fitting did not produce a checkpoint")
    model.load_state_dict(best_state)
    final_validation = _validation_metric(
        model,
        kind,
        channel,
        validation_shards,
        worlds_per_batch=worlds_per_batch,
        device=device,
        consistency_weight=consistency_weight,
    )
    metadata: dict[str, Any] = {
        "schema": "frank_eq_stagea_v3_basis_checkpoint_v1",
        "config_sha256": config.config_sha256,
        "model_id": model_id,
        "model_role": model_role,
        "kind": kind,
        "channel": channel,
        "registered_seed": registered_seed,
        "initialization_seed": initialization_seed,
        "input_width": input_width,
        "depth_count": depth_count,
        "primary_parameter_count": primary_parameters,
        "parameter_count": observed_parameters,
        "parameter_relative_error": (
            abs(observed_parameters - primary_parameters) / primary_parameters
        ),
        "best_epoch": best_epoch,
        "epochs_observed": len(history),
        "best_selection_metric": best_metric,
        "final_validation": final_validation,
        "onboarding": onboarding,
        "capture_sha256": dict(sorted(capture_sha256.items())),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save({"metadata": metadata, "state_dict": best_state}, temporary)
    os.replace(temporary, target)
    return metadata


def load_basis_predictor(
    config: StageAV3Config,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    device_name: str = "auto",
) -> tuple[nn.Module, dict[str, Any]]:
    source = Path(checkpoint_path)
    if expected_sha256 is not None:
        from frank_eq.utils import sha256_file

        if sha256_file(source) != expected_sha256:
            raise ValueError("basis checkpoint hash mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    metadata = dict(payload["metadata"])
    if metadata.get("schema") != "frank_eq_stagea_v3_basis_checkpoint_v1":
        raise ValueError("unsupported basis checkpoint schema")
    if metadata.get("config_sha256") != config.config_sha256:
        raise ValueError("basis checkpoint belongs to a different frozen config")
    model = make_basis_predictor(
        config,
        kind=str(metadata["kind"]),
        input_width=int(metadata["input_width"]),
        target_parameter_count=int(metadata["primary_parameter_count"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(resolve_device(device_name))
    model.eval()
    return model, metadata


def predict_basis_logits(
    model: nn.Module,
    metadata: dict[str, Any],
    shard: V3CaptureShard,
    *,
    worlds_per_batch: int,
    device_name: str = "auto",
) -> torch.Tensor:
    if shard.model_id != metadata["model_id"]:
        raise ValueError("basis checkpoint and capture shard model IDs differ")
    device = resolve_device(device_name)
    model.to(device)
    output = torch.empty((shard.rows, shard.coordinate_count), dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        for indices in _world_batches(
            shard,
            worlds_per_batch=worlds_per_batch,
            seed=0,
            shuffle=False,
        ):
            logits, _, _ = _predict_logits(
                model,
                str(metadata["kind"]),
                shard,
                indices,
                device=device,
            )
            output[indices] = logits.detach().float().cpu()
    return output


def predict_basis_ensemble(
    config: StageAV3Config,
    checkpoint_paths: list[str | Path],
    shard: V3CaptureShard,
    *,
    checkpoint_sha256: dict[str, str] | None = None,
    device_name: str = "auto",
) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, Any]]]:
    registered = config.section("compiler")["seeds"]
    if len(checkpoint_paths) != len(registered):
        raise ValueError("basis ensemble must include every frozen compiler seed")
    logits: list[torch.Tensor] = []
    metadata_rows: list[dict[str, Any]] = []
    worlds_per_batch = int(config.section("training")["worlds_per_batch"])
    for path in checkpoint_paths:
        source = Path(path)
        expected = None
        if checkpoint_sha256 is not None:
            expected = checkpoint_sha256.get(str(source)) or checkpoint_sha256.get(source.name)
        model, metadata = load_basis_predictor(
            config,
            source,
            expected_sha256=expected,
            device_name=device_name,
        )
        logits.append(
            predict_basis_logits(
                model,
                metadata,
                shard,
                worlds_per_batch=worlds_per_batch,
                device_name=device_name,
            )
        )
        metadata_rows.append(metadata)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    seeds = [int(row["registered_seed"]) for row in metadata_rows]
    if seeds != registered or len(set(seeds)) != len(seeds):
        raise ValueError("basis ensemble checkpoint order or seeds differ from registration")
    kinds = {str(row["kind"]) for row in metadata_rows}
    channels = {str(row["channel"]) for row in metadata_rows}
    models = {str(row["model_id"]) for row in metadata_rows}
    if len(kinds) != 1 or len(channels) != 1 or models != {shard.model_id}:
        raise ValueError("basis ensemble mixes model, kind, or channel namespaces")
    mean_logit = torch.stack(logits).mean(dim=0)
    return torch.sigmoid(mean_logit), logits, metadata_rows


def _continuous_validation_metric(
    model: HistoricalContinuousQuotient,
    shards: dict[int, V3CaptureShard],
    descriptors: dict[int, torch.Tensor],
    *,
    worlds_per_batch: int,
    device: torch.device,
    consistency_weight: float,
) -> dict[str, float]:
    model.eval()
    squared_errors: list[torch.Tensor] = []
    variances: list[float] = []
    with torch.inference_mode():
        for entity_count, indices in _epoch_batches(
            shards,
            worlds_per_batch=worlds_per_batch,
            seed=0,
            shuffle=False,
        ):
            shard = shards[entity_count]
            logits, _ = model(
                shard.residuals[indices].to(device),
                shard.attention_mask[indices].to(device),
                descriptors[entity_count].to(device),
            )
            probabilities = torch.sigmoid(logits)
            targets = shard.operation_targets[indices].to(device)
            squared_errors.append(((probabilities - targets) ** 2).cpu().reshape(-1))
            variances.append(
                float(
                    _renderer_variance(
                        probabilities,
                        shard.world_ids[indices].to(device),
                    ).cpu()
                )
            )
    brier = float(torch.cat(squared_errors).mean())
    renderer_variance = float(np.mean(variances))
    return {
        "brier": brier,
        "renderer_variance": renderer_variance,
        "selection_metric": brier + consistency_weight * renderer_variance,
    }


def _validate_operation_descriptors(
    descriptors: dict[int, torch.Tensor],
    shards: dict[int, V3CaptureShard],
) -> None:
    if set(descriptors) != set(shards):
        raise ValueError("continuous baseline descriptors omit a registered complexity")
    for entity_count, descriptor in descriptors.items():
        operations = shards[entity_count].operation_targets.shape[1]
        if descriptor.shape != (operations, 35) or descriptor.dtype != torch.float32:
            raise ValueError("continuous baseline operation descriptors have the wrong shape")


def train_continuous_quotient(
    config: StageAV3Config,
    *,
    train_shards: dict[int, V3CaptureShard],
    validation_shards: dict[int, V3CaptureShard],
    operation_descriptors: dict[int, torch.Tensor],
    registered_seed: int,
    checkpoint_path: str | Path,
    onboarding: bool,
    capture_sha256: dict[str, str],
    descriptor_sha256: dict[str, str],
    device_name: str = "auto",
) -> dict[str, Any]:
    """Fit the historical final-token continuous quotient and learned operation head."""

    if registered_seed not in config.section("compiler")["seeds"]:
        raise ValueError("continuous baseline seed is not part of the frozen ensemble")
    model_id, input_width, depth_count = _validate_shard_pair(
        train_shards, validation_shards, config
    )
    _validate_operation_descriptors(operation_descriptors, train_shards)
    model_role = next(model.role for model in config.models if model.model_id == model_id)
    if onboarding != (model_role == "held"):
        raise ValueError("continuous onboarding flag disagrees with model role")
    initialization_seed = registered_seed * 100 + 31
    seed_everything(initialization_seed)
    model = HistoricalContinuousQuotient(
        input_width=input_width,
        n_depths=depth_count,
    )
    training = config.section("training")
    consistency_weight = float(config.section("compiler")["renderer_consistency_weight"])
    epochs = int(training["onboarding_epochs"] if onboarding else training["epochs"])
    learning_rate = float(
        training["onboarding_learning_rate"] if onboarding else training["learning_rate"]
    )
    patience = int(training["patience"])
    worlds_per_batch = int(training["worlds_per_batch"])
    device = resolve_device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training["weight_decay"]),
    )
    descriptors = {
        entity_count: descriptor.to(device)
        for entity_count, descriptor in operation_descriptors.items()
    }

    best_state: dict[str, torch.Tensor] | None = None
    best_metric = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(epochs):
        model.train()
        train_rows: list[dict[str, float]] = []
        for entity_count, indices in _epoch_batches(
            train_shards,
            worlds_per_batch=worlds_per_batch,
            seed=initialization_seed + epoch * 10_007,
            shuffle=True,
        ):
            shard = train_shards[entity_count]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(
                shard.residuals[indices].to(device),
                shard.attention_mask[indices].to(device),
                descriptors[entity_count],
            )
            targets = shard.operation_targets[indices].to(device)
            prediction_loss = F.binary_cross_entropy_with_logits(logits, targets)
            renderer_variance = _renderer_variance(
                torch.sigmoid(logits), shard.world_ids[indices].to(device)
            )
            loss = prediction_loss + consistency_weight * renderer_variance
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            train_rows.append(
                {
                    "prediction_loss": float(prediction_loss.detach().cpu()),
                    "renderer_variance": float(renderer_variance.detach().cpu()),
                    "total": float(loss.detach().cpu()),
                }
            )
        validation = _continuous_validation_metric(
            model,
            validation_shards,
            descriptors,
            worlds_per_batch=worlds_per_batch,
            device=device,
            consistency_weight=consistency_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "train": {
                    key: float(np.mean([row[key] for row in train_rows]))
                    for key in ("prediction_loss", "renderer_variance", "total")
                },
                "validation": validation,
            }
        )
        if validation["selection_metric"] < best_metric - 1e-8:
            best_metric = validation["selection_metric"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("continuous baseline fitting did not produce a checkpoint")
    model.load_state_dict(best_state)
    final_validation = _continuous_validation_metric(
        model,
        validation_shards,
        descriptors,
        worlds_per_batch=worlds_per_batch,
        device=device,
        consistency_weight=consistency_weight,
    )
    metadata: dict[str, Any] = {
        "schema": "frank_eq_stagea_v3_continuous_checkpoint_v1",
        "config_sha256": config.config_sha256,
        "model_id": model_id,
        "model_role": model_role,
        "kind": "historical_continuous_quotient",
        "channel": "semantic",
        "registered_seed": registered_seed,
        "initialization_seed": initialization_seed,
        "input_width": input_width,
        "depth_count": depth_count,
        "parameter_count": parameter_count(model),
        "best_epoch": best_epoch,
        "epochs_observed": len(history),
        "best_selection_metric": best_metric,
        "final_validation": final_validation,
        "onboarding": onboarding,
        "capture_sha256": dict(sorted(capture_sha256.items())),
        "descriptor_sha256": dict(sorted(descriptor_sha256.items())),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save({"metadata": metadata, "state_dict": best_state}, temporary)
    os.replace(temporary, target)
    return metadata


def load_continuous_quotient(
    config: StageAV3Config,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    device_name: str = "auto",
) -> tuple[HistoricalContinuousQuotient, dict[str, Any]]:
    source = Path(checkpoint_path)
    if expected_sha256 is not None:
        from frank_eq.utils import sha256_file

        if sha256_file(source) != expected_sha256:
            raise ValueError("continuous checkpoint hash mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    metadata = dict(payload["metadata"])
    if metadata.get("schema") != "frank_eq_stagea_v3_continuous_checkpoint_v1":
        raise ValueError("unsupported continuous checkpoint schema")
    if metadata.get("config_sha256") != config.config_sha256:
        raise ValueError("continuous checkpoint belongs to another frozen config")
    model = HistoricalContinuousQuotient(
        input_width=int(metadata["input_width"]),
        n_depths=int(metadata["depth_count"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(resolve_device(device_name))
    model.eval()
    return model, metadata


def predict_continuous_logits(
    model: HistoricalContinuousQuotient,
    metadata: dict[str, Any],
    shard: V3CaptureShard,
    operation_descriptors: torch.Tensor,
    *,
    worlds_per_batch: int,
    device_name: str = "auto",
) -> torch.Tensor:
    if shard.model_id != metadata["model_id"]:
        raise ValueError("continuous checkpoint and capture shard model IDs differ")
    device = resolve_device(device_name)
    model.to(device)
    descriptors = operation_descriptors.to(device)
    output = torch.empty_like(shard.operation_targets, dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        for indices in _world_batches(
            shard,
            worlds_per_batch=worlds_per_batch,
            seed=0,
            shuffle=False,
        ):
            logits, _ = model(
                shard.residuals[indices].to(device),
                shard.attention_mask[indices].to(device),
                descriptors,
            )
            output[indices] = logits.detach().float().cpu()
    return output


def predict_continuous_ensemble(
    config: StageAV3Config,
    checkpoint_paths: list[str | Path],
    shard: V3CaptureShard,
    operation_descriptors: torch.Tensor,
    *,
    checkpoint_sha256: dict[str, str] | None = None,
    device_name: str = "auto",
) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, Any]]]:
    registered = config.section("compiler")["seeds"]
    if len(checkpoint_paths) != len(registered):
        raise ValueError("continuous ensemble must include every frozen seed")
    logits: list[torch.Tensor] = []
    metadata_rows: list[dict[str, Any]] = []
    worlds_per_batch = int(config.section("training")["worlds_per_batch"])
    for path in checkpoint_paths:
        source = Path(path)
        expected = None
        if checkpoint_sha256 is not None:
            expected = checkpoint_sha256.get(str(source)) or checkpoint_sha256.get(source.name)
        model, metadata = load_continuous_quotient(
            config,
            source,
            expected_sha256=expected,
            device_name=device_name,
        )
        logits.append(
            predict_continuous_logits(
                model,
                metadata,
                shard,
                operation_descriptors,
                worlds_per_batch=worlds_per_batch,
                device_name=device_name,
            )
        )
        metadata_rows.append(metadata)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    seeds = [int(row["registered_seed"]) for row in metadata_rows]
    if seeds != registered or {row["model_id"] for row in metadata_rows} != {shard.model_id}:
        raise ValueError("continuous ensemble checkpoint order or model differs")
    mean_logit = torch.stack(logits).mean(dim=0)
    return torch.sigmoid(mean_logit), logits, metadata_rows
