"""Founder training and held-sender onboarding for Stage 0."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from frank_eq.config import RunConfig
from frank_eq.data.synthetic import ObservationDataset, SyntheticBundle, WorldBatchSampler
from frank_eq.models import OperationalQuotientModel
from frank_eq.telemetry import WandbTelemetry
from frank_eq.utils import atomic_write_json, parameter_count, resolve_device, seed_everything

from .objectives import LossBreakdown, compute_objective


class Stage0Trainer:
    """Train independent model charts around one shared operational quotient."""

    def __init__(
        self,
        config: RunConfig,
        bundle: SyntheticBundle,
        output_dir: str | Path,
        telemetry: WandbTelemetry | None = None,
    ):
        self.config = config
        self.telemetry = telemetry
        torch.set_num_threads(config.training.num_threads)
        self.bundle = bundle
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(config.training.seed)
        self.device = resolve_device(config.training.device)
        self.operation_descriptors = torch.from_numpy(bundle.operation_descriptors).float().to(
            self.device
        )
        self.train_operation_ids = torch.tensor(
            bundle.split.train_operation_ids,
            dtype=torch.long,
            device=self.device,
        )
        self.model = OperationalQuotientModel(
            model_hidden_dims=bundle.model_hidden_dims,
            n_layers=bundle.n_layers,
            n_facts=bundle.facts.shape[1],
            n_residual=bundle.residual.shape[1],
            operation_descriptor_dim=bundle.operation_descriptors.shape[1],
            config=config.model,
        ).to(self.device)
        self.history_path = self.output_dir / "training_history.jsonl"

    def _loader(
        self,
        *,
        world_ids: Iterable[int],
        model_ids: Iterable[int],
        shuffle: bool,
        epoch: int,
    ) -> tuple[ObservationDataset, DataLoader[dict[str, torch.Tensor]], WorldBatchSampler]:
        indices = self.bundle.indices_for(
            world_ids=tuple(world_ids),
            model_ids=tuple(model_ids),
        )
        dataset = ObservationDataset(self.bundle, indices)
        sampler = WorldBatchSampler(
            dataset,
            self.config.training.worlds_per_batch,
            shuffle=shuffle,
            seed=self.config.training.seed,
        )
        sampler.set_epoch(epoch)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        return dataset, loader, sampler

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    @staticmethod
    def _average_breakdowns(items: list[dict[str, float]]) -> dict[str, float]:
        if not items:
            return {}
        keys = items[0].keys()
        return {key: float(np.mean([item[key] for item in items])) for key in keys}

    def _run_epoch(
        self,
        loader: DataLoader[dict[str, torch.Tensor]],
        *,
        optimizer: torch.optim.Optimizer | None,
        onboarding: bool,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)
        breakdowns: list[dict[str, float]] = []
        for raw_batch in loader:
            batch = self._move_batch(raw_batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                output = self.model(
                    batch["hidden"],
                    batch["model_id"],
                    self.operation_descriptors,
                )
                losses: LossBreakdown = compute_objective(
                    model=self.model,
                    output=output,
                    batch=batch,
                    train_operation_ids=self.train_operation_ids,
                    loss_config=self.config.losses,
                    model_config=self.config.model,
                    onboarding=onboarding,
                )
                if training:
                    losses.total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.training.gradient_clip,
                    )
                    optimizer.step()
            breakdowns.append(losses.detached())
        return self._average_breakdowns(breakdowns)

    def _append_history(self, payload: dict[str, object]) -> None:
        with self.history_path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _fit_phase(
        self,
        *,
        phase: str,
        train_worlds: tuple[int, ...],
        validation_worlds: tuple[int, ...],
        model_ids: tuple[int, ...],
        epochs: int,
        learning_rate: float,
        onboarding: bool,
    ) -> dict[str, object]:
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError(f"phase {phase} has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_validation = float("inf")
        best_epoch = -1
        stale_epochs = 0
        start = time.time()
        epochs_completed = 0

        for epoch in range(epochs):
            epochs_completed = epoch + 1
            _, train_loader, _ = self._loader(
                world_ids=train_worlds,
                model_ids=model_ids,
                shuffle=True,
                epoch=epoch,
            )
            _, validation_loader, _ = self._loader(
                world_ids=validation_worlds,
                model_ids=model_ids,
                shuffle=False,
                epoch=0,
            )
            train_metrics = self._run_epoch(
                train_loader,
                optimizer=optimizer,
                onboarding=onboarding,
            )
            validation_metrics = self._run_epoch(
                validation_loader,
                optimizer=None,
                onboarding=onboarding,
            )
            validation_total = (
                validation_metrics["signature"]
                + self.config.losses.facts * validation_metrics["facts"]
                + self.config.losses.residual * validation_metrics["residual"]
                + self.config.losses.renderer_invariance * validation_metrics["renderer_invariance"]
                + self.config.losses.cross_model_invariance * validation_metrics["cross_model_invariance"]
                + self.config.losses.world_contrastive * validation_metrics["world_contrastive"]
            )
            self._append_history(
                {
                    "phase": phase,
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
            if self.telemetry is not None:
                self.telemetry.log(
                    {
                        f"train/{phase}": {**train_metrics, "validation_total": validation_total},
                        f"validation/{phase}": validation_metrics,
                    },
                    step=epoch,
                )
            if validation_total < best_validation - 1e-6:
                best_validation = validation_total
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.config.training.patience:
                break

        if best_state is None:
            raise RuntimeError(f"phase {phase} did not produce a checkpoint")
        self.model.load_state_dict(best_state)
        summary = {
            "phase": phase,
            "best_epoch": best_epoch,
            "best_validation_total": best_validation,
            "epochs_observed": epochs_completed,
            "elapsed_seconds": time.time() - start,
        }
        if self.telemetry is not None:
            self.telemetry.log({f"phase/{phase}": summary})
        return summary

    def train(self) -> dict[str, object]:
        """Train founder charts, then onboard the held sender without decoder updates."""

        if self.history_path.exists():
            self.history_path.unlink()
        founder_ids = self.bundle.split.founder_model_ids
        founder_summary = self._fit_phase(
            phase="founders",
            train_worlds=self.bundle.split.train_world_ids,
            validation_worlds=self.bundle.split.validation_world_ids,
            model_ids=founder_ids,
            epochs=self.config.training.epochs,
            learning_rate=self.config.training.learning_rate,
            onboarding=False,
        )
        founder_trainable_parameter_count = parameter_count(self.model, trainable_only=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config.as_dict(),
                "phase": "founders",
            },
            self.output_dir / "founders.pt",
        )

        held_summary: dict[str, object] | None = None
        held_model_id = self.bundle.split.held_model_id
        if held_model_id is not None:
            self.model.freeze_except_chart(held_model_id)
            held_summary = self._fit_phase(
                phase="held_sender_onboarding",
                train_worlds=self.bundle.split.train_world_ids,
                validation_worlds=self.bundle.split.validation_world_ids,
                model_ids=(held_model_id,),
                epochs=self.config.training.onboarding_epochs,
                learning_rate=self.config.training.onboarding_learning_rate,
                onboarding=True,
            )

        final_checkpoint = {
            "state_dict": self.model.state_dict(),
            "config": self.config.as_dict(),
            "phase": "final",
            "model_hidden_dims": self.bundle.model_hidden_dims,
            "n_layers": self.bundle.n_layers,
            "n_facts": int(self.bundle.facts.shape[1]),
            "n_residual": int(self.bundle.residual.shape[1]),
            "operation_descriptor_dim": int(self.bundle.operation_descriptors.shape[1]),
        }
        torch.save(final_checkpoint, self.output_dir / "final.pt")

        summary = {
            "schema": "frank_eq_training_summary_v1",
            "device": str(self.device),
            "parameter_count": parameter_count(self.model),
            "founder_trainable_parameter_count": founder_trainable_parameter_count,
            "founders": founder_summary,
            "held_sender": held_summary,
            "workspace_active_fraction": {
                model_id: self.model.charts[str(model_id)].workspace_gate.active_fraction()
                for model_id in range(len(self.bundle.model_hidden_dims))
            },
            "checkpoint": str(self.output_dir / "final.pt"),
        }
        atomic_write_json(self.output_dir / "training_summary.json", summary)
        if self.telemetry is not None:
            self.telemetry.log(
                {
                    "training": {
                        key: value
                        for key, value in summary.items()
                        if key not in {"schema", "checkpoint"}
                    }
                }
            )
        return summary
