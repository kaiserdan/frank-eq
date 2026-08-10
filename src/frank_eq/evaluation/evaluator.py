"""End-to-end evaluator and machine decision writer."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from frank_eq.config import RunConfig
from frank_eq.data.synthetic import (
    OPERATION_FAMILIES,
    ObservationDataset,
    SyntheticBundle,
    facts_only_signatures,
)
from frank_eq.models import OperationalQuotientModel
from frank_eq.packet import OperationalPacketV1, QueryConditionedSelector
from frank_eq.utils import atomic_write_json, resolve_device, sha256_file

from .bootstrap import bootstrap_statistic
from .gates import reduce_stage0
from .metrics import (
    binary_accuracy,
    brier_score,
    cross_model_retrieval,
    model_identity_probe,
    pairwise_hidden_ridge_r2,
    renderer_invariance_cosine,
)


class Stage0Evaluator:
    """Evaluate future-signature fidelity, invariance, and establishment."""

    @staticmethod
    def _world_means(values: np.ndarray, world_ids: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        worlds = np.asarray(world_ids, dtype=np.int64)
        return np.asarray(
            [values[worlds == world].mean(axis=0) for world in np.unique(worlds)],
            dtype=np.float64,
        )

    def __init__(
        self,
        config: RunConfig,
        bundle: SyntheticBundle,
        checkpoint_path: str | Path,
        output_dir: str | Path,
    ):
        self.config = config
        torch.set_num_threads(config.training.num_threads)
        self.bundle = bundle
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(config.training.device)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model = OperationalQuotientModel(
            model_hidden_dims=bundle.model_hidden_dims,
            n_layers=bundle.n_layers,
            n_facts=bundle.facts.shape[1],
            n_residual=bundle.residual.shape[1],
            operation_descriptor_dim=bundle.operation_descriptors.shape[1],
            config=config.model,
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.descriptors = torch.from_numpy(bundle.operation_descriptors).float().to(self.device)

    def _collect(
        self,
        world_ids: Iterable[int],
        model_ids: Iterable[int],
    ) -> dict[str, np.ndarray]:
        indices = self.bundle.indices_for(
            world_ids=tuple(world_ids),
            model_ids=tuple(model_ids),
        )
        dataset = ObservationDataset(self.bundle, indices)
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        collected: dict[str, list[np.ndarray]] = {
            "row_index": [],
            "world_id": [],
            "model_id": [],
            "renderer_id": [],
            "code": [],
            "signature": [],
            "fact": [],
            "residual": [],
            "truth_signature": [],
            "truth_fact": [],
            "truth_residual": [],
            "hidden": [],
        }
        with torch.no_grad():
            for batch in loader:
                hidden = batch["hidden"].to(self.device)
                model_id = batch["model_id"].to(self.device)
                output = self.model(hidden, model_id, self.descriptors)
                collected["row_index"].append(batch["row_index"].numpy())
                collected["world_id"].append(batch["world_id"].numpy())
                collected["model_id"].append(batch["model_id"].numpy())
                collected["renderer_id"].append(batch["renderer_id"].numpy())
                collected["code"].append(output.code.cpu().numpy())
                collected["signature"].append(torch.sigmoid(output.signature_logits).cpu().numpy())
                collected["fact"].append(torch.sigmoid(output.fact_logits).cpu().numpy())
                collected["residual"].append(output.residual.cpu().numpy())
                collected["truth_signature"].append(batch["signatures"].numpy())
                collected["truth_fact"].append(batch["facts"].numpy())
                collected["truth_residual"].append(batch["residual"].numpy())
                collected["hidden"].append(batch["hidden"].numpy())
        return {key: np.concatenate(values, axis=0) for key, values in collected.items()}

    def _quantization_retention(self, test: dict[str, np.ndarray], operation_ids: np.ndarray) -> float:
        code = torch.from_numpy(test["code"]).float().to(self.device)
        with torch.no_grad():
            quantized = self.model.hard_quantize_code(
                code,
                bits=self.config.evaluation.packet_quantization_bits,
            )
            logits, _, _ = self.model.decode_from_code(quantized, self.descriptors)
            prediction = torch.sigmoid(logits).cpu().numpy()
        truth = test["truth_signature"][:, operation_ids]
        original = float(brier_score(truth, test["signature"][:, operation_ids]))
        quantized_brier = float(brier_score(truth, prediction[:, operation_ids]))
        baseline = float(brier_score(truth, np.full_like(truth, 0.5)))
        denominator = baseline - original
        if denominator <= 1e-8:
            return -1.0
        return float((baseline - quantized_brier) / denominator)

    def _packet_audit(self, test: dict[str, np.ndarray]) -> dict[str, object]:
        selector = QueryConditionedSelector(self.bundle.operation_descriptors)
        sample_count = min(8, len(test["world_id"]))
        byte_lengths: list[int] = []
        for index in range(sample_count):
            query_id = int(self.bundle.split.heldout_operation_ids[index % len(self.bundle.split.heldout_operation_ids)])
            probe_ids = selector.select(query_id, self.config.evaluation.packet_probe_count)
            uncertainty = float(
                np.mean(test["signature"][index] * (1.0 - test["signature"][index])) * 4.0
            )
            packet = OperationalPacketV1.build(
                task_family="synthetic_future_operations",
                query_operation_id=query_id,
                fact_probabilities=test["fact"][index],
                signature_probabilities=test["signature"][index],
                probe_ids=probe_ids,
                quantization_bits=self.config.evaluation.packet_quantization_bits,
                uncertainty=uncertainty,
            )
            serialized = packet.serialize()
            recovered = OperationalPacketV1.deserialize(serialized)
            if recovered != packet:
                raise RuntimeError("packet round-trip changed content")
            byte_lengths.append(len(serialized))
        return {
            "samples": sample_count,
            "roundtrip_passed": True,
            "mean_serialized_bytes": float(np.mean(byte_lengths)),
            "max_serialized_bytes": int(max(byte_lengths, default=0)),
        }

    def evaluate(self) -> tuple[dict[str, object], dict[str, object]]:
        all_models = tuple(range(len(self.bundle.model_hidden_dims)))
        train = self._collect(self.bundle.split.train_world_ids, all_models)
        test = self._collect(self.bundle.split.test_world_ids, all_models)
        heldout_ops = np.asarray(self.bundle.split.heldout_operation_ids, dtype=np.int64)
        residual_ops = np.asarray(
            [
                operation.operation_id
                for operation in self.bundle.operations
                if operation.family in {"residual", "hybrid"}
            ],
            dtype=np.int64,
        )

        view_brier = brier_score(
            test["truth_signature"][:, heldout_ops],
            test["signature"][:, heldout_ops],
            axis=1,
        )
        view_fact_accuracy = binary_accuracy(test["truth_fact"], test["fact"], axis=1)
        renderer_scores = renderer_invariance_cosine(
            test["code"],
            test["world_id"],
            test["model_id"],
        )
        retrieval_outcomes, retrieval_margins, retrieval_worlds = cross_model_retrieval(
            test["code"],
            test["world_id"],
            test["model_id"],
        )

        facts_only = facts_only_signatures(
            test["fact"], self.bundle.operation_descriptors, self.bundle.residual.shape[1]
        )
        residual_full_per_view = brier_score(
            test["truth_signature"][:, residual_ops],
            test["signature"][:, residual_ops],
            axis=1,
        )
        residual_fact_per_view = brier_score(
            test["truth_signature"][:, residual_ops],
            facts_only[:, residual_ops],
            axis=1,
        )
        residual_gain_per_view = residual_fact_per_view - residual_full_per_view
        world_brier = self._world_means(view_brier, test["world_id"])
        world_fact_accuracy = self._world_means(view_fact_accuracy, test["world_id"])
        world_residual_gain = self._world_means(residual_gain_per_view, test["world_id"])
        world_retrieval = self._world_means(retrieval_outcomes, retrieval_worlds)
        world_margin = self._world_means(retrieval_margins, retrieval_worlds)

        founder_mask = np.isin(test["model_id"], self.bundle.split.founder_model_ids)
        held_model_id = self.bundle.split.held_model_id
        baseline_brier = brier_score(
            test["truth_signature"][:, heldout_ops],
            np.full_like(test["truth_signature"][:, heldout_ops], 0.5),
            axis=1,
        )
        founder_gain = float(np.mean(baseline_brier[founder_mask] - view_brier[founder_mask]))
        if held_model_id is None:
            held_retention = 1.0
            held_brier = None
        else:
            held_mask = test["model_id"] == held_model_id
            held_gain = float(np.mean(baseline_brier[held_mask] - view_brier[held_mask]))
            held_retention = -1.0 if founder_gain <= 1e-8 else held_gain / founder_gain
            held_brier = float(np.mean(view_brier[held_mask]))

        leakage_accuracy = model_identity_probe(
            train["code"],
            train["model_id"],
            test["code"],
            test["model_id"],
        )
        chance = 1.0 / len(all_models)

        replicates = self.config.evaluation.bootstrap_replicates
        seed = self.config.evaluation.bootstrap_seed
        hidden_r2 = pairwise_hidden_ridge_r2(
            train["hidden"],
            train["world_id"],
            train["model_id"],
            test["hidden"],
            test["world_id"],
            test["model_id"],
        )

        metrics: dict[str, object] = {
            "schema": "frank_eq_stage0_metrics_v1",
            "scope": "synthetic future-defined causal-state Stage 0",
            "n_test_views": int(len(test["world_id"])),
            "n_test_worlds": int(len(np.unique(test["world_id"]))),
            "n_models": len(all_models),
            "heldout_operation_ids": heldout_ops.tolist(),
            "heldout_signature_brier": float(np.mean(view_brier)),
            "heldout_signature_brier_ci": bootstrap_statistic(
                world_brier, replicates=replicates, seed=seed
            ).to_dict(),
            "fact_accuracy": float(np.mean(view_fact_accuracy)),
            "fact_accuracy_ci": bootstrap_statistic(
                world_fact_accuracy, replicates=replicates, seed=seed + 1
            ).to_dict(),
            "renderer_cosine": float(np.mean(renderer_scores)),
            "cross_model_retrieval_top1": float(np.mean(retrieval_outcomes)),
            "cross_model_retrieval_top1_ci": bootstrap_statistic(
                world_retrieval, replicates=replicates, seed=seed + 2
            ).to_dict(),
            "wrong_world_margin": float(np.mean(retrieval_margins)),
            "wrong_world_margin_ci": bootstrap_statistic(
                world_margin, replicates=replicates, seed=seed + 3
            ).to_dict(),
            "residual_brier_gain": float(np.mean(residual_gain_per_view)),
            "residual_brier_gain_ci": bootstrap_statistic(
                world_residual_gain, replicates=replicates, seed=seed + 4
            ).to_dict(),
            "facts_only_residual_brier": float(np.mean(residual_fact_per_view)),
            "full_residual_brier": float(np.mean(residual_full_per_view)),
            "quantization_retention": self._quantization_retention(test, heldout_ops),
            "founder_heldout_brier": float(np.mean(view_brier[founder_mask])),
            "founder_gain_over_constant": founder_gain,
            "held_model_heldout_brier": held_brier,
            "held_model_gain_over_constant": (
                None if held_model_id is None else float(np.mean(baseline_brier[test["model_id"] == held_model_id] - view_brier[test["model_id"] == held_model_id]))
            ),
            "held_model_retention": float(held_retention),
            "model_leakage_accuracy": leakage_accuracy,
            "model_leakage_chance": chance,
            "model_leakage_over_chance": leakage_accuracy - chance,
            "pairwise_hidden_ridge_r2": hidden_r2,
            "mean_pairwise_hidden_ridge_r2": float(np.nanmean(list(hidden_r2.values())))
            if hidden_r2
            else None,
            "packet_audit": self._packet_audit(test),
            "operation_families": list(OPERATION_FAMILIES),
        }
        decision = reduce_stage0(metrics, self.config.gates)
        atomic_write_json(self.output_dir / "metrics.json", metrics)
        atomic_write_json(self.output_dir / "decision.json", decision)
        np.savez_compressed(
            self.output_dir / "predictions.npz",
            world_id=test["world_id"],
            model_id=test["model_id"],
            renderer_id=test["renderer_id"],
            code=test["code"],
            signature=test["signature"],
            fact=test["fact"],
            truth_signature=test["truth_signature"],
            truth_fact=test["truth_fact"],
        )
        manifest = {
            "schema": "frank_eq_artifact_manifest_v1",
            "checkpoint": {
                "path": str(self.checkpoint_path),
                "sha256": sha256_file(self.checkpoint_path),
            },
            "artifacts": {
                name: sha256_file(self.output_dir / name)
                for name in ("metrics.json", "decision.json", "predictions.npz")
            },
        }
        atomic_write_json(self.output_dir / "artifact_manifest.json", manifest)
        return metrics, decision
