from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from frank_eq.contracts import FutureBranchRecord, FutureSignatureRecord, StateCaptureRecord
from frank_eq.data.hf_backend import CapturedModelRows
from frank_eq.data.real import RealBundle, build_real_cache, validate_real_cache
from frank_eq.real_config import RealModelSpec, RealPanelConfig, RealRunConfig
from frank_eq.utils import sha256_bytes


class FakeHFModelAdapter:
    def __init__(self, spec, capture):
        self.spec = spec
        self.capture = capture

    def capture_panel(self, panel, split_by_world):
        model_number = int(self.spec.model_id[-1])
        hidden_dim = 10 + model_number
        hidden_rows = []
        world_ids = []
        renderer_ids = []
        signatures = []
        records = []
        required = {str(row.definition.operation_id) for row in panel.operations}
        for world in panel.worlds:
            for renderer_id in range(int(panel.config["n_renderers"])):
                facts = world.fact_vector().astype(np.float32)
                base = np.concatenate(
                    [facts[: hidden_dim - 2], np.asarray(world.residual_vector(), dtype=np.float32)]
                )
                if len(base) < hidden_dim:
                    base = np.pad(base, (0, hidden_dim - len(base)))
                layers = np.stack(
                    [base[:hidden_dim] + 0.01 * layer + 0.001 * renderer_id for layer in range(3)]
                ).astype(np.float32)
                state_id = f"w{world.world_id}-m{model_number}-r{renderer_id}"
                oracle = np.asarray(panel.oracle_signatures[world.world_id], dtype=np.float32)
                branch = np.clip(oracle + (model_number - 1) * 0.01, 1e-4, 1 - 1e-4)
                capture = StateCaptureRecord(
                    state_id=state_id,
                    world_id=str(world.world_id),
                    model_id=self.spec.model_id,
                    renderer_id=str(renderer_id),
                    split=split_by_world[world.world_id],
                    prefix_sha256=sha256_bytes(f"prefix-{world.world_id}-{renderer_id}".encode()),
                    hidden_artifact_sha256=sha256_bytes(layers.tobytes()),
                    captured_before_operation=True,
                    capture_step=20,
                )
                branches = tuple(
                    FutureBranchRecord(
                        state_id=state_id,
                        operation_id=str(operation.definition.operation_id),
                        operation_descriptor_sha256=operation.descriptor_sha256,
                        outcome_probabilities=(1.0 - float(probability), float(probability)),
                        branch_seed=1729,
                        operation_reveal_step=21,
                    )
                    for operation, probability in zip(panel.operations, branch, strict=True)
                )
                record = FutureSignatureRecord(capture=capture, branches=branches)
                record.validate(required)
                hidden_rows.append(layers)
                world_ids.append(world.world_id)
                renderer_ids.append(renderer_id)
                signatures.append(branch)
                records.append(record)
        return CapturedModelRows(
            hidden=np.stack(hidden_rows),
            world_ids=np.asarray(world_ids, dtype=np.int64),
            renderer_ids=np.asarray(renderer_ids, dtype=np.int64),
            branch_signatures=np.stack(signatures),
            records=records,
            hidden_dim=hidden_dim,
            layer_indices=[1, 2, 3],
            answer_labels=(" A", " B"),
            branch_mode_counts={"kv_reuse": len(records) * len(panel.operations), "exact_prefix_replay": 0},
            parity_audit={
                "sample_size": int(getattr(self.capture, "parity_sample_size", 0)),
                "entries": [
                    {
                        "state_id": "parity",
                        "operation_id": str(operation.definition.operation_id),
                        "kv_probability": 0.5 + 0.01 * model_number,
                        "replay_probability": 0.5
                        + 0.01 * model_number
                        + float(getattr(self.capture, "parity_audit_bias", 0.0)),
                    }
                    for operation in panel.operations[: int(getattr(self.capture, "parity_sample_size", 0))]
                ],
            },
            model_revision="fake-revision",
        )


def test_real_cache_build_and_validation_are_complete(monkeypatch, tmp_path: Path) -> None:
    import frank_eq.data.real as real_module

    monkeypatch.setattr(real_module, "HFModelAdapter", FakeHFModelAdapter)
    config = RealRunConfig(
        panel=RealPanelConfig(n_worlds=24, n_entities=5, n_operations=16, seed=31),
        models=[
            RealModelSpec("model-0", "fake/0", "founder"),
            RealModelSpec("model-1", "fake/1", "founder"),
            RealModelSpec("model-2", "fake/2", "held"),
        ],
    )
    config.capture.normalized_depths = [0.25, 0.5, 0.75]
    cache = tmp_path / "cache"
    summary = build_real_cache(config, cache)
    validation = validate_real_cache(cache)
    bundle = RealBundle.load(cache)
    assert summary["status"] == "passed"
    assert validation["authorizes_training"] is True
    assert bundle.n_views == 24 * 3 * 2
    assert bundle.model_hidden_dims == [10, 11, 12]
    assert validation["causal_boundary_passed"] is True


def test_real_cache_parity_audit_fails_on_divergence(monkeypatch, tmp_path: Path) -> None:
    import frank_eq.data.real as real_module

    class BiasedParityAdapter(FakeHFModelAdapter):
        def capture_panel(self, panel, split_by_world):
            rows = super().capture_panel(panel, split_by_world)
            for entry in rows.parity_audit["entries"]:
                entry["replay_probability"] += 0.1
            return rows

    monkeypatch.setattr(real_module, "HFModelAdapter", BiasedParityAdapter)
    config = RealRunConfig(
        panel=RealPanelConfig(n_worlds=24, n_entities=5, n_operations=16, seed=31),
        models=[
            RealModelSpec("model-0", "fake/0", "founder"),
            RealModelSpec("model-1", "fake/1", "founder"),
            RealModelSpec("model-2", "fake/2", "held"),
        ],
    )
    config.capture.normalized_depths = [0.25, 0.5, 0.75]
    config.capture.parity_sample_size = 8
    config.capture.parity_max_abs_diff = 0.001
    try:
        build_real_cache(config, tmp_path / "cache")
    except RuntimeError as error:
        assert "parity" in str(error).lower()
    else:
        raise AssertionError("divergent parity sample must fail the cache build")


def test_real_cache_parity_audit_passes_when_parity_holds(monkeypatch, tmp_path: Path) -> None:
    import frank_eq.data.real as real_module

    monkeypatch.setattr(real_module, "HFModelAdapter", FakeHFModelAdapter)
    config = RealRunConfig(
        panel=RealPanelConfig(n_worlds=24, n_entities=5, n_operations=16, seed=31),
        models=[
            RealModelSpec("model-0", "fake/0", "founder"),
            RealModelSpec("model-1", "fake/1", "founder"),
            RealModelSpec("model-2", "fake/2", "held"),
        ],
    )
    config.capture.normalized_depths = [0.25, 0.5, 0.75]
    config.capture.parity_sample_size = 8
    config.capture.parity_max_abs_diff = 0.01
    cache = tmp_path / "cache"
    summary = build_real_cache(config, cache)
    assert summary["status"] == "passed"
    metadata = json.loads((cache / "metadata.json").read_text())
    for entry in metadata["extraction"]["models"]:
        assert entry["parity_audit"]["sample_size"] == 8
        assert entry["parity_audit"]["max_abs_diff"] <= 0.01


def test_real_cache_parity_audit_requested_but_missing_fails(monkeypatch, tmp_path: Path) -> None:
    import frank_eq.data.real as real_module

    class MissingParityAdapter(FakeHFModelAdapter):
        def capture_panel(self, panel, split_by_world):
            rows = super().capture_panel(panel, split_by_world)
            rows.parity_audit = {"sample_size": 8, "entries": []}
            return rows

    monkeypatch.setattr(real_module, "HFModelAdapter", MissingParityAdapter)
    config = RealRunConfig(
        panel=RealPanelConfig(n_worlds=24, n_entities=5, n_operations=16, seed=31),
        models=[
            RealModelSpec("model-0", "fake/0", "founder"),
            RealModelSpec("model-1", "fake/1", "founder"),
            RealModelSpec("model-2", "fake/2", "held"),
        ],
    )
    config.capture.normalized_depths = [0.25, 0.5, 0.75]
    config.capture.parity_sample_size = 8
    try:
        build_real_cache(config, tmp_path / "cache")
    except RuntimeError as error:
        assert "no dual-mode sample" in str(error)
    else:
        raise AssertionError("a requested-but-missing parity sample must fail the build")
