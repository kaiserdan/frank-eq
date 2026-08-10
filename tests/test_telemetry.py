"""Fail-open W&B telemetry behavior for the real Stage-A workflow."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from frank_eq.real_config import LoggingConfig, WandBLoggingConfig
from frank_eq.telemetry import WandbTelemetry, _flatten


class FakeWandbModule:
    def __init__(self) -> None:
        self.init_calls: list[dict] = []
        self.log_calls: list[tuple[dict, int | None]] = []
        self.finished = 0
        self.fail_init = False
        self.fail_log = False

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise RuntimeError("simulated init failure")
        return SimpleNamespace(
            log=self._log,
            finish=lambda: setattr(self, "finished", self.finished + 1),
        )

    def _log(self, payload, step=None):
        if self.fail_log:
            raise RuntimeError("simulated log failure")
        self.log_calls.append((payload, step))


@pytest.fixture
def fake_wandb(monkeypatch) -> FakeWandbModule:
    module = FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def test_flatten_keeps_scalars_and_drops_containers() -> None:
    flat = _flatten(
        {
            "a": 1,
            "b": {"c": 2.5, "d": {"e": True}},
            "skip": [1, 2],
            "none": None,
            "text": "x",
        }
    )
    assert flat == {"a": 1, "b.c": 2.5, "b.d.e": True, "text": "x"}


def test_disabled_by_config_never_touches_wandb(fake_wandb) -> None:
    telemetry = WandbTelemetry(
        WandBLoggingConfig(enabled=False),
        run_name="r",
    )
    telemetry.log({"x": 1})
    telemetry.finish()
    assert telemetry.enabled is False
    assert "disabled" in telemetry.reason
    assert fake_wandb.init_calls == []


def test_missing_api_key_disables(monkeypatch, fake_wandb) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    telemetry = WandbTelemetry(WandBLoggingConfig(enabled=True), run_name="r")
    assert telemetry.enabled is False
    assert "WANDB_API_KEY" in telemetry.reason
    assert fake_wandb.init_calls == []


def test_offline_mode_inits_and_logs(monkeypatch, fake_wandb) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    telemetry = WandbTelemetry(
        WandBLoggingConfig(enabled=True, project="proj", tags=["t"], offline=True),
        run_name="run-1",
        job={"slurm_job_id": "42"},
    )
    assert telemetry.enabled is True
    telemetry.log({"train/founders": {"signature": 0.25, "epoch": 0}}, step=0)
    telemetry.finish()
    assert fake_wandb.init_calls[0]["project"] == "proj"
    assert fake_wandb.init_calls[0]["tags"] == ["t"]
    assert fake_wandb.log_calls == [({"train/founders.signature": 0.25, "train/founders.epoch": 0}, 0)]
    assert fake_wandb.finished == 1


def test_wandb_failures_are_swallowed(monkeypatch, fake_wandb) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    failing = WandbTelemetry(
        WandBLoggingConfig(enabled=True, offline=True),
        run_name="r",
    )
    fake_wandb.fail_init = True
    failing.log({"x": 1})  # must not raise
    assert failing.enabled is False
    assert failing.failures == 1
    assert "init" in failing.reason

    fake_wandb.fail_init = False
    fake_wandb.fail_log = True
    logging = WandbTelemetry(
        WandBLoggingConfig(enabled=True, offline=True),
        run_name="r",
    )
    logging.log({"x": 1})  # must not raise
    logging.finish()
    assert logging.failures == 1


def test_logging_config_validation(tmp_path: Path) -> None:
    from frank_eq.real_config import load_real_config

    base = {
        "run_name": "r",
        "panel": {"n_worlds": 24},
        "models": [
            {"model_id": "m0", "hf_id": "h0", "role": "founder"},
            {"model_id": "m1", "hf_id": "h1", "role": "founder"},
            {"model_id": "m2", "hf_id": "h2", "role": "held"},
        ],
    }
    with_telemetry = {**base, "logging": {"wandb": {"enabled": True, "project": "p"}}}
    config_path = tmp_path / "with.yaml"
    config_path.write_text(__import__("yaml").safe_dump(with_telemetry))
    config = load_real_config(config_path)
    assert config.logging.wandb.enabled is True
    assert config.logging.wandb.project == "p"

    empty_project = {**base, "logging": {"wandb": {"enabled": True, "project": "  "}}}
    config_path = tmp_path / "empty.yaml"
    config_path.write_text(__import__("yaml").safe_dump(empty_project))
    with pytest.raises(ValueError, match="project"):
        load_real_config(config_path)

    unknown = {**base, "logging": {"nope": 1}}
    config_path = tmp_path / "unknown.yaml"
    config_path.write_text(__import__("yaml").safe_dump(unknown))
    with pytest.raises(ValueError, match="LoggingConfig"):
        load_real_config(config_path)


def test_real_workflow_logs_run_cache_train_eval(monkeypatch, fake_wandb, tmp_path: Path) -> None:
    from test_real_cache import FakeHFModelAdapter

    import frank_eq.data.real as real_module
    from frank_eq.config import EvaluationConfig, GateConfig, LossConfig, TrainingConfig
    from frank_eq.real_config import CaptureConfig, RealModelSpec, RealPanelConfig, RealRunConfig
    from frank_eq.workflow import run_real_stagea

    monkeypatch.setattr(real_module, "HFModelAdapter", FakeHFModelAdapter)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    config = RealRunConfig(
        run_name="telemetry-test",
        panel=RealPanelConfig(n_worlds=24, seed=7),
        models=[
            RealModelSpec(model_id="fake-0", hf_id="f0", role="founder"),
            RealModelSpec(model_id="fake-1", hf_id="f1", role="founder"),
            RealModelSpec(model_id="fake-2", hf_id="f2", role="held"),
        ],
        capture=CaptureConfig(
            normalized_depths=[0.25, 0.5, 0.75],
            device="cpu",
            branch_mode="kv_reuse",
            local_files_only=False,
        ),
        logging=LoggingConfig(
            wandb=WandBLoggingConfig(enabled=True, project="test-proj", offline=True)
        ),
        losses=LossConfig(
            facts=0.5,
            residual=0.25,
            renderer_invariance=0.1,
            cross_model_invariance=0.1,
            world_contrastive=0.1,
        ),
        training=TrainingConfig(
            epochs=1,
            onboarding_epochs=1,
            worlds_per_batch=8,
            patience=1,
            device="cpu",
            seed=7,
        ),
        evaluation=EvaluationConfig(bootstrap_replicates=10, bootstrap_seed=7),
        gates=GateConfig(
            max_heldout_signature_brier=1.0,
            min_fact_accuracy=0.0,
            min_renderer_cosine=-1.0,
            min_cross_model_retrieval_top1=0.0,
            min_wrong_world_margin=-2.0,
            min_residual_brier_gain=-1.0,
            min_quantization_retention=-2.0,
            min_held_model_retention=-2.0,
            max_model_leakage_over_chance=1.0,
        ),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(__import__("yaml").safe_dump(config.as_dict()))
    summary = run_real_stagea(
        config,
        config_path=config_path,
        output_dir=tmp_path / "runs",
        stages="cache,validate,train,eval",
    )

    assert summary["status"] == "completed"
    assert summary["telemetry"]["enabled"] is True
    logged = {key for payload, _ in fake_wandb.log_calls for key in payload}
    assert "run.run_name" in logged
    assert "cache.views" in logged
    assert "cache.branch_kv_reuse" in logged
    assert "cache_validation.authorizes_training" in logged
    assert any(key.startswith("train/founders.") for key in logged)
    assert any(key.startswith("phase/founders.") for key in logged)
    assert "eval.heldout_signature_brier" in logged
    assert "decision.decision" in logged
    assert fake_wandb.finished == 1
