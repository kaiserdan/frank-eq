"""Configuration loading and validation for Frank-EQ experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    """Synthetic Stage-0 data construction settings."""

    n_worlds: int = 384
    n_founder_models: int = 3
    include_held_model: bool = True
    n_renderers: int = 2
    n_layers: int = 3
    model_hidden_dims: list[int] = field(default_factory=lambda: [48, 64, 80, 56])
    n_facts: int = 12
    n_residual: int = 4
    n_operations: int = 36
    operation_holdout_fraction: float = 0.25
    train_fraction: float = 0.65
    validation_fraction: float = 0.15
    observation_noise: float = 0.02
    renderer_nuisance_scale: float = 0.7
    model_private_scale: float = 0.35
    seed: int = 1729

    @property
    def n_models(self) -> int:
        return self.n_founder_models + int(self.include_held_model)


@dataclass(slots=True)
class ModelConfig:
    """Operational quotient model settings."""

    code_dim: int = 24
    chart_hidden_dim: int = 96
    operation_hidden_dim: int = 64
    dropout: float = 0.05
    quantization_bits: int = 8
    use_workspace_gate: bool = True
    workspace_l1: float = 2e-4
    gradient_reversal_weight: float = 0.10


@dataclass(slots=True)
class LossConfig:
    """Training objective weights."""

    signature: float = 1.0
    facts: float = 0.40
    residual: float = 0.20
    renderer_invariance: float = 0.20
    cross_model_invariance: float = 0.10
    world_contrastive: float = 0.15
    model_adversary: float = 0.05
    code_variance_floor: float = 0.02


@dataclass(slots=True)
class TrainingConfig:
    """Optimization and reproducibility settings."""

    epochs: int = 80
    onboarding_epochs: int = 50
    worlds_per_batch: int = 16
    learning_rate: float = 2e-3
    onboarding_learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    patience: int = 15
    num_threads: int = 1
    device: str = "auto"
    seed: int = 1729


@dataclass(slots=True)
class EvaluationConfig:
    """Evaluation, packet, and uncertainty settings."""

    bootstrap_replicates: int = 500
    bootstrap_seed: int = 991
    packet_probe_count: int = 8
    packet_quantization_bits: int = 8
    retrieval_chunk_size: int = 512


@dataclass(slots=True)
class GateConfig:
    """Prospective Stage-0 promotion gates.

    Thresholds intentionally target a synthetic proof-of-implementation, not a
    scientific claim about real LLMs. Real-model campaigns must freeze their own
    gates before any outcome-bearing evaluation.
    """

    max_heldout_signature_brier: float = 0.16
    min_fact_accuracy: float = 0.82
    min_renderer_cosine: float = 0.90
    min_cross_model_retrieval_top1: float = 0.65
    min_wrong_world_margin: float = 0.08
    min_residual_brier_gain: float = 0.015
    min_quantization_retention: float = 0.90
    min_held_model_retention: float = 0.70
    max_model_leakage_over_chance: float = 0.18


@dataclass(slots=True)
class RunConfig:
    """Top-level Frank-EQ experiment configuration."""

    run_name: str = "frank-eq-stage0"
    output_dir: str = "runs/stage0"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    gates: GateConfig = field(default_factory=GateConfig)

    def validate(self) -> None:
        if self.data.n_worlds < 24:
            raise ValueError("data.n_worlds must be at least 24 for grouped splits")
        if self.data.n_founder_models < 2:
            raise ValueError("at least two founder models are required")
        if len(self.data.model_hidden_dims) < self.data.n_models:
            raise ValueError(
                "model_hidden_dims must contain one dimension per founder and held model"
            )
        if self.data.n_renderers < 2:
            raise ValueError("renderer invariance requires at least two renderers")
        if not 0.0 < self.data.operation_holdout_fraction < 0.5:
            raise ValueError("operation_holdout_fraction must be in (0, 0.5)")
        if self.data.train_fraction <= 0 or self.data.validation_fraction <= 0:
            raise ValueError("train and validation fractions must be positive")
        if self.data.train_fraction + self.data.validation_fraction >= 0.9:
            raise ValueError("at least 10% of worlds must remain for the test split")
        if self.model.code_dim < 4:
            raise ValueError("model.code_dim must be at least 4")
        if not 1 <= self.model.quantization_bits <= 16:
            raise ValueError("model.quantization_bits must be between 1 and 16")
        if self.training.epochs < 1 or self.training.onboarding_epochs < 1:
            raise ValueError("training epochs must be positive")
        if self.training.num_threads < 1:
            raise ValueError("training.num_threads must be positive")
        if self.evaluation.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_dataclass(cls: type[Any], raw: dict[str, Any] | None) -> Any:
    values = raw or {}
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML experiment configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    allowed = {"run_name", "output_dir", "data", "model", "losses", "training", "evaluation", "gates"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")

    defaults = RunConfig()
    config = RunConfig(
        run_name=raw.get("run_name", defaults.run_name),
        output_dir=raw.get("output_dir", defaults.output_dir),
        data=_construct_dataclass(DataConfig, raw.get("data")),
        model=_construct_dataclass(ModelConfig, raw.get("model")),
        losses=_construct_dataclass(LossConfig, raw.get("losses")),
        training=_construct_dataclass(TrainingConfig, raw.get("training")),
        evaluation=_construct_dataclass(EvaluationConfig, raw.get("evaluation")),
        gates=_construct_dataclass(GateConfig, raw.get("gates")),
    )
    config.validate()
    return config
