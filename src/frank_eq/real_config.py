"""Configuration for the real-checkpoint Stage-A canary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from frank_eq.config import (
    DataConfig,
    EvaluationConfig,
    GateConfig,
    LossConfig,
    ModelConfig,
    RunConfig,
    TrainingConfig,
)

GRAPH_OPERATION_FAMILIES = (
    "lookup",
    "inverse",
    "mutual",
    "compose",
    "compare_outdegree",
    "counterfactual_add",
    "density",
    "reciprocity",
)


@dataclass(slots=True)
class RealModelSpec:
    """One frozen checkpoint participating as founder or held sender."""

    model_id: str
    hf_id: str
    role: str
    tokenizer_id: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False


@dataclass(slots=True)
class RealPanelConfig:
    """Controlled relational-world panel revealed before future operations."""

    n_worlds: int = 48
    n_entities: int = 6
    n_operations: int = 16
    n_renderers: int = 2
    train_fraction: float = 0.65
    validation_fraction: float = 0.15
    operation_holdout_fraction: float = 0.25
    oracle_smoothing: float = 0.02
    min_operation_positive_fraction: float = 0.12
    max_operation_positive_fraction: float = 0.88
    max_generation_attempts: int = 128
    seed: int = 1729

    @property
    def n_facts(self) -> int:
        return self.n_entities * (self.n_entities - 1)

    @property
    def n_residual(self) -> int:
        return 2


@dataclass(slots=True)
class WandBLoggingConfig:
    """Fail-open W&B telemetry settings for the real Stage-A workflow.

    Telemetry is a convenience layer: it never changes scientific outcomes and
    must never fail the workflow. Credentials stay in the environment
    (``WANDB_API_KEY``); only the project identity is configured here.
    """

    enabled: bool = False
    project: str = "frank-eq-stagea"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)
    offline: bool = False

    def validate(self) -> None:
        if self.enabled and not self.project.strip():
            raise ValueError("logging.wandb.project must not be empty when enabled")


@dataclass(slots=True)
class LoggingConfig:
    """Real Stage-A telemetry configuration."""

    wandb: WandBLoggingConfig = field(default_factory=WandBLoggingConfig)

    def validate(self) -> None:
        self.wandb.validate()


@dataclass(slots=True)
class CaptureConfig:
    """Frozen Hugging Face source-capture and future-branch contract."""

    normalized_depths: list[float] = field(default_factory=lambda: [0.35, 0.60, 0.85])
    max_length: int = 512
    dtype: str = "bfloat16"
    device: str = "cuda"
    branch_mode: str = "auto"
    allow_exact_replay_fallback: bool = True
    answer_token_pairs: list[list[str]] = field(
        default_factory=lambda: [[" A", " B"], ["A", "B"], [" No", " Yes"]]
    )
    branch_seed: int = 1729
    branch_batch_size: int = 8
    local_files_only: bool = False


@dataclass(slots=True)
class RealRunConfig:
    """Complete Stage-A cache, chart, and gate configuration."""

    run_name: str = "frank-eq-real-stagea"
    output_dir: str = "runs/real-stagea"
    panel: RealPanelConfig = field(default_factory=RealPanelConfig)
    models: list[RealModelSpec] = field(default_factory=list)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            code_dim=32,
            chart_hidden_dim=128,
            operation_hidden_dim=64,
            decoder_type="graph",
            graph_n_entities=6,
        )
    )
    losses: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            epochs=60,
            onboarding_epochs=40,
            worlds_per_batch=8,
            learning_rate=1e-3,
            onboarding_learning_rate=1.5e-3,
            patience=12,
            device="auto",
        )
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    gates: GateConfig = field(
        default_factory=lambda: GateConfig(
            max_heldout_signature_brier=0.20,
            min_fact_accuracy=0.65,
            min_renderer_cosine=0.80,
            min_cross_model_retrieval_top1=0.20,
            min_wrong_world_margin=0.01,
            min_residual_brier_gain=-0.01,
            min_quantization_retention=0.80,
            min_held_model_retention=0.50,
            max_model_leakage_over_chance=0.30,
        )
    )

    def validate(self) -> None:
        if self.panel.n_worlds < 24:
            raise ValueError("panel.n_worlds must be at least 24")
        if self.panel.n_entities < 4:
            raise ValueError("panel.n_entities must be at least 4")
        family_count = len(GRAPH_OPERATION_FAMILIES)
        if self.panel.n_operations < 2 * family_count:
            raise ValueError(
                f"panel.n_operations must be at least {2 * family_count} so every family has a holdout"
            )
        if self.panel.n_operations % family_count != 0:
            raise ValueError("panel.n_operations must be divisible by the operation-family count")
        if self.panel.n_renderers < 2:
            raise ValueError("real Stage A requires at least two renderers")
        if not 0.0 < self.panel.operation_holdout_fraction < 0.5:
            raise ValueError("panel.operation_holdout_fraction must be in (0, 0.5)")
        if self.panel.train_fraction + self.panel.validation_fraction >= 0.9:
            raise ValueError("at least 10% of worlds must remain for test")
        if not 0.0 <= self.panel.oracle_smoothing < 0.5:
            raise ValueError("panel.oracle_smoothing must be in [0, 0.5)")
        if not self.models:
            raise ValueError("models must not be empty")
        model_ids = [spec.model_id for spec in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model_id values must be unique")
        founders = [spec for spec in self.models if spec.role == "founder"]
        held = [spec for spec in self.models if spec.role == "held"]
        if len(founders) < 2:
            raise ValueError("real Stage A requires at least two founder models")
        if len(held) != 1:
            raise ValueError("real Stage A requires exactly one held sender")
        if self.models[-1].role != "held":
            raise ValueError("the held sender must be the final model entry")
        if any(spec.role not in {"founder", "held"} for spec in self.models):
            raise ValueError("model role must be founder or held")
        if self.capture.branch_mode not in {"auto", "kv_reuse", "exact_prefix_replay"}:
            raise ValueError("capture.branch_mode must be auto, kv_reuse, or exact_prefix_replay")
        self.logging.validate()
        if not self.capture.normalized_depths:
            raise ValueError("capture.normalized_depths must not be empty")
        if any(not 0.0 < depth <= 1.0 for depth in self.capture.normalized_depths):
            raise ValueError("capture.normalized_depths must lie in (0, 1]")
        if len(set(self.capture.normalized_depths)) != len(self.capture.normalized_depths):
            raise ValueError("capture.normalized_depths must be unique")
        if self.capture.max_length < 64:
            raise ValueError("capture.max_length must be at least 64")
        if not self.capture.answer_token_pairs or any(len(pair) != 2 for pair in self.capture.answer_token_pairs):
            raise ValueError("capture.answer_token_pairs must contain false/true token pairs")
        self.model.decoder_type = "graph"
        self.model.graph_n_entities = self.panel.n_entities
        if self.model.graph_temperature <= 0:
            raise ValueError("model.graph_temperature must be positive")

    @property
    def founder_count(self) -> int:
        return sum(spec.role == "founder" for spec in self.models)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_stage0_config(self, model_hidden_dims: list[int]) -> RunConfig:
        """Build the existing trainer/evaluator contract after cache dimensions are known."""

        config = RunConfig(
            run_name=self.run_name,
            output_dir=self.output_dir,
            data=DataConfig(
                n_worlds=self.panel.n_worlds,
                n_founder_models=self.founder_count,
                include_held_model=True,
                n_renderers=self.panel.n_renderers,
                n_layers=len(self.capture.normalized_depths),
                model_hidden_dims=list(model_hidden_dims),
                n_facts=self.panel.n_facts,
                n_residual=self.panel.n_residual,
                n_operations=self.panel.n_operations,
                operation_holdout_fraction=self.panel.operation_holdout_fraction,
                train_fraction=self.panel.train_fraction,
                validation_fraction=self.panel.validation_fraction,
                seed=self.panel.seed,
            ),
            model=self.model,
            losses=self.losses,
            training=self.training,
            evaluation=self.evaluation,
            gates=self.gates,
        )
        config.validate()
        return config


def _construct(cls: type[Any], raw: dict[str, Any] | None) -> Any:
    values = raw or {}
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_real_config(path: str | Path) -> RealRunConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    allowed = {
        "run_name",
        "output_dir",
        "panel",
        "models",
        "capture",
        "logging",
        "model",
        "losses",
        "training",
        "evaluation",
        "gates",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown real-config keys: {sorted(unknown)}")
    defaults = RealRunConfig()
    models = [_construct(RealModelSpec, item) for item in raw.get("models", [])]
    logging_raw = dict(raw.get("logging") or {})
    if "wandb" in logging_raw:
        logging_raw["wandb"] = _construct(WandBLoggingConfig, logging_raw["wandb"])
    config = RealRunConfig(
        run_name=raw.get("run_name", defaults.run_name),
        output_dir=raw.get("output_dir", defaults.output_dir),
        panel=_construct(RealPanelConfig, raw.get("panel")),
        models=models,
        capture=_construct(CaptureConfig, raw.get("capture")),
        logging=_construct(LoggingConfig, logging_raw),
        model=_construct(ModelConfig, raw.get("model")),
        losses=_construct(LossConfig, raw.get("losses")),
        training=_construct(TrainingConfig, raw.get("training")),
        evaluation=_construct(EvaluationConfig, raw.get("evaluation")),
        gates=_construct(GateConfig, raw.get("gates")),
    )
    config.validate()
    return config
