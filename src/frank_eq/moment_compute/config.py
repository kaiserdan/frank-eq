"""Frozen configuration for the Stage-M operation-closed event-basis audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from frank_eq.rate_compute.config import ResponseProtocolConfig
from frank_eq.real_config import CaptureConfig, LoggingConfig, RealModelSpec, WandBLoggingConfig


@dataclass(slots=True)
class MomentPanelConfig:
    """Development-only panels with disjoint calibration, selection, and validation roles."""

    entity_counts: list[int] = field(default_factory=lambda: [4])
    worlds_per_complexity: int = 64
    n_target_operations: int = 32
    n_renderers: int = 2
    calibration_fraction: float = 0.50
    selection_fraction: float = 0.20
    oracle_smoothing: float = 0.02
    min_operation_positive_fraction: float = 0.12
    max_operation_positive_fraction: float = 0.88
    max_generation_attempts: int = 512
    seed: int = 20260831


@dataclass(slots=True)
class MomentEvaluationConfig:
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 260831
    calibration_l2: float = 1e-3
    calibration_max_steps: int = 100
    ece_bins: int = 10
    probability_epsilon: float = 1e-7


@dataclass(slots=True)
class MomentGateConfig:
    """Development gates. A pass authorizes only drafting a successor protocol."""

    min_event_brier_gain_lower95: float = 0.0
    min_event_balanced_accuracy_lower95: float = 0.55
    min_moment_over_marginal_gain_lower95: float = 0.0
    min_moment_over_direct_gain_lower95: float = 0.0
    min_atomic_retention_lower95: float = -1e-12
    max_executor_mismatches: int = 0


@dataclass(slots=True)
class MomentAuthorizationConfig:
    development_only: bool = True
    held_sender_authorized: bool = False
    claim_bearing_test_authorized: bool = False
    successor_protocol_draft_requires_pass: bool = True
    receiver_protocol_draft_authorized: bool = False
    receiver_execution_authorized: bool = False
    scientific_claim_authorized: bool = False
    paper_claim_authorized: bool = False


@dataclass(slots=True)
class MomentComputeRunConfig:
    """Complete Stage-M0 operation-closed predictive-state audit contract."""

    schema: str = "frank_eq_moment_compute_registration_v1"
    protocol_version: str = "stage-m0-operation-closed-basis"
    run_name: str = "frank-eq-moment-compute-m0"
    output_dir: str = "runs/moment-compute-m0"
    require_revision_pins: bool = True
    models: list[RealModelSpec] = field(default_factory=list)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    panel: MomentPanelConfig = field(default_factory=MomentPanelConfig)
    protocols: ResponseProtocolConfig = field(default_factory=ResponseProtocolConfig)
    evaluation: MomentEvaluationConfig = field(default_factory=MomentEvaluationConfig)
    gates: MomentGateConfig = field(default_factory=MomentGateConfig)
    authorization: MomentAuthorizationConfig = field(default_factory=MomentAuthorizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        if self.schema != "frank_eq_moment_compute_registration_v1":
            raise ValueError("unsupported moment-compute registration schema")
        if self.protocol_version != "stage-m0-operation-closed-basis":
            raise ValueError("unsupported moment-compute protocol version")
        if len(self.models) < 2:
            raise ValueError("Stage M0 requires at least two founder models")
        if any(model.role != "founder" for model in self.models):
            raise ValueError("Stage M0 accepts founder models only; held senders are forbidden")
        model_ids = [model.model_id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Stage M0 model IDs must be unique")
        if self.require_revision_pins and any(model.revision is None for model in self.models):
            raise ValueError("every Stage M0 checkpoint requires an exact revision pin")

        if self.panel.entity_counts != [4]:
            raise ValueError("Stage M0 is frozen to the four-entity canary")
        if self.panel.worlds_per_complexity < 48:
            raise ValueError("Stage M0 requires at least 48 development worlds")
        if self.panel.n_target_operations < 32 or self.panel.n_target_operations % 8 != 0:
            raise ValueError("Stage M0 target operations must be a multiple of eight and at least 32")
        if self.panel.n_renderers != 2:
            raise ValueError("Stage M0 requires the two historical renderer views")
        if not 0.25 <= self.panel.calibration_fraction <= 0.65:
            raise ValueError("calibration_fraction must lie in [0.25, 0.65]")
        if not 0.10 <= self.panel.selection_fraction <= 0.35:
            raise ValueError("selection_fraction must lie in [0.10, 0.35]")
        if self.panel.calibration_fraction + self.panel.selection_fraction >= 0.85:
            raise ValueError("Stage M0 must reserve at least 15% of worlds for validation")
        if not 0.0 < self.panel.oracle_smoothing < 0.5:
            raise ValueError("oracle_smoothing must lie strictly inside (0, 0.5)")

        if self.capture.prompt_format != "chat_turn":
            raise ValueError("Stage M0 requires corrected chat_turn state formation")
        if self.capture.branch_mode != "kv_reuse":
            raise ValueError("Stage M0 requires literal cloned-KV branching")
        if self.capture.allow_exact_replay_fallback:
            raise ValueError("Stage M0 forbids exact-prefix replay fallback")
        if self.capture.branch_batch_size < 1:
            raise ValueError("branch_batch_size must be positive")
        if not self.capture.local_files_only:
            raise ValueError("Stage M0 cluster execution requires local_files_only=true")

        allowed_protocols = {"answer_token", "sequence", "reason", "pause"}
        if set(self.protocols.target_protocols) - allowed_protocols:
            raise ValueError("Stage M0 contains an unsupported target response protocol")
        if self.protocols.basis_protocol != "sequence":
            raise ValueError("Stage M0 event interrogation is frozen to semantic sequence scoring")
        if self.protocols.rationale_budget != self.protocols.pause_budget:
            raise ValueError("reason and pause budgets must match")
        if self.protocols.rationale_budget < 1:
            raise ValueError("Stage M0 requires positive matched compute budgets")

        if self.evaluation.bootstrap_replicates < 500:
            raise ValueError("Stage M0 requires at least 500 bootstrap replicates")
        if self.evaluation.calibration_max_steps < 1:
            raise ValueError("calibration_max_steps must be positive")
        if not 0.0 < self.evaluation.probability_epsilon < 1e-2:
            raise ValueError("probability_epsilon must lie in (0, 1e-2)")

        authorization = self.authorization
        if not authorization.development_only:
            raise ValueError("Stage M0 must remain development-only")
        if any(
            (
                authorization.held_sender_authorized,
                authorization.claim_bearing_test_authorized,
                authorization.receiver_protocol_draft_authorized,
                authorization.receiver_execution_authorized,
                authorization.scientific_claim_authorized,
                authorization.paper_claim_authorized,
            )
        ):
            raise ValueError("Stage M0 protected authorization fields must remain false")
        self.logging.validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[Any], payload: dict[str, Any] | None) -> Any:
    values = payload or {}
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_moment_compute_config(path: str | Path) -> MomentComputeRunConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"moment-compute configuration not found: {source}")
    raw = yaml.safe_load(source.read_text()) or {}
    allowed = {
        "schema",
        "protocol_version",
        "run_name",
        "output_dir",
        "require_revision_pins",
        "models",
        "capture",
        "panel",
        "protocols",
        "evaluation",
        "gates",
        "authorization",
        "logging",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown moment-compute top-level keys: {sorted(unknown)}")
    defaults = MomentComputeRunConfig()
    logging_raw = dict(raw.get("logging") or {})
    if "wandb" in logging_raw:
        logging_raw["wandb"] = _construct(WandBLoggingConfig, logging_raw["wandb"])
    config = MomentComputeRunConfig(
        schema=raw.get("schema", defaults.schema),
        protocol_version=raw.get("protocol_version", defaults.protocol_version),
        run_name=raw.get("run_name", defaults.run_name),
        output_dir=raw.get("output_dir", defaults.output_dir),
        require_revision_pins=raw.get("require_revision_pins", defaults.require_revision_pins),
        models=[_construct(RealModelSpec, item) for item in raw.get("models", [])],
        capture=_construct(CaptureConfig, raw.get("capture")),
        panel=_construct(MomentPanelConfig, raw.get("panel")),
        protocols=_construct(ResponseProtocolConfig, raw.get("protocols")),
        evaluation=_construct(MomentEvaluationConfig, raw.get("evaluation")),
        gates=_construct(MomentGateConfig, raw.get("gates")),
        authorization=_construct(MomentAuthorizationConfig, raw.get("authorization")),
        logging=_construct(LoggingConfig, logging_raw),
    )
    config.validate()
    return config
