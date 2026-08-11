"""Frozen configuration for the rate--compute operational-basis audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from frank_eq.real_config import (
    CaptureConfig,
    LoggingConfig,
    RealModelSpec,
    WandBLoggingConfig,
)


@dataclass(slots=True)
class RateComputePanelConfig:
    """Development-only graph panels used to localize the Stage-Q wall."""

    entity_counts: list[int] = field(default_factory=lambda: [4, 6])
    worlds_per_complexity: int = 64
    n_target_operations: int = 32
    n_renderers: int = 2
    train_fraction: float = 0.70
    oracle_smoothing: float = 0.02
    min_operation_positive_fraction: float = 0.12
    max_operation_positive_fraction: float = 0.88
    max_generation_attempts: int = 512
    seed: int = 20260820


@dataclass(slots=True)
class ResponseProtocolConfig:
    """Readout and downstream-compute protocols compared on paired worlds."""

    candidate_false: str = " false"
    candidate_true: str = " true"
    basis_protocol: str = "sequence"
    target_protocols: list[str] = field(
        default_factory=lambda: ["answer_token", "sequence", "reason", "pause"]
    )
    compute_families: list[str] = field(
        default_factory=lambda: [
            "mutual",
            "compose",
            "compare_outdegree",
            "counterfactual_add",
        ]
    )
    rationale_budget: int = 32
    pause_budget: int = 32
    pause_text: str = " ..."
    reasoning_instruction: str = (
        "Work through the registered operation before answering. "
        "Use the following scratchpad positions to reason, and do not give the final answer yet."
    )
    final_cue: str = "\nNow give only the final truth value. Final answer:"
    sequence_cue: str = "\nGive only the final truth value. Answer:"
    normalize_sequence_log_likelihood: bool = True
    max_saved_reasoning_characters: int = 512


@dataclass(slots=True)
class RateComputeEvaluationConfig:
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 991
    calibration_l2: float = 1e-3
    calibration_max_steps: int = 100
    ece_bins: int = 10
    basis_quantization_bits: list[int] = field(default_factory=lambda: [1, 2, 4, 8])


@dataclass(slots=True)
class RateComputeGateConfig:
    """Development gates; no value directly authorizes a claim-bearing run."""

    min_basis_brier_gain_lower95: float = 0.0
    min_basis_balanced_accuracy: float = 0.60
    min_compiled_prior_gain_lower95: float = 0.0
    min_compiled_direct_gain_lower95: float = 0.0
    min_answer_channel_gain_lower95: float = 0.0
    min_reason_over_pause_gain_lower95: float = 0.0


@dataclass(slots=True)
class RateComputeRunConfig:
    """Complete development-only rate--compute audit contract."""

    run_name: str = "frank-eq-rate-compute-rc0"
    output_dir: str = "runs/rate-compute-rc0"
    require_revision_pins: bool = True
    models: list[RealModelSpec] = field(default_factory=list)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    panel: RateComputePanelConfig = field(default_factory=RateComputePanelConfig)
    protocols: ResponseProtocolConfig = field(default_factory=ResponseProtocolConfig)
    evaluation: RateComputeEvaluationConfig = field(default_factory=RateComputeEvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    gates: RateComputeGateConfig = field(default_factory=RateComputeGateConfig)

    def validate(self) -> None:
        if len(self.models) < 2:
            raise ValueError("rate--compute audit requires at least two founder models")
        if any(model.role != "founder" for model in self.models):
            raise ValueError("rate--compute audit accepts founder models only; no held role")
        model_ids = [model.model_id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("rate--compute model_id values must be unique")
        if self.require_revision_pins and any(model.revision is None for model in self.models):
            raise ValueError("every rate--compute model requires an exact revision pin")

        if not self.panel.entity_counts:
            raise ValueError("panel.entity_counts must not be empty")
        if sorted(set(self.panel.entity_counts)) != sorted(self.panel.entity_counts):
            raise ValueError("panel.entity_counts must be unique")
        if any(value < 4 or value > 10 for value in self.panel.entity_counts):
            raise ValueError("panel.entity_counts must lie in [4, 10]")
        if self.panel.worlds_per_complexity < 32:
            raise ValueError("worlds_per_complexity must be at least 32")
        if self.panel.n_target_operations < 16 or self.panel.n_target_operations % 8 != 0:
            raise ValueError("n_target_operations must be a multiple of eight and at least 16")
        if self.panel.n_renderers != 2:
            raise ValueError("the current graph panel requires exactly two renderers")
        if not 0.5 <= self.panel.train_fraction < 0.9:
            raise ValueError("panel.train_fraction must lie in [0.5, 0.9)")
        if not 0.0 <= self.panel.oracle_smoothing < 0.5:
            raise ValueError("panel.oracle_smoothing must lie in [0, 0.5)")

        if self.capture.prompt_format != "chat_turn":
            raise ValueError("rate--compute audit requires prompt_format=chat_turn")
        if self.capture.branch_mode != "kv_reuse":
            raise ValueError("rate--compute audit requires literal kv_reuse branching")
        if self.capture.allow_exact_replay_fallback:
            raise ValueError("rate--compute audit forbids exact-replay fallback")
        if self.capture.max_length < 256:
            raise ValueError("capture.max_length must be at least 256")

        allowed_protocols = {"answer_token", "sequence", "reason", "pause"}
        if not self.protocols.target_protocols:
            raise ValueError("protocols.target_protocols must not be empty")
        unknown = set(self.protocols.target_protocols) - allowed_protocols
        if unknown:
            raise ValueError(f"unsupported target protocols: {sorted(unknown)}")
        if self.protocols.basis_protocol not in {"sequence", "reason", "pause"}:
            raise ValueError("basis_protocol must be sequence, reason, or pause")
        graph_families = {
            "lookup",
            "inverse",
            "mutual",
            "compose",
            "compare_outdegree",
            "counterfactual_add",
            "density",
            "reciprocity",
        }
        unknown_compute = set(self.protocols.compute_families) - graph_families
        if unknown_compute:
            raise ValueError(f"unsupported compute_families: {sorted(unknown_compute)}")
        if not self.protocols.compute_families:
            raise ValueError("protocols.compute_families must not be empty")
        if self.protocols.candidate_false == self.protocols.candidate_true:
            raise ValueError("candidate truth strings must differ")
        if not self.protocols.candidate_false or not self.protocols.candidate_true:
            raise ValueError("candidate truth strings must be non-empty")
        if self.protocols.rationale_budget < 0 or self.protocols.pause_budget < 0:
            raise ValueError("compute budgets must be non-negative")
        if "reason" in self.protocols.target_protocols and self.protocols.rationale_budget < 1:
            raise ValueError("reason protocol requires a positive rationale budget")
        if "pause" in self.protocols.target_protocols and self.protocols.pause_budget < 1:
            raise ValueError("pause protocol requires a positive pause budget")
        if self.protocols.rationale_budget != self.protocols.pause_budget:
            raise ValueError("reason and pause budgets must match for the primary compute contrast")
        if not self.protocols.pause_text:
            raise ValueError("protocols.pause_text must be non-empty")

        if self.evaluation.bootstrap_replicates < 200:
            raise ValueError("bootstrap_replicates must be at least 200")
        if self.evaluation.calibration_l2 < 0:
            raise ValueError("calibration_l2 must be non-negative")
        if self.evaluation.calibration_max_steps < 1:
            raise ValueError("calibration_max_steps must be positive")
        if self.evaluation.ece_bins < 2:
            raise ValueError("ece_bins must be at least two")
        bits = self.evaluation.basis_quantization_bits
        if not bits or sorted(set(bits)) != sorted(bits):
            raise ValueError("basis_quantization_bits must be non-empty and unique")
        if any(value < 1 or value > 16 for value in bits):
            raise ValueError("basis_quantization_bits must lie in [1, 16]")
        self.logging.validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[Any], payload: dict[str, Any] | None) -> Any:
    values = payload or {}
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_rate_compute_config(path: str | Path) -> RateComputeRunConfig:
    """Load and fail-closed validate one frozen audit configuration."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"rate--compute configuration not found: {source}")
    raw = yaml.safe_load(source.read_text()) or {}
    allowed = {
        "run_name",
        "output_dir",
        "require_revision_pins",
        "models",
        "capture",
        "panel",
        "protocols",
        "evaluation",
        "logging",
        "gates",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown rate--compute top-level keys: {sorted(unknown)}")
    defaults = RateComputeRunConfig()
    logging_raw = dict(raw.get("logging") or {})
    if "wandb" in logging_raw:
        logging_raw["wandb"] = _construct(WandBLoggingConfig, logging_raw["wandb"])
    config = RateComputeRunConfig(
        run_name=raw.get("run_name", defaults.run_name),
        output_dir=raw.get("output_dir", defaults.output_dir),
        require_revision_pins=raw.get(
            "require_revision_pins", defaults.require_revision_pins
        ),
        models=[_construct(RealModelSpec, item) for item in raw.get("models", [])],
        capture=_construct(CaptureConfig, raw.get("capture")),
        panel=_construct(RateComputePanelConfig, raw.get("panel")),
        protocols=_construct(ResponseProtocolConfig, raw.get("protocols")),
        evaluation=_construct(RateComputeEvaluationConfig, raw.get("evaluation")),
        logging=_construct(LoggingConfig, logging_raw),
        gates=_construct(RateComputeGateConfig, raw.get("gates")),
    )
    config.validate()
    return config
