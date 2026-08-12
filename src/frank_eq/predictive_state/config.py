"""Frozen configuration for the development-only PSR0 predictive-state census."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from frank_eq.real_config import (
    CaptureConfig,
    LoggingConfig,
    RealModelSpec,
    WandBLoggingConfig,
)

from .automaton import PredictiveAutomaton


@dataclass(slots=True)
class PredictiveAutomatonConfig:
    state_names: list[str] = field(default_factory=lambda: ["S0", "S1", "S2", "S3"])
    action_names: list[str] = field(default_factory=lambda: ["orbit", "fold", "shift"])
    observation_names: list[str] = field(default_factory=lambda: ["amber", "blue", "coral"])
    transition_matrices: list[list[list[float]]] = field(default_factory=list)
    emission_matrix: list[list[float]] = field(default_factory=list)
    initial_belief: list[float] = field(default_factory=lambda: [0.25] * 4)
    candidate_horizons: list[int] = field(default_factory=lambda: [1, 2, 3])
    n_target_tests: int = 18
    target_seed: int = 20260831
    max_core_condition_number: float = 10.0
    max_target_executor_l1: float = 3.0


@dataclass(slots=True)
class PredictivePanelRoleConfig:
    lengths: list[int] = field(default_factory=list)
    histories_per_length: int = 64
    seed: int = 0


@dataclass(slots=True)
class PredictivePanelConfig:
    train: PredictivePanelRoleConfig = field(
        default_factory=lambda: PredictivePanelRoleConfig(
            lengths=[8, 16], histories_per_length=128, seed=2026083201
        )
    )
    validation: PredictivePanelRoleConfig = field(
        default_factory=lambda: PredictivePanelRoleConfig(
            lengths=[8, 16, 32], histories_per_length=64, seed=2026083202
        )
    )
    fit_renderers: list[str] = field(default_factory=lambda: ["narrative", "table"])
    validation_renderers: list[str] = field(
        default_factory=lambda: ["narrative", "table", "symbolic"]
    )
    min_belief_entropy: float = 0.15
    max_belief_entropy: float = 1.35
    min_core_variance: float = 0.0025
    max_generation_attempt_multiplier: int = 100


@dataclass(slots=True)
class PredictiveTeacherConfig:
    candidate_false: str = " false"
    candidate_true: str = " true"
    sequence_cue: str = "\nGive only the final truth value. Answer:"
    normalize_sequence_log_likelihood: bool = True
    branch_batch_size: int = 8


@dataclass(slots=True)
class PredictiveProbeConfig:
    ridge_grid: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    selection_fraction: float = 0.20
    selection_seed: int = 20260833
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 991
    calibration_l2: float = 1e-3
    calibration_max_steps: int = 100
    token_hash_position_period: int = 64
    quantization_bits: list[int] = field(default_factory=lambda: [2, 4, 8])


@dataclass(slots=True)
class PredictiveGateConfig:
    min_core_gain_over_prior_lower95: float = 0.0
    min_activation_over_token_lower95: float = 0.0
    min_wrong_history_margin_lower95: float = 0.0
    min_unseen_renderer_gain_lower95: float = 0.0
    min_length_transfer_gain_lower95: float = 0.0
    min_compiled_over_prior_lower95: float = 0.0
    min_compiled_over_direct_lower95: float = 0.0
    max_oracle_executor_abs_error: float = 1e-10
    max_public_basis_condition_number: float = 10.0
    max_renderer_pair_brier_gap: float = 0.02


@dataclass(slots=True)
class PredictiveAuthorizationConfig:
    development_run_authorized: bool = True
    claim_bearing_test_access_authorized: bool = False
    held_sender_onboarding_authorized: bool = False
    receiver_execution_authorized: bool = False
    scientific_claim_authorized: bool = False
    paper_claim_authorized: bool = False


@dataclass(slots=True)
class PredictiveStateRunConfig:
    protocol_version: str = "frank_eq_predictive_state_psr0_v1"
    run_name: str = "frank-eq-predictive-state-psr0"
    output_dir: str = "runs/predictive-state-psr0"
    development_only: bool = True
    require_revision_pins: bool = True
    models: list[RealModelSpec] = field(default_factory=list)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    automaton: PredictiveAutomatonConfig = field(default_factory=PredictiveAutomatonConfig)
    panel: PredictivePanelConfig = field(default_factory=PredictivePanelConfig)
    teacher: PredictiveTeacherConfig = field(default_factory=PredictiveTeacherConfig)
    probe: PredictiveProbeConfig = field(default_factory=PredictiveProbeConfig)
    gates: PredictiveGateConfig = field(default_factory=PredictiveGateConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    authorization: PredictiveAuthorizationConfig = field(
        default_factory=PredictiveAuthorizationConfig
    )

    def build_automaton(self) -> PredictiveAutomaton:
        return PredictiveAutomaton(
            state_names=self.automaton.state_names,
            action_names=self.automaton.action_names,
            observation_names=self.automaton.observation_names,
            transition_matrices=np.asarray(
                self.automaton.transition_matrices, dtype=np.float64
            ),
            emission_matrix=np.asarray(self.automaton.emission_matrix, dtype=np.float64),
            initial_belief=np.asarray(self.automaton.initial_belief, dtype=np.float64),
        )

    def validate(self) -> None:
        if self.protocol_version != "frank_eq_predictive_state_psr0_v1":
            raise ValueError("unsupported predictive-state protocol version")
        if not self.development_only:
            raise ValueError("PSR0 must remain development-only")
        if len(self.models) < 2:
            raise ValueError("PSR0 requires at least two development models")
        if any(model.role != "founder" for model in self.models):
            raise ValueError("PSR0 accepts development founder roles only")
        if len({model.model_id for model in self.models}) != len(self.models):
            raise ValueError("PSR0 model IDs must be unique")
        if self.require_revision_pins and any(model.revision is None for model in self.models):
            raise ValueError("PSR0 model revisions must be pinned")

        if self.capture.prompt_format != "chat_turn":
            raise ValueError("PSR0 requires corrected chat_turn capture")
        if self.capture.branch_mode != "kv_reuse":
            raise ValueError("PSR0 requires literal KV reuse")
        if self.capture.allow_exact_replay_fallback:
            raise ValueError("PSR0 forbids exact-replay fallback")
        if self.capture.max_length < 1024:
            raise ValueError("PSR0 capture.max_length must be at least 1024")
        if len(self.capture.normalized_depths) < 3:
            raise ValueError("PSR0 requires at least three registered depths")

        automaton = self.build_automaton()
        if (automaton.n_states, automaton.n_actions, automaton.n_observations) != (
            4,
            3,
            3,
        ):
            raise ValueError("PSR0 renderer contract requires 4 states, 3 actions, 3 observations")
        if not np.allclose(automaton.initial_belief, np.full(4, 0.25), atol=1e-12):
            raise ValueError("PSR0 renderer contract requires the frozen uniform prior")
        basis = automaton.build_basis(
            horizons=self.automaton.candidate_horizons,
            n_target_tests=self.automaton.n_target_tests,
            target_seed=self.automaton.target_seed,
            max_condition_number=self.automaton.max_core_condition_number,
            max_target_l1=self.automaton.max_target_executor_l1,
        )
        if basis.condition_number > self.gates.max_public_basis_condition_number:
            raise ValueError("PSR0 public basis exceeds the frozen gate condition number")

        renderer_registry = {"narrative", "table", "symbolic"}
        if set(self.panel.fit_renderers) != {"narrative", "table"}:
            raise ValueError("PSR0 fit renderers must be narrative and table")
        if set(self.panel.validation_renderers) != renderer_registry:
            raise ValueError("PSR0 validation must include the unseen symbolic renderer")
        for role_name, role in (
            ("train", self.panel.train),
            ("validation", self.panel.validation),
        ):
            if not role.lengths or any(length < 4 for length in role.lengths):
                raise ValueError(f"PSR0 {role_name} lengths must be at least four")
            if len(set(role.lengths)) != len(role.lengths):
                raise ValueError(f"PSR0 {role_name} lengths must be unique")
            if role.histories_per_length < 32:
                raise ValueError(f"PSR0 {role_name} needs at least 32 histories per length")
        if 32 not in self.panel.validation.lengths or 32 in self.panel.train.lengths:
            raise ValueError("PSR0 freezes length 32 as validation-only transfer")
        if not 0.0 <= self.panel.min_belief_entropy < self.panel.max_belief_entropy:
            raise ValueError("PSR0 belief-entropy interval is invalid")
        if self.panel.max_belief_entropy > np.log(automaton.n_states) + 1e-9:
            raise ValueError("PSR0 maximum belief entropy exceeds log(number of states)")
        if self.panel.min_core_variance < 0:
            raise ValueError("PSR0 minimum core variance must be non-negative")
        if self.panel.max_generation_attempt_multiplier < 1:
            raise ValueError("PSR0 generation-attempt multiplier must be positive")

        if self.teacher.candidate_false == self.teacher.candidate_true:
            raise ValueError("PSR0 semantic candidate strings must differ")
        if self.teacher.branch_batch_size < 1:
            raise ValueError("PSR0 branch batch size must be positive")
        if not self.probe.ridge_grid or any(value <= 0 for value in self.probe.ridge_grid):
            raise ValueError("PSR0 ridge grid must contain positive values")
        if not 0.1 <= self.probe.selection_fraction <= 0.4:
            raise ValueError("PSR0 selection fraction must lie in [0.1, 0.4]")
        if self.probe.bootstrap_replicates < 500:
            raise ValueError("PSR0 requires at least 500 bootstrap replicates")
        if self.probe.token_hash_position_period < 8:
            raise ValueError("PSR0 token-hash position period must be at least eight")
        if not self.probe.quantization_bits or any(
            value < 1 or value > 16 for value in self.probe.quantization_bits
        ):
            raise ValueError("PSR0 quantization bits must lie in [1,16]")
        if self.gates.max_oracle_executor_abs_error <= 0:
            raise ValueError("PSR0 oracle-executor tolerance must be positive")
        if self.gates.max_public_basis_condition_number <= 0:
            raise ValueError("PSR0 basis-condition gate must be positive")
        if self.gates.max_renderer_pair_brier_gap < 0:
            raise ValueError("PSR0 renderer-gap gate must be non-negative")

        if self.authorization.development_run_authorized is not True:
            raise ValueError("PSR0 development execution must be explicitly authorized")
        protected = (
            self.authorization.claim_bearing_test_access_authorized,
            self.authorization.held_sender_onboarding_authorized,
            self.authorization.receiver_execution_authorized,
            self.authorization.scientific_claim_authorized,
            self.authorization.paper_claim_authorized,
        )
        if any(protected):
            raise ValueError("PSR0 cannot authorize protected roles or claims")
        self.logging.validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[Any], payload: dict[str, Any] | None) -> Any:
    values = dict(payload or {})
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_predictive_state_config(path: str | Path) -> PredictiveStateRunConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"predictive-state configuration not found: {source}")
    raw = yaml.safe_load(source.read_text()) or {}
    allowed = set(PredictiveStateRunConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown predictive-state top-level keys: {sorted(unknown)}")
    defaults = PredictiveStateRunConfig()

    panel_raw = dict(raw.get("panel") or {})
    if "train" in panel_raw:
        panel_raw["train"] = _construct(PredictivePanelRoleConfig, panel_raw["train"])
    if "validation" in panel_raw:
        panel_raw["validation"] = _construct(
            PredictivePanelRoleConfig, panel_raw["validation"]
        )
    logging_raw = dict(raw.get("logging") or {})
    if "wandb" in logging_raw:
        logging_raw["wandb"] = _construct(WandBLoggingConfig, logging_raw["wandb"])

    config = PredictiveStateRunConfig(
        protocol_version=raw.get("protocol_version", defaults.protocol_version),
        run_name=raw.get("run_name", defaults.run_name),
        output_dir=raw.get("output_dir", defaults.output_dir),
        development_only=raw.get("development_only", defaults.development_only),
        require_revision_pins=raw.get(
            "require_revision_pins", defaults.require_revision_pins
        ),
        models=[_construct(RealModelSpec, value) for value in raw.get("models", [])],
        capture=_construct(CaptureConfig, raw.get("capture")),
        automaton=_construct(PredictiveAutomatonConfig, raw.get("automaton")),
        panel=_construct(PredictivePanelConfig, panel_raw),
        teacher=_construct(PredictiveTeacherConfig, raw.get("teacher")),
        probe=_construct(PredictiveProbeConfig, raw.get("probe")),
        gates=_construct(PredictiveGateConfig, raw.get("gates")),
        logging=_construct(LoggingConfig, logging_raw),
        authorization=_construct(
            PredictiveAuthorizationConfig, raw.get("authorization")
        ),
    )
    config.validate()
    return config
