"""Frozen configuration for the development-only SPQ0 census."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from frank_eq.real_config import LoggingConfig, WandBLoggingConfig

from .automaton import (
    ControlledSystem,
    SharedPredictiveBasis,
    build_shared_predictive_basis,
    generate_system_family,
)

_ACTIVE_MODELS = {
    "qwen3-4b": (
        "Qwen/Qwen3-4B",
        "qwen3",
        "1cfa9a7208912126459214e8b04321603b3df60c",
    ),
    "mistral-7b-v03": (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral",
        "c170c708c41dac9275d15a8fff4eca08d52bab71",
    ),
}
_RESERVED_MODELS = {
    "olmo2-7b-held": (
        "allenai/OLMo-2-1124-7B-Instruct",
        "olmo2",
        "470b1fba1ae01581f270116362ee4aa1b97f4c84",
    ),
    "granite31-8b-held": (
        "ibm-granite/granite-3.1-8b-instruct",
        "granite",
        "4009206d5fc95d2e65a7b7633e159d6e97e25d35",
    ),
}
_REQUIRED_SURFACES = {
    "final_token_residual",
    "event_boundary_residuals",
    "all_token_summary",
    "mean_input_embedding",
    "parameter_matched_token_sequence",
}
_REQUIRED_CONTROLS = {
    "history_prior",
    "last_observation_filter",
    "empirical_observation_filter",
    "deterministic_token_hash_ridge",
    "parameter_matched_token_sequence_encoder",
    "mean_input_embedding",
    "final_token_residual",
    "direct_probability_forecast",
    "exact_bayes_filter",
    "exact_public_core",
    "overcomplete_test_bank",
    "shuffled_history",
    "wrong_history",
    "renderer_shuffled",
    "zero_packet",
}


@dataclass(slots=True)
class SPQModelSpec:
    model_id: str
    hf_id: str
    revision: str
    role: str
    family: str
    tokenizer_id: str | None = None
    trust_remote_code: bool = False


@dataclass(slots=True)
class ReservedCheckpointSpec:
    model_id: str
    hf_id: str
    revision: str
    family: str
    access: str = "reserved_unopened"


@dataclass(slots=True)
class SPQSystemConfig:
    latent_states: int = 4
    actions: int = 3
    observations: int = 3
    fit_systems: int = 2
    validation_only_systems: int = 1
    full_support_min_probability: float = 0.04
    predictive_rank: int = 4
    core_condition_number_max: float = 5.0
    target_executor_l1_max: float = 4.0
    future_horizons: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    target_tests: int = 24
    system_seed: int = 2026084158
    core_selection_seed: int = 2026084102


@dataclass(slots=True)
class SPQPanelRoleConfig:
    lengths: list[int] = field(default_factory=list)
    histories_per_system_length: int = 64
    seed: int = 0


@dataclass(slots=True)
class SPQPanelRolesConfig:
    calibration: SPQPanelRoleConfig = field(
        default_factory=lambda: SPQPanelRoleConfig([8, 16], 96, 2026084201)
    )
    selection: SPQPanelRoleConfig = field(
        default_factory=lambda: SPQPanelRoleConfig([8, 16], 48, 2026084202)
    )
    validation: SPQPanelRoleConfig = field(
        default_factory=lambda: SPQPanelRoleConfig([8, 16, 32], 64, 2026084203)
    )


@dataclass(slots=True)
class SPQPanelConfig:
    roles: SPQPanelRolesConfig = field(default_factory=SPQPanelRolesConfig)
    fit_renderers: list[str] = field(default_factory=lambda: ["narrative", "table"])
    validation_renderers: list[str] = field(
        default_factory=lambda: ["narrative", "table", "symbolic"]
    )
    validation_only_length: int = 32
    min_belief_entropy: float = 0.15
    max_belief_entropy: float = 1.35
    min_core_variance: float = 0.0025
    max_generation_attempt_multiplier: int = 200


@dataclass(slots=True)
class SPQCaptureConfig:
    prompt_format: str = "chat_turn"
    chat_template_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"enable_thinking": False}
    )
    normalized_depths: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    surfaces: list[str] = field(default_factory=lambda: sorted(_REQUIRED_SURFACES))
    selected_kv_surface_enabled: bool = False
    dtype: str = "bfloat16"
    serialized_dtype: str = "float32"
    device: str = "cuda"
    branch_mode: str = "kv_reuse"
    allow_exact_replay_fallback: bool = False
    branch_batch_size: int = 8
    local_files_only: bool = True
    max_length: int = 4096


@dataclass(slots=True)
class SPQProbabilityProtocolConfig:
    type: str = "categorical_bins"
    bins: list[float] = field(
        default_factory=lambda: [0.05 + 0.10 * index for index in range(10)]
    )
    candidate_labels: list[str] = field(
        default_factory=lambda: [f" {letter}" for letter in "ABCDEFGHIJ"]
    )
    normalize_candidate_log_likelihoods: bool = True
    temperature_grid: list[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0]
    )
    candidate_temperature_fit_role: str = "calibration"
    direct_protocol_selection_role: str = "selection"


@dataclass(slots=True)
class SPQSemanticEncoderConfig:
    methods: list[str] = field(default_factory=lambda: ["ridge", "reduced_rank_regression"])
    rank_grid: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8])
    ridge_grid: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    surface_selection_role: str = "selection"
    target: str = "rank_conditioned_public_tests"
    token_hash_position_period: int = 64
    token_sequence_decay_grid: list[float] = field(
        default_factory=lambda: [0.50, 0.75, 0.90, 0.97]
    )


@dataclass(slots=True)
class SPQTargetReaderConfig:
    input: str = "exact_public_core"
    output: str = "model_probability_bin_signature"
    fit_role: str = "calibration"
    selection_role: str = "selection"
    ridge_grid: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    pair_specific_parameters: bool = False
    freeze_before_source_evaluation: bool = True


@dataclass(slots=True)
class SPQBehavioralResidualConfig:
    enabled: bool = True
    method: str = "maxvar_gcca"
    rank_grid: list[int] = field(default_factory=lambda: [0, 1, 2, 4])
    fit_role: str = "calibration"
    selection_role: str = "selection"
    promotional: bool = False


@dataclass(slots=True)
class SPQEvaluationConfig:
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 2026084291
    independent_unit: str = "system_history"
    quantization_bits: list[int] = field(default_factory=lambda: [2, 4, 8])
    amortized_future_query_counts: list[int] = field(default_factory=lambda: [1, 4, 16, 32])
    required_conditions: list[str] = field(
        default_factory=lambda: [
            "seen",
            "unseen_renderer",
            "unseen_system",
            "length_transfer",
            "joint_ood",
        ]
    )
    source_query_cost_brier_equivalent: float = 0.02
    packet_bit_cost_brier_equivalent: float = 0.0005


@dataclass(slots=True)
class SPQGateConfig:
    max_oracle_executor_abs_error: float = 1e-10
    source_probability_gain_over_prior_lower95_min: float = 0.0
    semantic_core_gain_over_prior_lower95_min: float = 0.0
    activation_over_token_sequence_lower95_strict_gt: float = 0.0
    wrong_history_margin_lower95_strict_gt: float = 0.0
    cross_family_target_prior_gain_lower95_strict_gt: float = 0.0
    min_cross_family_oracle_reader_gain_retention: float = 0.70
    rank4_noninferior_to_higher_ranks: bool = True
    min_four_bit_gain_retention: float = 0.95
    max_sender_identity_accuracy_over_chance: float = 0.15
    amortized_query_count_for_primary_utility: int = 16
    amortized_utility_lower95_strict_gt: float = 0.0


@dataclass(slots=True)
class SPQAuthorizationConfig:
    development_audit_authorized: bool = True
    held_sender_access_authorized: bool = False
    claim_bearing_test_access_authorized: bool = False
    receiver_execution_authorized: bool = False
    scientific_claim_authorized: bool = False
    paper_claim_authorized: bool = False
    pass_authorizes_only_spq1_draft: bool = True


@dataclass(slots=True)
class SPQRunConfig:
    schema: str = "frank_eq_shared_predictive_quotient_spq0_v1"
    protocol_version: str = "spq0-development-census"
    run_name: str = "frank-eq-spq0-olivia"
    output_dir: str = "runs/spq0-olivia"
    development_only: bool = True
    require_revision_pins: bool = True
    models: list[SPQModelSpec] = field(default_factory=list)
    reserved_unopened_models: list[ReservedCheckpointSpec] = field(default_factory=list)
    systems: SPQSystemConfig = field(default_factory=SPQSystemConfig)
    panel: SPQPanelConfig = field(default_factory=SPQPanelConfig)
    capture: SPQCaptureConfig = field(default_factory=SPQCaptureConfig)
    probability_protocol: SPQProbabilityProtocolConfig = field(
        default_factory=SPQProbabilityProtocolConfig
    )
    semantic_encoder: SPQSemanticEncoderConfig = field(
        default_factory=SPQSemanticEncoderConfig
    )
    strong_controls: list[str] = field(default_factory=lambda: sorted(_REQUIRED_CONTROLS))
    target_local_reader: SPQTargetReaderConfig = field(
        default_factory=SPQTargetReaderConfig
    )
    behavioral_residual: SPQBehavioralResidualConfig = field(
        default_factory=SPQBehavioralResidualConfig
    )
    evaluation: SPQEvaluationConfig = field(default_factory=SPQEvaluationConfig)
    gates: SPQGateConfig = field(default_factory=SPQGateConfig)
    authorization: SPQAuthorizationConfig = field(default_factory=SPQAuthorizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def build_systems_and_basis(
        self,
    ) -> tuple[tuple[ControlledSystem, ...], SharedPredictiveBasis]:
        systems = generate_system_family(
            latent_states=self.systems.latent_states,
            actions=self.systems.actions,
            observations=self.systems.observations,
            fit_systems=self.systems.fit_systems,
            validation_only_systems=self.systems.validation_only_systems,
            minimum_probability=self.systems.full_support_min_probability,
            seed=self.systems.system_seed,
        )
        basis = build_shared_predictive_basis(
            systems,
            horizons=self.systems.future_horizons,
            exact_rank=self.systems.predictive_rank,
            maximum_rank=max(self.semantic_encoder.rank_grid),
            n_target_tests=self.systems.target_tests,
            target_seed=self.systems.core_selection_seed,
            max_core_condition_number=self.systems.core_condition_number_max,
            max_target_l1=self.systems.target_executor_l1_max,
        )
        return systems, basis

    def validate(self) -> None:
        if self.schema != "frank_eq_shared_predictive_quotient_spq0_v1":
            raise ValueError("unsupported SPQ0 registration schema")
        if self.protocol_version != "spq0-development-census" or not self.development_only:
            raise ValueError("SPQ0 must remain the frozen development-only census")
        observed_active = {
            model.model_id: (model.hf_id, model.family, model.revision)
            for model in self.models
        }
        if observed_active != _ACTIVE_MODELS:
            raise ValueError("SPQ0 active founder families or exact revisions changed")
        if any(model.role != "founder" for model in self.models):
            raise ValueError("SPQ0 may load development founders only")
        if len({model.family for model in self.models}) != len(self.models):
            raise ValueError("SPQ0 requires founders from distinct model families")
        if self.require_revision_pins and any(len(model.revision) != 40 for model in self.models):
            raise ValueError("every SPQ0 founder requires an exact 40-hex revision")
        observed_reserved = {
            model.model_id: (model.hf_id, model.family, model.revision)
            for model in self.reserved_unopened_models
        }
        if observed_reserved != _RESERVED_MODELS:
            raise ValueError("SPQ0 reserved checkpoint registry changed")
        if any(model.access != "reserved_unopened" for model in self.reserved_unopened_models):
            raise ValueError("reserved checkpoints must remain explicitly unopened")
        if {model.hf_id for model in self.models} & {
            model.hf_id for model in self.reserved_unopened_models
        }:
            raise ValueError("active and reserved checkpoint registries overlap")

        systems, basis = self.build_systems_and_basis()
        if len(systems) != self.systems.fit_systems + self.systems.validation_only_systems:
            raise ValueError("generated system-family size changed")
        if sum(system.role == "validation_only" for system in systems) < 1:
            raise ValueError("SPQ0 requires a validation-only transition/emission system")
        if basis.exact_rank != 4 or basis.maximum_rank < 8:
            raise ValueError("SPQ0 requires an exact rank-4 core and overcomplete sweep")
        if max(basis.core_condition_numbers.values()) > self.systems.core_condition_number_max:
            raise ValueError("shared public core exceeds the registered condition bound")
        if basis.maximum_target_l1 > self.systems.target_executor_l1_max:
            raise ValueError("shared target executor exceeds the registered L1 bound")

        roles = self.panel.roles
        if set(roles.calibration.lengths) != {8, 16}:
            raise ValueError("SPQ0 calibration lengths must be 8 and 16")
        if set(roles.selection.lengths) != {8, 16}:
            raise ValueError("SPQ0 selection lengths must be 8 and 16")
        if set(roles.validation.lengths) != {8, 16, 32}:
            raise ValueError("SPQ0 validation lengths must be 8, 16, and 32")
        if self.panel.validation_only_length != 32:
            raise ValueError("SPQ0 freezes length 32 as validation-only")
        role_seeds = [roles.calibration.seed, roles.selection.seed, roles.validation.seed]
        if len(set(role_seeds)) != 3:
            raise ValueError("SPQ0 role seeds must be disjoint")
        for role in (roles.calibration, roles.selection, roles.validation):
            if role.histories_per_system_length < 32:
                raise ValueError("SPQ0 needs at least 32 histories per system and length")
        if set(self.panel.fit_renderers) != {"narrative", "table"}:
            raise ValueError("SPQ0 fit renderers must be narrative and table")
        if set(self.panel.validation_renderers) != {"narrative", "table", "symbolic"}:
            raise ValueError("SPQ0 validation must add the unseen symbolic renderer")
        if not 0.0 <= self.panel.min_belief_entropy < self.panel.max_belief_entropy:
            raise ValueError("SPQ0 belief-entropy interval is invalid")
        if self.panel.max_belief_entropy > np.log(self.systems.latent_states) + 1e-12:
            raise ValueError("SPQ0 maximum entropy exceeds log(number of states)")
        if self.panel.min_core_variance < 0 or self.panel.max_generation_attempt_multiplier < 1:
            raise ValueError("SPQ0 panel rejection settings are invalid")

        capture = self.capture
        if capture.prompt_format != "chat_turn" or capture.branch_mode != "kv_reuse":
            raise ValueError("SPQ0 requires corrected chat_turn and literal KV reuse")
        if capture.allow_exact_replay_fallback or not capture.local_files_only:
            raise ValueError("SPQ0 forbids replay and online checkpoint resolution")
        if capture.selected_kv_surface_enabled:
            raise ValueError("SPQ0 registers no cross-architecture-safe KV summary surface")
        if set(capture.surfaces) != _REQUIRED_SURFACES:
            raise ValueError("SPQ0 capture-surface census changed")
        if capture.serialized_dtype != "float32" or capture.max_length < 4096:
            raise ValueError("SPQ0 capture serialization or context budget changed")
        if len(capture.normalized_depths) != 4 or any(
            not 0.0 < depth <= 1.0 for depth in capture.normalized_depths
        ):
            raise ValueError("SPQ0 requires four valid normalized capture depths")
        if capture.branch_batch_size < 1:
            raise ValueError("SPQ0 categorical branch batch size must be positive")

        protocol = self.probability_protocol
        expected_bins = np.asarray([0.05 + 0.10 * index for index in range(10)])
        if protocol.type != "categorical_bins" or not np.allclose(
            protocol.bins, expected_bins, atol=1e-12, rtol=0.0
        ):
            raise ValueError("SPQ0 probability bins changed")
        if (
            len(protocol.candidate_labels) != len(protocol.bins)
            or len(set(protocol.candidate_labels)) != len(protocol.candidate_labels)
            or any(not label.strip() for label in protocol.candidate_labels)
        ):
            raise ValueError("SPQ0 categorical candidate labels are invalid")
        if not protocol.normalize_candidate_log_likelihoods:
            raise ValueError("SPQ0 must length-normalize candidate likelihoods")
        if protocol.candidate_temperature_fit_role != "calibration":
            raise ValueError("SPQ0 candidate temperature must be calibration-only")
        if protocol.direct_protocol_selection_role != "selection":
            raise ValueError("SPQ0 direct protocol selection role changed")
        if not protocol.temperature_grid or any(value <= 0 for value in protocol.temperature_grid):
            raise ValueError("SPQ0 temperature grid must be positive")

        encoder = self.semantic_encoder
        if set(encoder.methods) != {"ridge", "reduced_rank_regression"}:
            raise ValueError("SPQ0 encoder-method census changed")
        if encoder.rank_grid != [1, 2, 3, 4, 6, 8]:
            raise ValueError("SPQ0 bottleneck-rank sweep changed")
        if encoder.surface_selection_role != "selection":
            raise ValueError("SPQ0 architecture selection must use only the selection role")
        if encoder.target != "rank_conditioned_public_tests":
            raise ValueError("SPQ0 semantic encoder target changed")
        if not encoder.ridge_grid or any(value <= 0 for value in encoder.ridge_grid):
            raise ValueError("SPQ0 ridge grid is invalid")
        if set(self.strong_controls) != _REQUIRED_CONTROLS:
            raise ValueError("SPQ0 mandatory control registry changed")

        reader = self.target_local_reader
        if (
            reader.input != "exact_public_core"
            or reader.output != "model_probability_bin_signature"
            or reader.fit_role != "calibration"
            or reader.selection_role != "selection"
            or reader.pair_specific_parameters
            or not reader.freeze_before_source_evaluation
        ):
            raise ValueError("SPQ0 target-local reader freeze contract changed")
        residual = self.behavioral_residual
        if (
            not residual.enabled
            or residual.method != "maxvar_gcca"
            or residual.rank_grid != [0, 1, 2, 4]
            or residual.promotional
        ):
            raise ValueError("SPQ0 non-promotional behavioral-residual census changed")

        evaluation = self.evaluation
        if evaluation.bootstrap_replicates < 2000:
            raise ValueError("SPQ0 requires at least 2,000 grouped bootstrap replicates")
        if evaluation.independent_unit != "system_history":
            raise ValueError("SPQ0 independent unit must be system/history")
        if evaluation.quantization_bits != [2, 4, 8]:
            raise ValueError("SPQ0 quantization census changed")
        if evaluation.amortized_future_query_counts != [1, 4, 16, 32]:
            raise ValueError("SPQ0 amortized query frontier changed")
        required_conditions = {
            "seen",
            "unseen_renderer",
            "unseen_system",
            "length_transfer",
            "joint_ood",
        }
        if set(evaluation.required_conditions) != required_conditions:
            raise ValueError("SPQ0 transfer-condition registry changed")
        if (
            evaluation.source_query_cost_brier_equivalent <= 0
            or evaluation.packet_bit_cost_brier_equivalent <= 0
        ):
            raise ValueError("SPQ0 rate-aware utility exchange rates must be positive")

        gates = self.gates
        if gates.amortized_query_count_for_primary_utility != 16:
            raise ValueError("SPQ0 primary amortization boundary must remain 16 queries")
        if not 0.0 <= gates.min_cross_family_oracle_reader_gain_retention <= 1.0:
            raise ValueError("SPQ0 reader-retention gate is invalid")
        if not 0.0 <= gates.min_four_bit_gain_retention <= 1.0:
            raise ValueError("SPQ0 four-bit retention gate is invalid")
        if gates.max_oracle_executor_abs_error <= 0:
            raise ValueError("SPQ0 exact-executor tolerance must be positive")

        authorization = self.authorization
        if not authorization.development_audit_authorized:
            raise ValueError("SPQ0 development audit requires explicit authority")
        if not authorization.pass_authorizes_only_spq1_draft:
            raise ValueError("an SPQ0 pass may authorize only drafting SPQ1")
        if any(
            (
                authorization.held_sender_access_authorized,
                authorization.claim_bearing_test_access_authorized,
                authorization.receiver_execution_authorized,
                authorization.scientific_claim_authorized,
                authorization.paper_claim_authorized,
            )
        ):
            raise ValueError("SPQ0 protected authorization fields must remain false")
        self.logging.validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[Any], payload: dict[str, Any] | None) -> Any:
    values = dict(payload or {})
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_spq0_config(path: str | Path) -> SPQRunConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SPQ0 configuration not found: {source}")
    raw = yaml.safe_load(source.read_text()) or {}
    allowed = set(SPQRunConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown SPQ0 top-level keys: {sorted(unknown)}")
    defaults = SPQRunConfig()

    panel_raw = dict(raw.get("panel") or {})
    roles_raw = dict(panel_raw.get("roles") or {})
    roles_defaults = SPQPanelRolesConfig()
    panel_raw["roles"] = SPQPanelRolesConfig(
        calibration=_construct(
            SPQPanelRoleConfig,
            roles_raw.get("calibration") or asdict(roles_defaults.calibration),
        ),
        selection=_construct(
            SPQPanelRoleConfig,
            roles_raw.get("selection") or asdict(roles_defaults.selection),
        ),
        validation=_construct(
            SPQPanelRoleConfig,
            roles_raw.get("validation") or asdict(roles_defaults.validation),
        ),
    )
    logging_raw = dict(raw.get("logging") or {})
    if "wandb" in logging_raw:
        logging_raw["wandb"] = _construct(WandBLoggingConfig, logging_raw["wandb"])

    config = SPQRunConfig(
        schema=raw.get("schema", defaults.schema),
        protocol_version=raw.get("protocol_version", defaults.protocol_version),
        run_name=raw.get("run_name", defaults.run_name),
        output_dir=raw.get("output_dir", defaults.output_dir),
        development_only=raw.get("development_only", defaults.development_only),
        require_revision_pins=raw.get(
            "require_revision_pins", defaults.require_revision_pins
        ),
        models=[_construct(SPQModelSpec, item) for item in raw.get("models", [])],
        reserved_unopened_models=[
            _construct(ReservedCheckpointSpec, item)
            for item in raw.get("reserved_unopened_models", [])
        ],
        systems=_construct(SPQSystemConfig, raw.get("systems")),
        panel=_construct(SPQPanelConfig, panel_raw),
        capture=_construct(SPQCaptureConfig, raw.get("capture")),
        probability_protocol=_construct(
            SPQProbabilityProtocolConfig, raw.get("probability_protocol")
        ),
        semantic_encoder=_construct(
            SPQSemanticEncoderConfig, raw.get("semantic_encoder")
        ),
        strong_controls=list(raw.get("strong_controls", defaults.strong_controls)),
        target_local_reader=_construct(
            SPQTargetReaderConfig, raw.get("target_local_reader")
        ),
        behavioral_residual=_construct(
            SPQBehavioralResidualConfig, raw.get("behavioral_residual")
        ),
        evaluation=_construct(SPQEvaluationConfig, raw.get("evaluation")),
        gates=_construct(SPQGateConfig, raw.get("gates")),
        authorization=_construct(
            SPQAuthorizationConfig, raw.get("authorization")
        ),
        logging=_construct(LoggingConfig, logging_raw),
    )
    config.validate()
    return config
