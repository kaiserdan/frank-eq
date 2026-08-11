"""Hash-bound configuration loader for the frozen Stage-A v3-1 registration."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from frank_eq.utils import sha256_file

_TOP_LEVEL_KEYS = {
    "schema",
    "protocol_version",
    "run_name",
    "output_dir",
    "require_revision_pins",
    "models",
    "access",
    "panel",
    "capture",
    "teacher_protocol",
    "compiler",
    "training",
    "baselines",
    "packet",
    "evaluation",
    "gates",
    "authorization",
    "logging",
}
_MODEL_REVISIONS = {
    "qwen3-4b": ("Qwen/Qwen3-4B", "founder", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "founder", "b968826d9c46dd6066d109eabc6255188de91218"),
    "qwen3-14b-held": (
        "Qwen/Qwen3-14B",
        "held",
        "40c069824f4251a91eefaf281ebe4c544efd3e18",
    ),
}
_PROTECTED_AUTHORIZATION_FIELDS = (
    "receiver_execution_authorized",
    "new_receiver_world_access_authorized",
    "scientific_claim_authorized",
    "paper_claim_authorized",
)


@dataclass(frozen=True, slots=True)
class StageAV3ModelSpec:
    model_id: str
    hf_id: str
    role: str
    revision: str
    task_exposure: str


@dataclass(frozen=True, slots=True)
class StageAV3Config:
    """Validated immutable view over the machine-readable registration."""

    source_path: Path
    payload: dict[str, Any]
    models: tuple[StageAV3ModelSpec, ...]

    @property
    def protocol_version(self) -> str:
        return str(self.payload["protocol_version"])

    @property
    def run_name(self) -> str:
        return str(self.payload["run_name"])

    @property
    def output_dir(self) -> str:
        return str(self.payload["output_dir"])

    @property
    def founder_models(self) -> tuple[StageAV3ModelSpec, ...]:
        return tuple(model for model in self.models if model.role == "founder")

    @property
    def held_model(self) -> StageAV3ModelSpec:
        return next(model for model in self.models if model.role == "held")

    @property
    def config_sha256(self) -> str:
        return sha256_file(self.source_path)

    def section(self, name: str) -> dict[str, Any]:
        value = self.payload[name]
        if not isinstance(value, dict):
            raise TypeError(f"Stage-A v3 section {name!r} is not a mapping")
        return copy.deepcopy(value)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} must equal frozen value {expected!r}; got {observed!r}")


def _validate_payload(payload: dict[str, Any]) -> tuple[StageAV3ModelSpec, ...]:
    unknown = set(payload) - _TOP_LEVEL_KEYS
    missing = _TOP_LEVEL_KEYS - set(payload)
    if unknown or missing:
        raise ValueError(
            f"Stage-A v3 top-level keys differ; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    _require_equal(payload["schema"], "frank_eq_stagea_v3_registration_v1", "schema")
    _require_equal(payload["protocol_version"], "stagea-v3-1", "protocol_version")
    _require_equal(payload["require_revision_pins"], True, "require_revision_pins")

    rows = payload.get("models")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("Stage-A v3 requires exactly two founders and one held sender")
    models = tuple(StageAV3ModelSpec(**row) for row in rows)
    observed = {
        model.model_id: (model.hf_id, model.role, model.revision) for model in models
    }
    if observed != _MODEL_REVISIONS:
        raise ValueError("Stage-A v3 model identities, roles, or revisions differ from registration")
    if models[-1].role != "held" or models[-1].task_exposure != "unopened":
        raise ValueError("the final Stage-A v3 model must be the unopened held sender")

    access = payload["access"]
    _require_equal(
        access["allowed_stage_sequence"],
        ["prepare", "founder_fit", "freeze", "held_onboard", "evaluate"],
        "access.allowed_stage_sequence",
    )
    _require_equal(access["test_creation_after_freeze"], True, "test_creation_after_freeze")
    _require_equal(access["test_access_count"], 1, "test_access_count")
    _require_equal(access["receiver_access"], False, "receiver_access")

    panel = payload["panel"]
    _require_equal(panel["entity_counts"], [4, 6], "panel.entity_counts")
    _require_equal(panel["operation_seed"], 2026081213, "panel.operation_seed")
    _require_equal(panel["n_target_operations"], 32, "panel.n_target_operations")
    expected_roles = {
        "train": {"worlds_per_complexity": 80, "seed": 2026081201},
        "validation": {"worlds_per_complexity": 24, "seed": 2026081202},
        "test": {"worlds_per_complexity": 32, "seed": 2026081203},
    }
    _require_equal(panel["roles"], expected_roles, "panel.roles")
    _require_equal(panel["renderers"]["fit"], ["natural", "adjacency"], "fit renderers")
    _require_equal(
        panel["renderers"]["test_unseen"],
        ["canonical_edge_list"],
        "unseen renderer",
    )

    capture = payload["capture"]
    _require_equal(capture["prompt_format"], "chat_turn", "capture.prompt_format")
    _require_equal(capture["normalized_depths"], [0.25, 0.5, 0.75, 1.0], "capture depths")
    _require_equal(capture["sequence_scope"], "all_unpadded_prefix_tokens", "sequence scope")
    _require_equal(capture["branch_mode"], "kv_reuse", "capture.branch_mode")
    _require_equal(capture["allow_exact_replay_fallback"], False, "replay fallback")
    if int(capture["branch_batch_size"]) < 1:
        raise ValueError("capture.branch_batch_size must be positive")

    teacher = payload["teacher_protocol"]
    _require_equal(teacher["basis_protocol"], "sequence", "teacher basis protocol")
    _require_equal(teacher["direct_protocols"], ["sequence", "reason", "pause"], "direct protocols")
    if teacher["rationale_budget"] != teacher["pause_budget"] or teacher["pause_budget"] != 32:
        raise ValueError("reason and pause budgets must both remain 32")

    compiler = payload["compiler"]
    _require_equal(compiler["channels"], ["semantic", "behavioral"], "compiler channels")
    _require_equal(compiler["share_trainable_parameters_across_channels"], False, "channel sharing")
    _require_equal(compiler["share_trainable_parameters_across_models"], False, "model sharing")
    _require_equal(compiler["seeds"], [211, 223, 227], "compiler seeds")
    if compiler["model_dim"] % compiler["attention_heads"]:
        raise ValueError("compiler.model_dim must be divisible by attention_heads")

    required_baselines = {
        "train_edge_prior",
        "token_id_resampler",
        "final_token_public_mlp",
        "historical_continuous_quotient",
        "train_selected_direct_protocol",
        "interactive_basis",
        "deterministic_text_parser",
        "rate_matched_canonical_text",
        "oracle_basis",
        "shuffled_world_packet",
        "wrong_world_packet",
        "zero_packet",
    }
    if set(payload["baselines"]["required"]) != required_baselines:
        raise ValueError("Stage-A v3 baseline registry changed")

    packet = payload["packet"]
    _require_equal(packet["primary_quantization_bits"], 4, "primary packet bits")
    _require_equal(packet["quantization_frontier_bits"], [1, 2, 4, 8], "rate frontier")

    evaluation = payload["evaluation"]
    if int(evaluation["bootstrap_replicates"]) != 2000 or not evaluation["world_grouped"]:
        raise ValueError("Stage-A v3 requires 2,000 world-grouped bootstrap replicates")

    authorization = payload["authorization"]
    if not authorization[
        "one_representation_run_authorized_after_protocol_and_implementation_commits"
    ]:
        raise ValueError("Stage-A v3 representation run is not authorized")
    if any(authorization[field] for field in _PROTECTED_AUTHORIZATION_FIELDS):
        raise ValueError("Stage-A v3 registration opens a protected authorization")
    return models


def _verify_registration(source: Path, registration_path: Path | None) -> None:
    manifest_path = registration_path or source.with_name("registration.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage-A v3 registration manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "frank_eq_stagea_v3_registration_manifest_v1":
        raise ValueError("Stage-A v3 registration manifest has the wrong schema")
    expected = manifest.get("files", {}).get("configs/stagea_v3/real_olivia_v3.yaml")
    if expected is None or sha256_file(source) != expected:
        raise ValueError("Stage-A v3 config does not match its frozen registration hash")


def load_stagea_v3_config(
    path: str | Path,
    *,
    verify_registration: bool = True,
    registration_path: str | Path | None = None,
) -> StageAV3Config:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Stage-A v3 config not found: {source}")
    if verify_registration:
        _verify_registration(
            source,
            None if registration_path is None else Path(registration_path).resolve(),
        )
    payload = yaml.safe_load(source.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("Stage-A v3 config must contain a mapping")
    models = _validate_payload(payload)
    return StageAV3Config(source_path=source, payload=payload, models=models)
