"""Role-separated stochastic histories and paired SPQ0 renderer views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .automaton import (
    ControlledSystem,
    HistoryRecord,
    PredictiveTest,
    SharedPredictiveBasis,
    make_history_record,
)
from .config import SPQRunConfig

RENDERERS = ("narrative", "table", "symbolic")
ROLE_IDS = {"calibration": 0, "selection": 1, "validation": 2}
RENDERER_IDS = {name: index for index, name in enumerate(RENDERERS)}
SYSTEM_ROLE_IDS = {"fit": 0, "validation_only": 1}
_ROLE_OFFSETS = {
    "calibration": 1_000_000_000,
    "selection": 2_000_000_000,
    "validation": 3_000_000_000,
}


@dataclass(frozen=True, slots=True)
class SPQPanel:
    role: str
    histories: tuple[HistoryRecord, ...]
    lengths: tuple[int, ...]
    histories_per_system_length: int
    seed: int
    system_ids: tuple[str, ...]
    renderers: tuple[str, ...]
    schema: str = "frank_eq_spq0_panel_v1"

    def validate(self) -> None:
        if self.role not in ROLE_IDS:
            raise ValueError("SPQ0 panel role must be calibration, selection, or validation")
        if len(set(self.system_ids)) != len(self.system_ids) or not self.system_ids:
            raise ValueError("SPQ0 panel system registry must be non-empty and unique")
        if any(renderer not in RENDERER_IDS for renderer in self.renderers):
            raise ValueError("SPQ0 panel contains an unknown renderer")
        expected = len(self.system_ids) * len(self.lengths) * self.histories_per_system_length
        if len(self.histories) != expected:
            raise ValueError("SPQ0 panel has an unexpected history count")
        if len({history.history_id for history in self.histories}) != expected:
            raise ValueError("SPQ0 history IDs must be unique")
        if any(history.role != self.role for history in self.histories):
            raise ValueError("SPQ0 panel contains a history from another role")
        if {history.system_id for history in self.histories} != set(self.system_ids):
            raise ValueError("SPQ0 panel system coverage differs from its registry")
        if {history.length for history in self.histories} != set(self.lengths):
            raise ValueError("SPQ0 panel length coverage differs from its registry")
        if self.role != "validation" and any(
            history.system_role != "fit" for history in self.histories
        ):
            raise ValueError("validation-only systems leaked into fitting roles")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "role": self.role,
            "lengths": list(self.lengths),
            "histories_per_system_length": self.histories_per_system_length,
            "seed": self.seed,
            "system_ids": list(self.system_ids),
            "renderers": list(self.renderers),
            "histories": [history.to_dict() for history in self.histories],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SPQPanel:
        if payload.get("schema") != "frank_eq_spq0_panel_v1":
            raise ValueError("unsupported SPQ0 panel schema")
        panel = cls(
            role=str(payload["role"]),
            histories=tuple(HistoryRecord.from_dict(row) for row in payload["histories"]),
            lengths=tuple(int(value) for value in payload["lengths"]),
            histories_per_system_length=int(payload["histories_per_system_length"]),
            seed=int(payload["seed"]),
            system_ids=tuple(str(value) for value in payload["system_ids"]),
            renderers=tuple(str(value) for value in payload["renderers"]),
        )
        panel.validate()
        return panel


@dataclass(frozen=True, slots=True)
class RenderedPrefix:
    text: str
    event_end_markers: tuple[str, ...]

    def validate(self, expected_events: int) -> None:
        if len(self.event_end_markers) != expected_events:
            raise ValueError("rendered prefix has the wrong event-boundary count")
        cursor = 0
        for marker in self.event_end_markers:
            position = self.text.find(marker, cursor)
            if position < 0 or self.text.find(marker, position + 1) >= 0:
                raise ValueError("event-boundary marker is missing or non-unique")
            cursor = position + len(marker)


def generate_panel(
    systems: tuple[ControlledSystem, ...],
    basis: SharedPredictiveBasis,
    *,
    role: str,
    lengths: Iterable[int],
    histories_per_system_length: int,
    seed: int,
    renderers: Iterable[str],
    min_entropy: float,
    max_entropy: float,
    min_core_variance: float,
    max_attempt_multiplier: int,
) -> SPQPanel:
    if role not in ROLE_IDS:
        raise ValueError("SPQ0 panel role is invalid")
    eligible = tuple(system for system in systems if role == "validation" or system.role == "fit")
    if not eligible:
        raise ValueError("SPQ0 panel has no eligible systems")
    length_values = tuple(sorted(set(int(value) for value in lengths)))
    renderer_values = tuple(str(value) for value in renderers)
    if not length_values or any(length < 1 for length in length_values):
        raise ValueError("SPQ0 history lengths must be positive")
    if histories_per_system_length < 8:
        raise ValueError("SPQ0 needs at least eight histories per system/length")
    if any(renderer not in RENDERER_IDS for renderer in renderer_values):
        raise ValueError("SPQ0 panel contains an unsupported renderer")
    rng = np.random.default_rng(seed)
    rows: list[HistoryRecord] = []
    for system_index, system in enumerate(eligible):
        for length in length_values:
            accepted = 0
            attempts = 0
            limit = histories_per_system_length * max_attempt_multiplier
            while accepted < histories_per_system_length and attempts < limit:
                attempts += 1
                actions, observations, posterior = system.sample_history(length=length, rng=rng)
                entropy = system.entropy(posterior)
                core = basis.public_probabilities(
                    system.system_id,
                    posterior,
                    rank=basis.exact_rank,
                )
                if not min_entropy <= entropy <= max_entropy:
                    continue
                if float(np.var(core)) < min_core_variance:
                    continue
                history_id = (
                    _ROLE_OFFSETS[role] + system_index * 10_000_000 + length * 10_000 + accepted
                )
                rows.append(
                    make_history_record(
                        system,
                        basis,
                        history_id=history_id,
                        role=role,
                        actions=actions,
                        observations=observations,
                    )
                )
                accepted += 1
            if accepted != histories_per_system_length:
                raise RuntimeError(
                    f"could not generate {histories_per_system_length} {role} histories "
                    f"for {system.system_id} at length {length}; accepted {accepted} "
                    f"after {attempts} attempts"
                )
    panel = SPQPanel(
        role=role,
        histories=tuple(rows),
        lengths=length_values,
        histories_per_system_length=histories_per_system_length,
        seed=seed,
        system_ids=tuple(system.system_id for system in eligible),
        renderers=renderer_values,
    )
    panel.validate()
    return panel


def build_panels(
    config: SPQRunConfig,
    systems: tuple[ControlledSystem, ...],
    basis: SharedPredictiveBasis,
) -> dict[str, SPQPanel]:
    panel = config.panel
    result: dict[str, SPQPanel] = {}
    for role, role_config in (
        ("calibration", panel.roles.calibration),
        ("selection", panel.roles.selection),
        ("validation", panel.roles.validation),
    ):
        result[role] = generate_panel(
            systems,
            basis,
            role=role,
            lengths=role_config.lengths,
            histories_per_system_length=role_config.histories_per_system_length,
            seed=role_config.seed,
            renderers=(panel.validation_renderers if role == "validation" else panel.fit_renderers),
            min_entropy=panel.min_belief_entropy,
            max_entropy=panel.max_belief_entropy,
            min_core_variance=panel.min_core_variance,
            max_attempt_multiplier=panel.max_generation_attempt_multiplier,
        )
    role_sets = [{history.history_id for history in result[role].histories} for role in ROLE_IDS]
    if any(role_sets[left] & role_sets[right] for left in range(3) for right in range(left)):
        raise RuntimeError("SPQ0 role history IDs overlap")
    return result


def _number(value: float) -> str:
    return format(float(value), ".17g")


def _probability_rows(
    values: np.ndarray,
    row_names: tuple[str, ...],
    column_names: tuple[str, ...],
) -> list[str]:
    rows: list[str] = []
    for row_name, row in zip(row_names, values, strict=True):
        cells = ", ".join(
            f"{column_name}={_number(value)}"
            for column_name, value in zip(column_names, row, strict=True)
        )
        rows.append(f"{row_name}: {cells}")
    return rows


def render_prefix(
    system: ControlledSystem,
    history: HistoryRecord,
    renderer: str,
) -> RenderedPrefix:
    """Render one query-blind system/history with uniquely locatable event ends."""

    if renderer not in RENDERER_IDS:
        raise ValueError(f"unsupported SPQ0 renderer: {renderer!r}")
    states = system.state_names
    actions = system.action_names
    observations = system.observation_names
    markers: list[str] = []

    if renderer == "narrative":
        lines = [
            "A controlled stochastic machine has four hidden states. Its hidden state is never shown.",
            "At each recorded step, the named action changes the hidden state and the sensor then "
            "emits one observation. Preserve the current probability state for future tests that "
            "have not yet been selected.",
            "No future action sequence, terminal observation, probability bin, or answer candidate "
            "has been revealed.",
            "future_test=UNSELECTED; probability_bin=UNSELECTED; candidate=UNSELECTED",
            "",
            "Initial hidden-state probabilities: "
            + ", ".join(
                f"{state}={_number(value)}"
                for state, value in zip(states, system.initial_belief, strict=True)
            ),
            "Transition model:",
        ]
        for action_index, action_name in enumerate(actions):
            lines.append(f"Action {action_name}:")
            lines.extend(
                "  from "
                + state_name
                + ", next-state probabilities are "
                + ", ".join(
                    f"{target_name}={_number(probability)}"
                    for target_name, probability in zip(
                        states,
                        system.transitions[action_index, state_index],
                        strict=True,
                    )
                )
                for state_index, state_name in enumerate(states)
            )
        lines.append("Sensor model:")
        lines.extend(
            "  " + row for row in _probability_rows(system.emissions, states, observations)
        )
        lines.append("Observed history:")
        for step, (action, observation) in enumerate(
            zip(history.actions, history.observations, strict=True), start=1
        ):
            marker = (
                f"Step {step:02d}: action {actions[action]} was applied and the sensor reported "
                f"{observations[observation]}."
            )
            markers.append(marker)
            lines.append(marker)
        rendered = RenderedPrefix("\n".join(lines) + "\n", tuple(markers))
        rendered.validate(history.length)
        return rendered

    if renderer == "table":
        lines = [
            "CONTROLLED STOCHASTIC FILTER",
            "future_test=UNSELECTED; probability_bin=UNSELECTED; candidate=UNSELECTED",
            "prior="
            + ",".join(
                f"{state}:{_number(value)}"
                for state, value in zip(states, system.initial_belief, strict=True)
            ),
            "TRANSITIONS",
        ]
        for action_index, action_name in enumerate(actions):
            lines.append(f"[{action_name}]")
            lines.extend(_probability_rows(system.transitions[action_index], states, states))
        lines.append("EMISSIONS")
        lines.extend(_probability_rows(system.emissions, states, observations))
        lines.append("HISTORY step | action | observation")
        for step, (action, observation) in enumerate(
            zip(history.actions, history.observations, strict=True), start=1
        ):
            marker = f"{step:02d} | {actions[action]} | {observations[observation]}"
            markers.append(marker)
            lines.append(marker)
        rendered = RenderedPrefix("\n".join(lines) + "\n", tuple(markers))
        rendered.validate(history.length)
        return rendered

    transition_blocks = []
    for action_index, action_name in enumerate(actions):
        matrix = ";".join(
            ",".join(_number(value) for value in row) for row in system.transitions[action_index]
        )
        transition_blocks.append(f"T[{action_name}]=[{matrix}]")
    emission = ";".join(",".join(_number(value) for value in row) for row in system.emissions)
    event_rows = []
    for step, (action, observation) in enumerate(
        zip(history.actions, history.observations, strict=True), start=1
    ):
        marker = f"e{step:02d}=({actions[action]},{observations[observation]})"
        markers.append(marker)
        event_rows.append(marker)
    text = (
        "SPQ0|latent="
        + ",".join(states)
        + "|obs="
        + ",".join(observations)
        + "|prior="
        + ",".join(_number(value) for value in system.initial_belief)
        + "|future=UNSELECTED|bin=UNSELECTED|candidate=UNSELECTED\n"
        + "\n".join(transition_blocks)
        + f"\nE=[{emission}]\nH=["
        + ";".join(event_rows)
        + "]\n"
    )
    rendered = RenderedPrefix(text, tuple(markers))
    rendered.validate(history.length)
    return rendered


def render_probability_query(
    system: ControlledSystem,
    test: PredictiveTest,
    *,
    bins: Iterable[float],
    candidate_labels: Iterable[str],
) -> str:
    """Reveal one future test and its categorical bins only after capture."""

    actions = " then ".join(system.action_names[action] for action in test.actions)
    observation = system.observation_names[test.observation]
    choices = ", ".join(
        f"{label.strip()}={float(value):.2f}"
        for label, value in zip(candidate_labels, bins, strict=True)
    )
    return (
        "Forecast a probability for this registered future test. Starting from the filtered "
        f"state after the observed history, apply {actions}; the terminal sensor observation is "
        f"{observation}. Choose the nearest probability bin. Registered labels: {choices}. "
        "Reply with exactly one label and no explanation. Final bin label:"
    )
