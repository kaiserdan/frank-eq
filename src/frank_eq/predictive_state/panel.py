"""Fresh development panels and renderer-shift contracts for PSR0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .automaton import HistoryRecord, PredictiveAutomaton, PredictiveBasis, PredictiveTest

_RENDERERS = ("narrative", "table", "symbolic")
_ROLE_OFFSETS = {"train": 1_000_000, "validation": 2_000_000}


@dataclass(frozen=True, slots=True)
class PredictivePanel:
    role: str
    histories: tuple[HistoryRecord, ...]
    lengths: tuple[int, ...]
    histories_per_length: int
    seed: int
    renderers: tuple[str, ...] = _RENDERERS
    schema: str = "frank_eq_predictive_panel_v1"

    def validate(self) -> None:
        if self.role not in _ROLE_OFFSETS:
            raise ValueError("predictive panel role must be train or validation")
        if self.renderers != _RENDERERS:
            raise ValueError("predictive panel renderer registry changed")
        expected = len(self.lengths) * self.histories_per_length
        if len(self.histories) != expected:
            raise ValueError("predictive panel has an unexpected history count")
        if len({history.history_id for history in self.histories}) != expected:
            raise ValueError("predictive panel history IDs must be unique")
        observed_lengths = sorted({history.length for history in self.histories})
        if observed_lengths != sorted(self.lengths):
            raise ValueError("predictive panel does not cover the registered lengths")
        if any(history.role != self.role for history in self.histories):
            raise ValueError("predictive panel contains a history from another role")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "role": self.role,
            "lengths": list(self.lengths),
            "histories_per_length": self.histories_per_length,
            "seed": self.seed,
            "renderers": list(self.renderers),
            "histories": [history.to_dict() for history in self.histories],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PredictivePanel:
        if payload.get("schema") != "frank_eq_predictive_panel_v1":
            raise ValueError("unsupported predictive-panel schema")
        panel = cls(
            role=str(payload["role"]),
            histories=tuple(HistoryRecord.from_dict(row) for row in payload["histories"]),
            lengths=tuple(int(value) for value in payload["lengths"]),
            histories_per_length=int(payload["histories_per_length"]),
            seed=int(payload["seed"]),
            renderers=tuple(str(value) for value in payload["renderers"]),
        )
        panel.validate()
        return panel


def generate_predictive_panel(
    automaton: PredictiveAutomaton,
    basis: PredictiveBasis,
    *,
    role: str,
    lengths: Iterable[int],
    histories_per_length: int,
    seed: int,
    min_entropy: float,
    max_entropy: float,
    min_core_variance: float,
    max_attempt_multiplier: int,
) -> PredictivePanel:
    """Sample role-separated histories while rejecting degenerate posterior states."""

    if role not in _ROLE_OFFSETS:
        raise ValueError("predictive panel role must be train or validation")
    length_registry = tuple(sorted(set(int(value) for value in lengths)))
    if not length_registry or any(value < 1 for value in length_registry):
        raise ValueError("predictive panel lengths must be positive")
    if histories_per_length < 8:
        raise ValueError("predictive panel needs at least eight histories per length")
    if not 0.0 <= min_entropy < max_entropy <= np.log(automaton.n_states) + 1e-12:
        raise ValueError("predictive panel entropy interval is invalid")
    if min_core_variance < 0:
        raise ValueError("minimum core variance must be non-negative")

    rng = np.random.default_rng(seed)
    rows: list[HistoryRecord] = []
    public_id = _ROLE_OFFSETS[role]
    for length in length_registry:
        accepted = 0
        attempts = 0
        limit = histories_per_length * max_attempt_multiplier
        while accepted < histories_per_length and attempts < limit:
            attempts += 1
            actions, observations, posterior = automaton.sample_history(length=length, rng=rng)
            entropy = automaton.entropy(posterior)
            core = posterior @ basis.core_matrix
            if not min_entropy <= entropy <= max_entropy:
                continue
            if float(np.var(core)) < min_core_variance:
                continue
            history_id = public_id + length * 10_000 + accepted
            rows.append(
                automaton.make_history_record(
                    history_id=history_id,
                    role=role,
                    actions=actions,
                    observations=observations,
                    basis=basis,
                )
            )
            accepted += 1
        if accepted != histories_per_length:
            raise RuntimeError(
                f"could not generate {histories_per_length} non-degenerate {role} histories "
                f"at length {length}; accepted {accepted} after {attempts} attempts"
            )

    panel = PredictivePanel(
        role=role,
        histories=tuple(rows),
        lengths=length_registry,
        histories_per_length=histories_per_length,
        seed=seed,
    )
    panel.validate()
    return panel


def _probability_rows(values: np.ndarray, row_names: tuple[str, ...], column_names: tuple[str, ...]) -> list[str]:
    rows: list[str] = []
    for row_name, row in zip(row_names, values, strict=True):
        cells = ", ".join(
            f"{column_name}={float(value):.6f}"
            for column_name, value in zip(column_names, row, strict=True)
        )
        rows.append(f"{row_name}: {cells}")
    return rows


def render_predictive_prefix(
    automaton: PredictiveAutomaton,
    history: HistoryRecord,
    renderer: str | int,
) -> str:
    """Render the same controlled-HMM history through two fit and one unseen grammar."""

    renderer_name = _RENDERERS[int(renderer)] if isinstance(renderer, int) else str(renderer)
    if renderer_name not in _RENDERERS:
        raise ValueError(f"unsupported predictive-state renderer: {renderer!r}")
    states = automaton.state_names
    actions = automaton.action_names
    observations = automaton.observation_names

    if renderer_name == "narrative":
        lines = [
            "A hidden machine occupies one of four internal states. The current state is never shown.",
            "Initially the four hidden states are equally likely. Each chosen action first changes "
            "the hidden state according to the registered transition probabilities, then the sensor "
            "emits one observation according to the registered emission probabilities.",
            "No future test has been selected yet. Preserve the posterior information needed for "
            "unknown future action-observation tests.",
            "",
            "Transition model:",
        ]
        for action_index, action_name in enumerate(actions):
            lines.append(f"Action {action_name}:")
            lines.extend(
                f"  from {state_name}, next-state probabilities are "
                + ", ".join(
                    f"{target_name} {float(probability):.6f}"
                    for target_name, probability in zip(
                        states, automaton.transitions[action_index, state_index], strict=True
                    )
                )
                for state_index, state_name in enumerate(states)
            )
        lines.extend(["", "Sensor model:"])
        lines.extend("  " + row for row in _probability_rows(automaton.emissions, states, observations))
        lines.extend(["", "Observed history:"])
        for step, (action, observation) in enumerate(
            zip(history.actions, history.observations, strict=True), start=1
        ):
            lines.append(
                f"  Step {step}: action {actions[action]} was applied; the sensor reported "
                f"{observations[observation]}."
            )
        return "\n".join(lines) + "\n"

    if renderer_name == "table":
        lines = [
            "HIDDEN-MACHINE FILTERING TASK",
            "prior=" + ",".join(f"{state}:0.250000" for state in states),
            "future_test=UNSELECTED",
            "TRANSITIONS",
        ]
        for action_index, action_name in enumerate(actions):
            lines.append(f"[{action_name}]")
            lines.extend(_probability_rows(automaton.transitions[action_index], states, states))
        lines.append("EMISSIONS")
        lines.extend(_probability_rows(automaton.emissions, states, observations))
        lines.append("HISTORY step | action | observation")
        lines.extend(
            f"{step:02d} | {actions[action]} | {observations[observation]}"
            for step, (action, observation) in enumerate(
                zip(history.actions, history.observations, strict=True), start=1
            )
        )
        return "\n".join(lines) + "\n"

    transition_blocks = []
    for action_index, action_name in enumerate(actions):
        matrix = ";".join(
            ",".join(f"{float(value):.6f}" for value in row)
            for row in automaton.transitions[action_index]
        )
        transition_blocks.append(f"T[{action_name}]=[{matrix}]")
    emission = ";".join(
        ",".join(f"{float(value):.6f}" for value in row) for row in automaton.emissions
    )
    events = ";".join(
        f"({actions[action]},{observations[observation]})"
        for action, observation in zip(history.actions, history.observations, strict=True)
    )
    return (
        "PSR0|latent="
        + ",".join(states)
        + "|obs="
        + ",".join(observations)
        + "|prior=(0.25,0.25,0.25,0.25)|query=NONE\n"
        + "\n".join(transition_blocks)
        + f"\nE=[{emission}]\nH=[{events}]\n"
    )


def render_future_test_query(
    automaton: PredictiveAutomaton,
    test: PredictiveTest,
    *,
    false_display: str,
    true_display: str,
    sequence_cue: str,
) -> str:
    """Reveal a future event only after the query-blind prefix state exists."""

    action_text = " then ".join(automaton.action_names[action] for action in test.actions)
    observation = automaton.observation_names[test.observation]
    return (
        "\nRegistered future test: starting from the posterior after the observed history, "
        f"apply action sequence {action_text}. A fresh run is sampled from the listed stochastic "
        f"machine. The terminal event is that the sensor reports {observation}. "
        "Evaluate the probability of this event using the stored history. "
        f"Reply with exactly {false_display.strip()} for the event being false or "
        f"{true_display.strip()} for the event being true."
        f"{sequence_cue}"
    )
