"""Public-basis composition metrics and development decision reducer."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from frank_eq.data.real_panel import RealPanel

from .calibration import aggregate_by_world, brier_score
from .compile import compile_validation_records
from .config import RateComputeRunConfig
from .logic import HARD_COMPOSITION_FAMILIES
from .metrics import evaluate_source_protocols, interval


def _compiled_group_summary(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    config: RateComputeRunConfig,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("compiled metric group is empty")
    replicates = config.evaluation.bootstrap_replicates
    truth = np.asarray([row["truth"] for row in rows], dtype=np.float64)
    direct = np.asarray([row["direct_probability"] for row in rows], dtype=np.float64)
    prior = np.asarray([row["prior_probability"] for row in rows], dtype=np.float64)
    compiled = np.asarray([row["compiled_probability"] for row in rows], dtype=np.float64)
    worlds = np.asarray([row["world_id"] for row in rows], dtype=np.int64)

    _, direct_world = aggregate_by_world(
        (truth - direct) ** 2 - (truth - compiled) ** 2, worlds
    )
    _, prior_world = aggregate_by_world(
        (truth - prior) ** 2 - (truth - compiled) ** 2, worlds
    )
    direct_ci = interval(direct_world, seed=seed, replicates=replicates)
    prior_ci = interval(prior_world, seed=seed + 1, replicates=replicates)

    quantized: dict[str, Any] = {}
    for offset, bits in enumerate(config.evaluation.basis_quantization_bits):
        field = f"compiled_probability_q{bits}"
        prediction = np.asarray([row[field] for row in rows], dtype=np.float64)
        _, q_direct_world = aggregate_by_world(
            (truth - direct) ** 2 - (truth - prediction) ** 2, worlds
        )
        _, q_prior_world = aggregate_by_world(
            (truth - prior) ** 2 - (truth - prediction) ** 2, worlds
        )
        rates = np.asarray(
            [row[f"basis_rate_bits_q{bits}"] for row in rows], dtype=np.float64
        )
        quantized[str(bits)] = {
            "bits_per_coordinate": bits,
            "mean_message_bits": float(np.mean(rates)),
            "min_message_bits": int(np.min(rates)),
            "max_message_bits": int(np.max(rates)),
            "compiled_brier": brier_score(truth, prediction),
            "compiled_over_direct_brier_gain_ci": interval(
                q_direct_world, seed=seed + 100 + 2 * offset, replicates=replicates
            ),
            "compiled_over_prior_brier_gain_ci": interval(
                q_prior_world, seed=seed + 101 + 2 * offset, replicates=replicates
            ),
        }

    oracle = np.asarray(
        [row["oracle_compiled_probability"] for row in rows], dtype=np.float64
    )
    return {
        "rows": len(rows),
        "worlds": len({int(row["world_id"]) for row in rows}),
        "mean_direct_generated_tokens": float(
            np.mean([int(row["direct_generated_tokens"]) for row in rows])
        ),
        "mean_direct_source_queries": float(
            np.mean([int(row["direct_source_queries"]) for row in rows])
        ),
        "mean_basis_source_queries": float(
            np.mean([int(row["basis_source_queries"]) for row in rows])
        ),
        "compiled_over_direct_brier_gain_ci": direct_ci,
        "compiled_over_prior_brier_gain_ci": prior_ci,
        "compiled_brier": brier_score(truth, compiled),
        "direct_brier": brier_score(truth, direct),
        "prior_brier": brier_score(truth, prior),
        "oracle_executor_brier": brier_score(truth, oracle),
        "compiled_over_direct_passed": (
            float(direct_ci["lower"])
            > config.gates.min_compiled_direct_gain_lower95
        ),
        "compiled_over_prior_passed": (
            float(prior_ci["lower"])
            >= config.gates.min_compiled_prior_gain_lower95
        ),
        "quantized_rate_frontier": quantized,
    }


def _stratified_compiled_summary(
    rows: list[dict[str, Any]],
    *,
    labels: list[tuple[str, Callable[[dict[str, Any]], bool]]],
    seed: int,
    config: RateComputeRunConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, (label, predicate) in enumerate(labels):
        selected = [row for row in rows if predicate(row)]
        if selected:
            result[label] = _compiled_group_summary(
                selected, seed=seed + offset, config=config
            )
    return result


def evaluate_rate_compute(
    records: list[dict[str, Any]],
    panels: dict[int, RealPanel],
    config: RateComputeRunConfig,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Combine source-protocol and compiled-basis evidence into one decision."""

    source_metrics, basis_checks, answer_channel, reason_pause = evaluate_source_protocols(
        records, config
    )
    compiled, direct_selection = compile_validation_records(records, panels, config)
    hard = [row for row in compiled if row["family"] in HARD_COMPOSITION_FAMILIES]
    if not hard:
        raise RuntimeError("rate--compute audit produced no hard-family compiled rows")

    seed = config.evaluation.bootstrap_seed
    aggregate = _compiled_group_summary(hard, seed=seed + 30_000, config=config)
    by_model: dict[str, Any] = {}
    group_checks: list[bool] = []
    for model_offset, model in enumerate(config.models):
        by_model[model.model_id] = {}
        for complexity_offset, n_entities in enumerate(config.panel.entity_counts):
            rows = [
                row
                for row in hard
                if row["model_id"] == model.model_id
                and row["entity_count"] == n_entities
            ]
            summary = _compiled_group_summary(
                rows,
                seed=seed + 31_000 + 100 * model_offset + complexity_offset,
                config=config,
            )
            group_checks.append(
                bool(summary["compiled_over_direct_passed"])
                and bool(summary["compiled_over_prior_passed"])
            )
            by_model[model.model_id][str(n_entities)] = summary

    by_family = _stratified_compiled_summary(
        hard,
        labels=[
            (family, lambda row, family=family: row["family"] == family)
            for family in sorted(HARD_COMPOSITION_FAMILIES)
        ],
        seed=seed + 35_000,
        config=config,
    )
    by_polarity = _stratified_compiled_summary(
        hard,
        labels=[
            ("positive", lambda row: float(row["polarity"]) >= 0),
            ("negative", lambda row: float(row["polarity"]) < 0),
        ],
        seed=seed + 36_000,
        config=config,
    )

    source_metrics["schema"] = "frank_eq_rate_compute_metrics_v1"
    source_metrics["scope"] = (
        "development-only response-channel, compute, and public-basis audit"
    )
    source_metrics["data_usage"] = {
        "test_worlds_used": 0,
        "held_sender_used": False,
        "claim_bearing_role": False,
        "models": [model.model_id for model in config.models],
        "entity_counts": config.panel.entity_counts,
        "worlds_per_complexity": config.panel.worlds_per_complexity,
    }
    source_metrics["compiled_basis"] = {
        "rows": len(compiled),
        "hard_family_rows": len(hard),
        "hard_families": sorted(HARD_COMPOSITION_FAMILIES),
        "direct_protocol_selection": direct_selection,
        "aggregate": aggregate,
        "by_model_and_complexity": by_model,
        "by_family": by_family,
        "by_polarity": by_polarity,
        "compiled_over_direct_brier_gain_ci": aggregate[
            "compiled_over_direct_brier_gain_ci"
        ],
        "compiled_over_prior_brier_gain_ci": aggregate[
            "compiled_over_prior_brier_gain_ci"
        ],
    }

    basis_passed = bool(basis_checks) and all(basis_checks)
    compiled_prior_passed = (
        bool(group_checks)
        and all(group_checks)
        and bool(aggregate["compiled_over_prior_passed"])
    )
    compiled_direct_passed = (
        bool(group_checks)
        and all(group_checks)
        and bool(aggregate["compiled_over_direct_passed"])
    )
    answer_channel_passed = bool(answer_channel) and (
        float(answer_channel["lower"])
        > config.gates.min_answer_channel_gain_lower95
    )
    compute_passed = bool(reason_pause) and (
        float(reason_pause["lower"])
        > config.gates.min_reason_over_pause_gain_lower95
    )
    passed = basis_passed and compiled_prior_passed and compiled_direct_passed
    if not basis_passed:
        diagnosis = "BASIS_READOUT_NOT_QUALIFIED"
    elif not compiled_prior_passed:
        diagnosis = "PUBLIC_BASIS_NOT_SUFFICIENT"
    elif not compiled_direct_passed:
        diagnosis = "NO_COMPOSITION_ADVANTAGE_OVER_TRAIN_SELECTED_DIRECT_BASELINE"
    else:
        diagnosis = "PUBLIC_BASIS_COMPOSITION_SUPPORTED"

    direct_ci = aggregate["compiled_over_direct_brier_gain_ci"]
    prior_ci = aggregate["compiled_over_prior_brier_gain_ci"]
    decision = {
        "schema": "frank_eq_rate_compute_decision_v1",
        "status": "pass" if passed else "fail",
        "decision": (
            "OPERATIONAL_BASIS_CANDIDATE_FOR_STAGEA_REGISTRATION"
            if passed
            else "STOP_BEFORE_STAGEA_V3"
        ),
        "diagnosis": diagnosis,
        "checks": {
            "basis_contract": {"passed": basis_passed},
            "compiled_over_prior": {
                "observed": prior_ci["lower"],
                "required": (
                    f"lower_95 >= {config.gates.min_compiled_prior_gain_lower95}"
                ),
                "passed": compiled_prior_passed,
            },
            "compiled_over_train_selected_direct": {
                "observed": direct_ci["lower"],
                "required": (
                    f"lower_95 > {config.gates.min_compiled_direct_gain_lower95}"
                ),
                "passed": compiled_direct_passed,
            },
            "answer_channel_effect": {
                "observed": None if answer_channel is None else answer_channel["lower"],
                "passed": answer_channel_passed,
                "diagnostic_only": True,
            },
            "contentful_compute_effect": {
                "observed": None if reason_pause is None else reason_pause["lower"],
                "passed": compute_passed,
                "diagnostic_only": True,
            },
        },
        "authorization": {
            "stagea_registration_draft_authorized": passed,
            "stagea_outcome_run_authorized": False,
            "claim_bearing_test_access_authorized": False,
            "receiver_execution_authorized": False,
            "scientific_claim_authorized": False,
        },
    }
    return source_metrics, decision, compiled, direct_selection
