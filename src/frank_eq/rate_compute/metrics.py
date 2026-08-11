"""World-grouped metrics for source readout and downstream compute."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from frank_eq.evaluation.bootstrap import bootstrap_statistic

from .calibration import (
    aggregate_by_world,
    balanced_accuracy,
    brier_score,
    expected_calibration_error,
)
from .config import RateComputeRunConfig
from .logic import HARD_COMPOSITION_FAMILIES


def interval(values: np.ndarray, *, seed: int, replicates: int) -> dict[str, Any]:
    return bootstrap_statistic(values, replicates=replicates, seed=seed).to_dict()


def world_gain_interval(
    rows: list[dict[str, Any]],
    *,
    candidate_field: str,
    baseline_field: str,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("cannot compute a gain interval for an empty row group")
    truth = np.asarray([row["truth"] for row in rows], dtype=np.float64)
    candidate = np.asarray([row[candidate_field] for row in rows], dtype=np.float64)
    baseline = np.asarray([row[baseline_field] for row in rows], dtype=np.float64)
    gains = (truth - baseline) ** 2 - (truth - candidate) ** 2
    _, world_gains = aggregate_by_world(
        gains, np.asarray([row["world_id"] for row in rows])
    )
    return interval(world_gains, seed=seed, replicates=replicates)


def summarize(rows: list[dict[str, Any]], config: RateComputeRunConfig) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("cannot summarize an empty row group")
    truth = np.asarray([row["truth"] for row in rows], dtype=np.float64)
    raw = np.asarray([row["probability_true"] for row in rows], dtype=np.float64)
    calibrated = np.asarray(
        [row["calibrated_probability"] for row in rows], dtype=np.float64
    )
    prior = np.asarray([row["prior_probability"] for row in rows], dtype=np.float64)
    return {
        "rows": len(rows),
        "worlds": len({int(row["world_id"]) for row in rows}),
        "mean_generated_tokens": float(
            np.mean([int(row.get("generated_token_count", 0)) for row in rows])
        ),
        "mean_false_candidate_tokens": float(
            np.mean([int(row.get("false_token_count", 0)) for row in rows])
        ),
        "mean_true_candidate_tokens": float(
            np.mean([int(row.get("true_token_count", 0)) for row in rows])
        ),
        "raw_brier": brier_score(truth, raw),
        "calibrated_brier": brier_score(truth, calibrated),
        "prior_brier": brier_score(truth, prior),
        "calibrated_balanced_accuracy": balanced_accuracy(truth, calibrated),
        "calibrated_ece": expected_calibration_error(
            truth, calibrated, bins=config.evaluation.ece_bins
        ),
    }


def paired_protocol_interval(
    rows: list[dict[str, Any]],
    *,
    baseline_protocol: str,
    candidate_protocol: str,
    families: set[str] | frozenset[str] | None,
    seed: int,
    replicates: int,
) -> dict[str, Any] | None:
    selected = [
        row
        for row in rows
        if row["split"] == "validation"
        and row["kind"] == "target"
        and row["protocol"] in {baseline_protocol, candidate_protocol}
        and (families is None or row["family"] in families)
    ]
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        key = (
            row["world_id"],
            row["model_id"],
            row["renderer_id"],
            row["operation_id"],
        )
        by_key[key][str(row["protocol"])] = row
    paired: list[tuple[int, float]] = []
    for conditions in by_key.values():
        if baseline_protocol not in conditions or candidate_protocol not in conditions:
            continue
        baseline = conditions[baseline_protocol]
        candidate = conditions[candidate_protocol]
        truth = float(candidate["truth"])
        improvement = (truth - float(baseline["calibrated_probability"])) ** 2 - (
            truth - float(candidate["calibrated_probability"])
        ) ** 2
        paired.append((int(candidate["world_id"]), improvement))
    if not paired:
        return None
    world_ids = np.asarray([item[0] for item in paired], dtype=np.int64)
    values = np.asarray([item[1] for item in paired], dtype=np.float64)
    _, world_values = aggregate_by_world(values, world_ids)
    return interval(world_values, seed=seed, replicates=replicates)


def evaluate_source_protocols(
    records: list[dict[str, Any]],
    config: RateComputeRunConfig,
) -> tuple[dict[str, Any], list[bool], dict[str, Any] | None, dict[str, Any] | None]:
    """Evaluate basis readout, response channels, compute, and operation sparsity."""

    validation = [row for row in records if row["split"] == "validation"]
    seed = config.evaluation.bootstrap_seed
    replicates = config.evaluation.bootstrap_replicates
    metrics: dict[str, Any] = {
        "basis": {},
        "direct": {},
        "operation_instance_diagnostics": {},
    }
    basis_checks: list[bool] = []

    for model_offset, model in enumerate(config.models):
        metrics["basis"][model.model_id] = {}
        metrics["direct"][model.model_id] = {}
        metrics["operation_instance_diagnostics"][model.model_id] = {}
        for complexity_offset, n_entities in enumerate(config.panel.entity_counts):
            basis_rows = [
                row
                for row in validation
                if row["kind"] == "basis"
                and row["model_id"] == model.model_id
                and row["entity_count"] == n_entities
                and row["protocol"] == config.protocols.basis_protocol
            ]
            basis_summary = summarize(basis_rows, config)
            basis_gain = world_gain_interval(
                basis_rows,
                candidate_field="calibrated_probability",
                baseline_field="prior_probability",
                seed=seed + 100 * model_offset + complexity_offset,
                replicates=replicates,
            )
            basis_passed = (
                float(basis_gain["lower"])
                >= config.gates.min_basis_brier_gain_lower95
                and basis_summary["calibrated_balanced_accuracy"]
                >= config.gates.min_basis_balanced_accuracy
            )
            basis_checks.append(basis_passed)
            metrics["basis"][model.model_id][str(n_entities)] = {
                **basis_summary,
                "coordinates": n_entities * (n_entities - 1),
                "minimum_exact_binary_rate_bits": n_entities * (n_entities - 1),
                "brier_gain_over_prior_ci": basis_gain,
                "passed": basis_passed,
            }

            direct_complexity: dict[str, Any] = {}
            instance_complexity: dict[str, Any] = {}
            families = sorted(
                {row["family"] for row in validation if row["kind"] == "target"}
            )
            for protocol_offset, protocol in enumerate(config.protocols.target_protocols):
                direct_complexity[protocol] = {}
                instance_complexity[protocol] = {}
                for family_offset, family in enumerate(families):
                    rows = [
                        row
                        for row in validation
                        if row["kind"] == "target"
                        and row["model_id"] == model.model_id
                        and row["entity_count"] == n_entities
                        and row["protocol"] == protocol
                        and row["family"] == family
                    ]
                    if not rows:
                        continue
                    payload: dict[str, Any] = {
                        **summarize(rows, config),
                        "brier_gain_over_prior_ci": world_gain_interval(
                            rows,
                            candidate_field="calibrated_probability",
                            baseline_field="prior_probability",
                            seed=(
                                seed
                                + 10_000
                                + 1000 * model_offset
                                + 100 * complexity_offset
                                + 10 * protocol_offset
                                + family_offset
                            ),
                            replicates=replicates,
                        ),
                        "by_polarity": {},
                    }
                    for polarity_offset, (label, sign) in enumerate(
                        (("positive", 1), ("negative", -1))
                    ):
                        polarity_rows = [
                            row
                            for row in rows
                            if (1 if float(row["polarity"]) >= 0 else -1) == sign
                        ]
                        if polarity_rows:
                            payload["by_polarity"][label] = {
                                **summarize(polarity_rows, config),
                                "brier_gain_over_prior_ci": world_gain_interval(
                                    polarity_rows,
                                    candidate_field="calibrated_probability",
                                    baseline_field="prior_probability",
                                    seed=(
                                        seed
                                        + 15_000
                                        + 1000 * model_offset
                                        + 100 * complexity_offset
                                        + 10 * protocol_offset
                                        + family_offset
                                        + 50_000 * polarity_offset
                                    ),
                                    replicates=replicates,
                                ),
                            }
                    direct_complexity[protocol][family] = payload

                operation_ids = sorted(
                    {
                        int(row["operation_id"])
                        for row in validation
                        if row["kind"] == "target"
                        and row["model_id"] == model.model_id
                        and row["entity_count"] == n_entities
                        and row["protocol"] == protocol
                    }
                )
                for operation_id in operation_ids:
                    rows = [
                        row
                        for row in validation
                        if row["kind"] == "target"
                        and row["model_id"] == model.model_id
                        and row["entity_count"] == n_entities
                        and row["protocol"] == protocol
                        and row["operation_id"] == operation_id
                    ]
                    instance_complexity[protocol][str(operation_id)] = {
                        "family": rows[0]["family"],
                        "polarity": rows[0]["polarity"],
                        "structural_support_size": rows[0]["structural_support_size"],
                        **summarize(rows, config),
                        "brier_gain_over_prior_ci": world_gain_interval(
                            rows,
                            candidate_field="calibrated_probability",
                            baseline_field="prior_probability",
                            seed=(
                                seed
                                + 70_000
                                + 1000 * model_offset
                                + 100 * complexity_offset
                                + 10 * protocol_offset
                                + operation_id
                            ),
                            replicates=replicates,
                        ),
                    }
            metrics["direct"][model.model_id][str(n_entities)] = direct_complexity
            metrics["operation_instance_diagnostics"][model.model_id][
                str(n_entities)
            ] = instance_complexity

    answer_channel = paired_protocol_interval(
        records,
        baseline_protocol="answer_token",
        candidate_protocol="sequence",
        families=None,
        seed=seed + 20_000,
        replicates=replicates,
    )
    reason_pause = paired_protocol_interval(
        records,
        baseline_protocol="pause",
        candidate_protocol="reason",
        families=HARD_COMPOSITION_FAMILIES,
        seed=seed + 20_001,
        replicates=replicates,
    )
    reason_sequence = paired_protocol_interval(
        records,
        baseline_protocol="sequence",
        candidate_protocol="reason",
        families=HARD_COMPOSITION_FAMILIES,
        seed=seed + 20_002,
        replicates=replicates,
    )
    metrics["contrasts"] = {
        "sequence_over_answer_token_brier_gain_ci": answer_channel,
        "reason_over_pause_hard_family_brier_gain_ci": reason_pause,
        "reason_over_sequence_hard_family_brier_gain_ci": reason_sequence,
    }

    support_metrics: dict[str, Any] = {}
    support_sizes = sorted(
        {
            int(row["structural_support_size"])
            for row in validation
            if row["kind"] == "target"
        }
    )
    for protocol_offset, protocol in enumerate(config.protocols.target_protocols):
        support_metrics[protocol] = {}
        for support_offset, support_size in enumerate(support_sizes):
            rows = [
                row
                for row in validation
                if row["kind"] == "target"
                and row["protocol"] == protocol
                and row["structural_support_size"] == support_size
            ]
            if rows:
                support_metrics[protocol][str(support_size)] = {
                    **summarize(rows, config),
                    "brier_gain_over_prior_ci": world_gain_interval(
                        rows,
                        candidate_field="calibrated_probability",
                        baseline_field="prior_probability",
                        seed=seed + 25_000 + 100 * protocol_offset + support_offset,
                        replicates=replicates,
                    ),
                }
    metrics["by_structural_support_size"] = support_metrics
    return metrics, basis_checks, answer_channel, reason_pause
