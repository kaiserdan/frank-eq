#!/usr/bin/env python3
"""Independently recompute the frozen RC0 decision-critical statistics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frank_eq.rate_compute.config import load_rate_compute_config  # noqa: E402
from frank_eq.utils import atomic_write_json  # noqa: E402

HARD_FAMILIES = frozenset(
    {"mutual", "compose", "compare_outdegree", "counterfactual_add"}
)
COMPILED_FAMILIES = HARD_FAMILIES | {"lookup", "inverse"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _brier(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((truth - prediction) ** 2))


def _balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    labels = truth >= 0.5
    predicted = prediction >= 0.5
    recalls = [float(np.mean(predicted[labels == value] == value)) for value in (False, True)]
    return float(np.mean(recalls))


def _world_means(values: np.ndarray, world_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [values[world_ids == world].mean() for world in np.unique(world_ids)],
        dtype=np.float64,
    )


def _interval(values: np.ndarray, *, seed: int, replicates: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, array.size, size=array.size)
        estimates[index] = float(array[sample].mean())
    return {
        "estimate": float(array.mean()),
        "lower": float(np.quantile(estimates, 0.025)),
        "upper": float(np.quantile(estimates, 0.975)),
        "replicates": replicates,
    }


def _compare_interval(
    failures: list[str],
    label: str,
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for field in ("estimate", "lower", "upper"):
        if not np.isclose(
            float(observed[field]),
            float(expected[field]),
            rtol=0.0,
            atol=1e-12,
        ):
            failures.append(f"{label} {field} differs from independent recomputation")
    if int(observed["replicates"]) != int(expected["replicates"]):
        failures.append(f"{label} replicate count differs")


def _compiled_summary(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    truth = np.asarray([row["truth"] for row in rows], dtype=np.float64)
    direct = np.asarray([row["direct_probability"] for row in rows], dtype=np.float64)
    prior = np.asarray([row["prior_probability"] for row in rows], dtype=np.float64)
    compiled = np.asarray([row["compiled_probability"] for row in rows], dtype=np.float64)
    worlds = np.asarray([row["world_id"] for row in rows], dtype=np.int64)
    direct_gain = _world_means((truth - direct) ** 2 - (truth - compiled) ** 2, worlds)
    prior_gain = _world_means((truth - prior) ** 2 - (truth - compiled) ** 2, worlds)
    return {
        "rows": len(rows),
        "worlds": int(np.unique(worlds).size),
        "compiled_brier": _brier(truth, compiled),
        "direct_brier": _brier(truth, direct),
        "prior_brier": _brier(truth, prior),
        "compiled_over_direct_brier_gain_ci": _interval(
            direct_gain, seed=seed, replicates=replicates
        ),
        "compiled_over_prior_brier_gain_ci": _interval(
            prior_gain, seed=seed + 1, replicates=replicates
        ),
    }


def _paired_interval(
    records: list[dict[str, Any]],
    *,
    baseline: str,
    candidate: str,
    families: frozenset[str] | None,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        if (
            row["split"] != "validation"
            or row["kind"] != "target"
            or row["protocol"] not in {baseline, candidate}
            or (families is not None and row["family"] not in families)
        ):
            continue
        key = (
            row["world_id"],
            row["model_id"],
            row["renderer_id"],
            row["operation_id"],
        )
        pairs[key][row["protocol"]] = row
    values: list[float] = []
    worlds: list[int] = []
    for conditions in pairs.values():
        base = conditions[baseline]
        cand = conditions[candidate]
        truth = float(cand["truth"])
        values.append(
            (truth - float(base["calibrated_probability"])) ** 2
            - (truth - float(cand["calibrated_probability"])) ** 2
        )
        worlds.append(int(cand["world_id"]))
    grouped = _world_means(
        np.asarray(values, dtype=np.float64),
        np.asarray(worlds, dtype=np.int64),
    )
    return _interval(grouped, seed=seed, replicates=replicates)


def audit_run(root: str | Path) -> dict[str, Any]:
    source = Path(root)
    config = load_rate_compute_config(source / "config.yaml")
    records = _read_jsonl(source / "records_calibrated.jsonl")
    compiled = _read_jsonl(source / "compiled_predictions.jsonl")
    metrics = _read_json(source / "metrics.json")
    decision = _read_json(source / "decision.json")
    selection_artifact = _read_json(source / "direct_protocol_selection.json")
    calibration = _read_json(source / "calibration.json")
    failures: list[str] = []
    seed = config.evaluation.bootstrap_seed
    replicates = config.evaluation.bootstrap_replicates

    basis_summaries: dict[str, Any] = {}
    basis_passes: list[bool] = []
    for model_offset, model in enumerate(config.models):
        for complexity_offset, n_entities in enumerate(config.panel.entity_counts):
            rows = [
                row
                for row in records
                if row["split"] == "validation"
                and row["kind"] == "basis"
                and row["model_id"] == model.model_id
                and row["entity_count"] == n_entities
                and row["protocol"] == config.protocols.basis_protocol
            ]
            truth = np.asarray([row["truth"] for row in rows], dtype=np.float64)
            calibrated = np.asarray(
                [row["calibrated_probability"] for row in rows], dtype=np.float64
            )
            prior = np.asarray([row["prior_probability"] for row in rows], dtype=np.float64)
            worlds = np.asarray([row["world_id"] for row in rows], dtype=np.int64)
            gain = _world_means(
                (truth - prior) ** 2 - (truth - calibrated) ** 2,
                worlds,
            )
            interval = _interval(
                gain,
                seed=seed + 100 * model_offset + complexity_offset,
                replicates=replicates,
            )
            balanced = _balanced_accuracy(truth, calibrated)
            label = f"{model.model_id}|{n_entities}"
            reported = metrics["basis"][model.model_id][str(n_entities)]
            _compare_interval(
                failures,
                f"basis {label}",
                reported["brier_gain_over_prior_ci"],
                interval,
            )
            if not np.isclose(
                float(reported["calibrated_balanced_accuracy"]),
                balanced,
                rtol=0.0,
                atol=1e-12,
            ):
                failures.append(f"basis {label} balanced accuracy differs")
            passed = (
                interval["lower"] >= config.gates.min_basis_brier_gain_lower95
                and balanced >= config.gates.min_basis_balanced_accuracy
            )
            if bool(reported["passed"]) != passed:
                failures.append(f"basis {label} pass flag differs")
            basis_passes.append(passed)
            basis_summaries[label] = {
                "rows": len(rows),
                "worlds": int(np.unique(worlds).size),
                "calibrated_brier": _brier(truth, calibrated),
                "balanced_accuracy": balanced,
                "gain_over_prior_ci": interval,
                "passed": passed,
            }

    train_groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if (
            row["split"] == "train"
            and row["kind"] == "target"
            and row["family"] in COMPILED_FAMILIES
        ):
            train_groups[
                (
                    str(row["model_id"]),
                    int(row["entity_count"]),
                    str(row["family"]),
                    str(row["protocol"]),
                )
            ].append(row)
    selected: dict[tuple[str, int, str], tuple[str, float, int]] = {}
    selection_candidates: dict[tuple[str, int, str], list[tuple[float, str, int]]] = defaultdict(
        list
    )
    for (model_id, n_entities, family, protocol), rows in train_groups.items():
        truth = np.asarray([row["truth"] for row in rows], dtype=np.float64)
        prediction = np.asarray(
            [row["calibrated_probability"] for row in rows], dtype=np.float64
        )
        selection_candidates[(model_id, n_entities, family)].append(
            (_brier(truth, prediction), protocol, len(rows))
        )
    for key, candidates in selection_candidates.items():
        score, protocol, count = min(candidates, key=lambda item: (item[0], item[1]))
        selected[key] = (protocol, score, count)
        artifact = selection_artifact["groups"]["|".join(map(str, key))]
        if artifact["protocol"] != protocol or not np.isclose(
            float(artifact["train_brier"]), score, rtol=0.0, atol=1e-12
        ):
            failures.append(f"direct protocol selection differs for {key}")

    for row in compiled:
        key = (str(row["model_id"]), int(row["entity_count"]), str(row["family"]))
        if row["direct_protocol"] != selected[key][0]:
            failures.append(f"compiled row uses an unselected direct protocol for {key}")
            break

    hard = [row for row in compiled if row["family"] in HARD_FAMILIES]
    aggregate = _compiled_summary(
        hard,
        seed=seed + 30_000,
        replicates=replicates,
    )
    reported_aggregate = metrics["compiled_basis"]["aggregate"]
    _compare_interval(
        failures,
        "compiled aggregate/direct",
        reported_aggregate["compiled_over_direct_brier_gain_ci"],
        aggregate["compiled_over_direct_brier_gain_ci"],
    )
    _compare_interval(
        failures,
        "compiled aggregate/prior",
        reported_aggregate["compiled_over_prior_brier_gain_ci"],
        aggregate["compiled_over_prior_brier_gain_ci"],
    )

    compiled_groups: dict[str, Any] = {}
    group_summaries: list[dict[str, Any]] = []
    for model_offset, model in enumerate(config.models):
        for complexity_offset, n_entities in enumerate(config.panel.entity_counts):
            rows = [
                row
                for row in hard
                if row["model_id"] == model.model_id
                and row["entity_count"] == n_entities
            ]
            summary = _compiled_summary(
                rows,
                seed=seed + 31_000 + 100 * model_offset + complexity_offset,
                replicates=replicates,
            )
            reported = metrics["compiled_basis"]["by_model_and_complexity"][
                model.model_id
            ][str(n_entities)]
            label = f"{model.model_id}|{n_entities}"
            for comparison in ("direct", "prior"):
                field = f"compiled_over_{comparison}_brier_gain_ci"
                _compare_interval(
                    failures,
                    f"compiled {label}/{comparison}",
                    reported[field],
                    summary[field],
                )
            group_summaries.append(summary)
            compiled_groups[label] = summary

    family_summaries: dict[str, Any] = {}
    for offset, family in enumerate(sorted(HARD_FAMILIES)):
        rows = [row for row in hard if row["family"] == family]
        summary = _compiled_summary(
            rows,
            seed=seed + 35_000 + offset,
            replicates=replicates,
        )
        reported = metrics["compiled_basis"]["by_family"][family]
        for comparison in ("direct", "prior"):
            field = f"compiled_over_{comparison}_brier_gain_ci"
            _compare_interval(
                failures,
                f"compiled family {family}/{comparison}",
                reported[field],
                summary[field],
            )
        family_summaries[family] = summary

    contrasts = {
        "sequence_over_answer_token_brier_gain_ci": _paired_interval(
            records,
            baseline="answer_token",
            candidate="sequence",
            families=None,
            seed=seed + 20_000,
            replicates=replicates,
        ),
        "reason_over_pause_hard_family_brier_gain_ci": _paired_interval(
            records,
            baseline="pause",
            candidate="reason",
            families=HARD_FAMILIES,
            seed=seed + 20_001,
            replicates=replicates,
        ),
        "reason_over_sequence_hard_family_brier_gain_ci": _paired_interval(
            records,
            baseline="sequence",
            candidate="reason",
            families=HARD_FAMILIES,
            seed=seed + 20_002,
            replicates=replicates,
        ),
    }
    for label, expected in contrasts.items():
        _compare_interval(failures, label, metrics["contrasts"][label], expected)

    oracle_mismatches = sum(
        (float(row["oracle_compiled_probability"]) >= 0.5) != bool(row["truth_hard"])
        for row in compiled
    )
    if oracle_mismatches:
        failures.append("public executor disagrees with the hard oracle")

    prior_passed = aggregate["compiled_over_prior_brier_gain_ci"]["lower"] >= (
        config.gates.min_compiled_prior_gain_lower95
    ) and all(
        item["compiled_over_prior_brier_gain_ci"]["lower"]
        >= config.gates.min_compiled_prior_gain_lower95
        for item in group_summaries
    )
    direct_passed = aggregate["compiled_over_direct_brier_gain_ci"]["lower"] > (
        config.gates.min_compiled_direct_gain_lower95
    ) and all(
        item["compiled_over_direct_brier_gain_ci"]["lower"]
        > config.gates.min_compiled_direct_gain_lower95
        for item in group_summaries
    )
    basis_passed = bool(basis_passes) and all(basis_passes)
    expected_diagnosis = (
        "BASIS_READOUT_NOT_QUALIFIED"
        if not basis_passed
        else "PUBLIC_BASIS_NOT_SUFFICIENT"
        if not prior_passed
        else "NO_COMPOSITION_ADVANTAGE_OVER_TRAIN_SELECTED_DIRECT_BASELINE"
        if not direct_passed
        else "PUBLIC_BASIS_COMPOSITION_SUPPORTED"
    )
    expected_pass = basis_passed and prior_passed and direct_passed
    if decision.get("diagnosis") != expected_diagnosis:
        failures.append("machine diagnosis differs from independent gate reduction")
    if (decision.get("status") == "pass") != expected_pass:
        failures.append("machine status differs from independent gate reduction")

    authorization = decision.get("authorization", {})
    if any(
        authorization.get(key) is not False
        for key in (
            "stagea_outcome_run_authorized",
            "claim_bearing_test_access_authorized",
            "receiver_execution_authorized",
            "scientific_claim_authorized",
        )
    ):
        failures.append("machine decision opens a protected authorization")
    if authorization.get("stagea_registration_draft_authorized") is not expected_pass:
        failures.append("registration-draft authorization differs from the gate result")

    calibrators = list(calibration["groups"].values())
    calibration_summary = {
        "groups": len(calibrators),
        "negative_slopes": sum(
            float(item["calibrator"]["alpha"]) < 0.0 for item in calibrators
        ),
        "not_converged": sum(
            not bool(item["calibrator"]["converged"]) for item in calibrators
        ),
    }
    if calibration_summary["not_converged"]:
        failures.append("one or more calibration maps did not converge")

    return {
        "schema": "frank_eq_rate_compute_independent_audit_v1",
        "overall": "passed" if not failures else "failed",
        "failures": failures,
        "records": len(records),
        "compiled_predictions": len(compiled),
        "hard_family_predictions": len(hard),
        "direct_selection_groups": len(selected),
        "oracle_hard_mismatches": oracle_mismatches,
        "basis": basis_summaries,
        "compiled_aggregate": aggregate,
        "compiled_by_model_and_complexity": compiled_groups,
        "compiled_by_family": family_summaries,
        "contrasts": contrasts,
        "calibration": calibration_summary,
        "independent_gate_reduction": {
            "basis_passed": basis_passed,
            "compiled_over_prior_passed": prior_passed,
            "compiled_over_direct_passed": direct_passed,
            "status": "pass" if expected_pass else "fail",
            "diagnosis": expected_diagnosis,
        },
        "authorization": authorization,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = audit_run(args.run)
    if args.out:
        atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["overall"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
