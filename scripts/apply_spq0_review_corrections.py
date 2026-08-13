#!/usr/bin/env python3
"""Apply the independent scientific-review corrections to PR #6.

This helper is intentionally fail-closed and is removed by the one-shot workflow
that invokes it.  It never runs a model or reads a reserved checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def _write(path: str, text: str) -> None:
    (ROOT / path).write_text(text)


def _replace_once(path: str, old: str, new: str) -> None:
    text = _read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match in {path}, found {count}: {old[:80]!r}")
    _write(path, text.replace(old, new, 1))


def _regex_once(path: str, pattern: str, replacement: str) -> None:
    text = _read(path)
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"expected exactly one regex match in {path}, found {count}")
    _write(path, updated)


def _global_replace(path: str, old: str, new: str) -> None:
    text = _read(path)
    if old not in text:
        raise RuntimeError(f"expected at least one match in {path}: {old!r}")
    _write(path, text.replace(old, new))


def _patch_code() -> None:
    rename_paths = [
        "configs/spq0/real_olivia_spq0.yaml",
        "frank_eq_spq0_config_skeleton.yaml",
        "frank_eq_spq0_research_and_implementation_plan.md",
        "docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md",
        "src/frank_eq/shared_predictive_quotient/config.py",
        "src/frank_eq/shared_predictive_quotient/evaluation.py",
        "src/frank_eq/shared_predictive_quotient/probes.py",
        "tests/test_spq0.py",
    ]
    for path in rename_paths:
        _global_replace(path, "reduced_rank_regression", "truncated_ridge")
    for path in (
        "configs/spq0/real_olivia_spq0.yaml",
        "frank_eq_spq0_config_skeleton.yaml",
        "frank_eq_spq0_research_and_implementation_plan.md",
        "docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md",
        "src/frank_eq/shared_predictive_quotient/config.py",
        "src/frank_eq/shared_predictive_quotient/evaluation.py",
    ):
        _global_replace(path, "maxvar_gcca", "pooled_residual_pca")
    _global_replace(
        "docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md",
        "MAXVAR-GCCA",
        "pooled residual PCA",
    )
    _global_replace(
        "frank_eq_spq0_research_and_implementation_plan.md",
        "MAXVAR-GCCA",
        "pooled residual PCA",
    )
    _global_replace(
        "src/frank_eq/shared_predictive_quotient/evaluation.py",
        "_fit_gcca_residual",
        "_fit_pooled_residual_pca",
    )
    _replace_once(
        "src/frank_eq/shared_predictive_quotient/probes.py",
        '"""A centered ridge map with an optional reduced-rank coefficient matrix."""',
        '"""A centered ridge map with an optional truncated-ridge coefficient matrix."""',
    )

    _replace_once(
        "src/frank_eq/shared_predictive_quotient/config.py",
        "    packet_bit_cost_brier_equivalent: float = 0.0005\n",
        "    packet_bit_cost_brier_equivalent: float = 0.0005\n"
        "    rate_scalarization_promotional: bool = False\n",
    )
    _replace_once(
        "src/frank_eq/shared_predictive_quotient/config.py",
        "    activation_over_token_sequence_lower95_strict_gt: float = 0.0\n",
        "    activation_over_token_sequence_lower95_strict_gt: float = 0.0\n"
        "    cross_family_activation_over_token_lower95_strict_gt: float = 0.0\n",
    )
    _replace_once(
        "src/frank_eq/shared_predictive_quotient/config.py",
        "    rank4_noninferior_to_higher_ranks: bool = True\n",
        "    rank4_noninferior_to_higher_ranks: bool = True\n"
        "    rank4_transfer_noninferiority_margin: float = 0.002\n",
    )
    _replace_once(
        "src/frank_eq/shared_predictive_quotient/config.py",
        "        if (\n"
        "            evaluation.source_query_cost_brier_equivalent <= 0\n"
        "            or evaluation.packet_bit_cost_brier_equivalent <= 0\n"
        "        ):\n"
        "            raise ValueError(\"SPQ0 rate-aware utility exchange rates must be positive\")\n",
        "        if (\n"
        "            evaluation.source_query_cost_brier_equivalent <= 0\n"
        "            or evaluation.packet_bit_cost_brier_equivalent <= 0\n"
        "        ):\n"
        "            raise ValueError(\"SPQ0 rate diagnostic exchange rates must be positive\")\n"
        "        if evaluation.rate_scalarization_promotional:\n"
        "            raise ValueError(\n"
        "                \"SPQ0 heuristic rate scalarization must remain non-promotional\"\n"
        "            )\n",
    )
    _replace_once(
        "src/frank_eq/shared_predictive_quotient/config.py",
        "        if not 0.0 <= gates.min_four_bit_gain_retention <= 1.0:\n"
        "            raise ValueError(\"SPQ0 four-bit retention gate is invalid\")\n",
        "        if not 0.0 <= gates.min_four_bit_gain_retention <= 1.0:\n"
        "            raise ValueError(\"SPQ0 four-bit retention gate is invalid\")\n"
        "        if gates.rank4_transfer_noninferiority_margin < 0.0:\n"
        "            raise ValueError(\"SPQ0 transfer-rank noninferiority margin is invalid\")\n",
    )

    for path in (
        "configs/spq0/real_olivia_spq0.yaml",
        "frank_eq_spq0_config_skeleton.yaml",
    ):
        _replace_once(
            path,
            "  packet_bit_cost_brier_equivalent: 0.0005\n",
            "  packet_bit_cost_brier_equivalent: 0.0005\n"
            "  rate_scalarization_promotional: false\n",
        )
        _replace_once(
            path,
            "  activation_over_token_sequence_lower95_strict_gt: 0.0\n",
            "  activation_over_token_sequence_lower95_strict_gt: 0.0\n"
            "  cross_family_activation_over_token_lower95_strict_gt: 0.0\n",
        )
        _replace_once(
            path,
            "  rank4_noninferior_to_higher_ranks: true\n",
            "  rank4_noninferior_to_higher_ranks: true\n"
            "  rank4_transfer_noninferiority_margin: 0.002\n",
        )

    control_pattern = (
        r"    token_sequence = parameter_matched_token_sequence_features\(.*?"
        r"    controls\[\"final_token_residual\"\] = final_map\.predict\(final\[:, selected_depth\]\)\n"
    )
    control_replacement = '''    control_selection: dict[str, Any] = {}

    def select_control(
        name: str,
        feature_layers: np.ndarray,
    ) -> LinearMap:
        candidates_for_control: list[dict[str, float | int]] = []
        for layer in range(feature_layers.shape[1]):
            for ridge in config.semantic_encoder.ridge_grid:
                candidate = fit_linear_map(
                    feature_layers[calibration, layer],
                    arrays["semantic_core"][calibration],
                    ridge=float(ridge),
                    method="ridge",
                )
                selection_prediction = candidate.predict(
                    feature_layers[selection, layer]
                )
                candidates_for_control.append(
                    {
                        "layer": int(layer),
                        "ridge": float(ridge),
                        "selection_brier": brier_score(
                            arrays["semantic_core"][selection],
                            selection_prediction,
                        ),
                    }
                )
        selected_control = min(
            candidates_for_control,
            key=lambda row: (
                row["selection_brier"],
                row["layer"],
                row["ridge"],
            ),
        )
        fitted = fit_linear_map(
            feature_layers[fit_role, int(selected_control["layer"])],
            arrays["semantic_core"][fit_role],
            ridge=float(selected_control["ridge"]),
            method="ridge",
        )
        controls[name] = fitted.predict(
            feature_layers[:, int(selected_control["layer"])]
        )
        control_selection[name] = {
            "feature_map_learned": False,
            "selection_role": "selection",
            "fit_role": "calibration",
            "refit_roles": ["calibration", "selection"],
            "selected": selected_control,
            "candidates": candidates_for_control,
            "linear_map": fitted.metadata(),
        }
        return fitted

    token_sequence = parameter_matched_token_sequence_features(
        arrays["token_ids"],
        arrays["attention_mask"],
        arrays["event_token_indices"],
        width=activation_width,
        decay_grid=config.semantic_encoder.token_sequence_decay_grid,
    )
    token_map = select_control(
        "parameter_matched_token_sequence",
        token_sequence[:, None, :],
    )

    token_hash = deterministic_token_hash_features(
        arrays["token_ids"],
        arrays["attention_mask"],
        width=activation_width,
        position_period=config.semantic_encoder.token_hash_position_period,
    )
    token_hash_map = select_control(
        "deterministic_token_hash",
        token_hash[:, None, :],
    )

    embedding_map = select_control(
        "mean_input_embedding",
        arrays["mean_input_embedding"][:, None, :],
    )
    final_map = select_control(
        "final_token_residual",
        arrays["final_token_residual"],
    )
'''
    _regex_once(
        "src/frank_eq/shared_predictive_quotient/evaluation.py",
        control_pattern,
        control_replacement,
    )
    _replace_once(
        "src/frank_eq/shared_predictive_quotient/evaluation.py",
        '            "controls": {\n'
        '                "parameter_matched_token_sequence": token_map.metadata(),\n'
        '                "deterministic_token_hash": token_hash_map.metadata(),\n'
        '                "mean_input_embedding": embedding_map.metadata(),\n'
        '                "final_token_residual": final_map.metadata(),\n'
        '            },\n',
        '            "controls": control_selection,\n',
    )

    cross_family_function = '''def _cross_family_compositions(
    config: SPQRunConfig,
    basis: SharedPredictiveBasis,
    captures: Mapping[str, Mapping[str, np.ndarray]],
    evaluations: Mapping[str, ModelEvaluation],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate frozen source compilers through independently frozen target readers.

    Every direction includes a source-token packet control and a rank-conditioned
    transfer sweep.  This prevents a positive target-reader result from being
    attributed to activations when the fixed transcript sketch transfers equally
    well, and identifies rank in the cross-family endpoint rather than in a
    source-local semantic proxy.
    """

    pairs: dict[str, Any] = {}
    prediction_arrays: dict[str, np.ndarray] = {}
    model_ids = sorted(evaluations)
    for source_id in model_ids:
        for target_id in model_ids:
            if source_id == target_id:
                continue
            pair_name = f"{source_id}__to__{target_id}"
            source_arrays = captures[source_id]
            target_arrays = captures[target_id]
            source_rows, target_rows = _align_source_to_target(
                source_arrays, target_arrays
            )
            reader = evaluations[target_id].target_reader

            source_core = evaluations[source_id].predictions["decoded_core"][
                source_rows
            ]
            source_targets = evaluations[source_id].predictions[
                "compiled_targets"
            ][source_rows]
            transferred = reader.predict(
                source_core,
                source_targets,
                basis.target_tests,
            )

            token_core = evaluations[source_id].predictions[
                "control__parameter_matched_token_sequence"
            ][source_rows]
            token_targets = execute_target_rows(
                basis,
                token_core,
                source_arrays["system_ids"][source_rows],
                rank=basis.exact_rank,
            )
            token_transferred = reader.predict(
                token_core,
                token_targets,
                basis.target_tests,
            )

            oracle = reader.predict(
                target_arrays["semantic_core"][target_rows],
                target_arrays["semantic_targets"][target_rows],
                basis.target_tests,
            )
            rank_predictions: dict[int, np.ndarray] = {}
            for rank in config.semantic_encoder.rank_grid:
                public_packet = evaluations[source_id].predictions[
                    f"rank_{rank}"
                ][source_rows]
                rank_core = decode_core_rows(
                    basis,
                    public_packet,
                    source_arrays["system_ids"][source_rows],
                    rank=rank,
                )
                rank_targets = execute_target_rows(
                    basis,
                    public_packet,
                    source_arrays["system_ids"][source_rows],
                    rank=rank,
                )
                rank_predictions[rank] = reader.predict(
                    rank_core,
                    rank_targets,
                    basis.target_tests,
                )

            quantized_core = quantize_probabilities(source_core, 4)
            quantized_targets = execute_target_rows(
                basis,
                quantized_core,
                source_arrays["system_ids"][source_rows],
                rank=basis.exact_rank,
            )
            quantized_transfer = reader.predict(
                quantized_core,
                quantized_targets,
                basis.target_tests,
            )

            truth = evaluations[target_id].behavior_signatures[target_rows]
            validation = (
                target_arrays["role_ids"][target_rows] == ROLE_IDS["validation"]
            )
            joint = (
                validation
                & (
                    target_arrays["system_role_ids"][target_rows]
                    == SYSTEM_ROLE_IDS["validation_only"]
                )
                & (target_arrays["lengths"][target_rows] == 32)
                & (
                    target_arrays["renderer_ids"][target_rows]
                    == RENDERER_IDS["symbolic"]
                )
            )
            fit = (
                target_arrays["role_ids"][target_rows]
                != ROLE_IDS["validation"]
            )
            prior = np.repeat(
                truth[fit].mean(axis=0, keepdims=True),
                len(truth),
                axis=0,
            )
            group_ids = target_arrays["history_ids"][target_rows][joint]
            gain_ci = paired_brier_gain_interval(
                truth[joint],
                transferred[joint],
                prior[joint],
                group_ids,
                replicates=config.evaluation.bootstrap_replicates,
                seed=config.evaluation.bootstrap_seed + 20000 + len(pairs),
            )
            activation_over_token_ci = paired_brier_gain_interval(
                truth[joint],
                transferred[joint],
                token_transferred[joint],
                group_ids,
                replicates=config.evaluation.bootstrap_replicates,
                seed=config.evaluation.bootstrap_seed + 21000 + len(pairs),
            )
            prior_brier = brier_score(truth[joint], prior[joint])
            transfer_brier = brier_score(truth[joint], transferred[joint])
            token_brier = brier_score(truth[joint], token_transferred[joint])
            oracle_brier = brier_score(truth[joint], oracle[joint])
            denominator = prior_brier - oracle_brier
            retention = (
                0.0
                if denominator <= 1e-15
                else (prior_brier - transfer_brier) / denominator
            )
            float_gain = prior_brier - transfer_brier
            quantized_brier = brier_score(
                truth[joint], quantized_transfer[joint]
            )
            quantized_gain = prior_brier - quantized_brier
            four_bit_retention = (
                1.0
                if abs(float_gain) <= 1e-15
                else quantized_gain / float_gain
            )

            rank_transfer: dict[str, Any] = {}
            rank4_comparisons: dict[str, Any] = {}
            rank4_prediction = rank_predictions[basis.exact_rank]
            for rank in config.semantic_encoder.rank_grid:
                current = rank_predictions[rank]
                rank_transfer[str(rank)] = {
                    "brier": brier_score(truth[joint], current[joint]),
                    "formally_undercomplete": rank < basis.exact_rank,
                    "formally_rank_complete": rank >= basis.exact_rank,
                }
                if rank != basis.exact_rank:
                    rank4_comparisons[str(rank)] = paired_brier_gain_interval(
                        truth[joint],
                        rank4_prediction[joint],
                        current[joint],
                        group_ids,
                        replicates=config.evaluation.bootstrap_replicates,
                        seed=(
                            config.evaluation.bootstrap_seed
                            + 22000
                            + len(pairs) * 100
                            + rank
                        ),
                    )

            pairs[pair_name] = {
                "source_model": source_id,
                "source_family": evaluations[source_id].metrics["family"],
                "target_model": target_id,
                "target_family": evaluations[target_id].metrics["family"],
                "pair_specific_mapper": False,
                "target_reader_frozen_before_source_evaluation": True,
                "condition": "joint_ood",
                "rows": int(joint.sum()),
                "histories": int(len(np.unique(group_ids))),
                "transferred_brier": transfer_brier,
                "token_packet_brier": token_brier,
                "target_prior_brier": prior_brier,
                "oracle_core_reader_brier": oracle_brier,
                "gain_over_target_prior_ci": gain_ci,
                "activation_over_token_packet_gain_ci": (
                    activation_over_token_ci
                ),
                "oracle_reader_gain_retention": float(retention),
                "rank_transfer": rank_transfer,
                "rank4_transfer_comparisons": rank4_comparisons,
                "rank4_transfer_noninferiority_margin": (
                    config.gates.rank4_transfer_noninferiority_margin
                ),
                "four_bit_transferred_brier": quantized_brier,
                "four_bit_cross_family_gain_retention": float(
                    four_bit_retention
                ),
            }
            prediction_arrays[f"{pair_name}__transferred"] = transferred
            prediction_arrays[f"{pair_name}__token_control"] = token_transferred
            prediction_arrays[f"{pair_name}__oracle"] = oracle
            prediction_arrays[f"{pair_name}__four_bit"] = quantized_transfer
            for rank, prediction in rank_predictions.items():
                prediction_arrays[f"{pair_name}__rank_{rank}"] = prediction
    return pairs, prediction_arrays


'''
    _regex_once(
        "src/frank_eq/shared_predictive_quotient/evaluation.py",
        r"def _cross_family_compositions\(.*?\n\ndef _sender_identity\(",
        cross_family_function + "def _sender_identity(",
    )

    _replace_once(
        "src/frank_eq/shared_predictive_quotient/evaluation.py",
        '            and row["pair_specific_mapper"] is False\n'
        '            and row["target_reader_frozen_before_source_evaluation"] is True\n',
        '            and row["activation_over_token_packet_gain_ci"]["lower"]\n'
        '            > gate.cross_family_activation_over_token_lower95_strict_gt\n'
        '            and row["pair_specific_mapper"] is False\n'
        '            and row["target_reader_frozen_before_source_evaluation"] is True\n',
    )
    _regex_once(
        "src/frank_eq/shared_predictive_quotient/evaluation.py",
        r"    rank_identified: dict\[str, bool\] = \{\}.*?    quantization = \{",
        '''    rank_identified: dict[str, bool] = {}
    for pair, row in metrics["cross_family_composition"].items():
        comparisons = row["rank4_transfer_comparisons"]
        lower_rank_separation = all(
            float(comparisons[str(rank)]["lower"]) > 0.0
            for rank in (1, 2, 3)
        )
        higher_rank_noninferiority = all(
            float(comparisons[str(rank)]["lower"])
            >= -gate.rank4_transfer_noninferiority_margin
            for rank in (6, 8)
        )
        rank_identified[pair] = (
            lower_rank_separation and higher_rank_noninferiority
            if gate.rank4_noninferior_to_higher_ranks
            else True
        )
    quantization = {''',
    )
    _replace_once(
        "src/frank_eq/shared_predictive_quotient/evaluation.py",
        '        "amortized_rate_utility": bool(amortized) and all(amortized.values()),\n',
        "",
    )
    _replace_once(
        "src/frank_eq/shared_predictive_quotient/evaluation.py",
        '    elif not (\n'
        '        checks["four_bit_retention"]\n'
        '        and checks["sender_identity_closed"]\n'
        '        and checks["amortized_rate_utility"]\n'
        '    ):\n',
        '    elif not (\n'
        '        checks["four_bit_retention"]\n'
        '        and checks["sender_identity_closed"]\n'
        '    ):\n',
    )
    _replace_once(
        "src/frank_eq/shared_predictive_quotient/evaluation.py",
        '            "amortized_rate_utility": amortized,\n',
        '            "heuristic_rate_scalarization_non_promotional": amortized,\n',
    )

    _replace_once(
        "docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md",
        "The deterministic token-sequence encoder uses token identities, positions,\n"
        "and event boundaries. Its feature width is chosen so its learned linear readout\n"
        "has exactly the same number of trainable coefficients as the selected\n"
        "activation encoder. It is a primary activation-specificity control, not a\n"
        "weaker token-hash baseline.\n",
        "The fixed causal token sketch uses token identities, positions, and event\n"
        "boundaries with frozen exponential traces. Only its linear readout is learned,\n"
        "with exactly the same coefficient count as the selected activation readout.\n"
        "Its regularization is selected independently on the selection role. This is a\n"
        "development control for contextualization beyond a fixed transcript sketch; it\n"
        "is not represented as a learned recurrent or Transformer text encoder, and a\n"
        "future claim-bearing stage must include such a stronger text baseline.\n",
    )
    _replace_once(
        "docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md",
        "The source model's complete local semantic encoder produces the typed packet;\n"
        "the target model's already frozen local reader consumes it. Pair-specific\n"
        "alignment, translation, calibration, or mapper parameters are zero by\n"
        "construction.\n",
        "The source model's complete local semantic encoder produces the typed packet;\n"
        "the target model's already frozen local reader consumes it. The same reader also\n"
        "consumes a packet produced by the source model's fixed causal token sketch. Both\n"
        "ordered directions must show a positive paired activation-over-token packet\n"
        "margin on joint OOD. Pair-specific alignment, translation, calibration, or\n"
        "mapper parameters are zero by construction.\n",
    )
    _replace_once(
        "docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md",
        "The direct baseline asks the source separately for each target future. The\n"
        "comparison reports a frontier at 1, 4, 16, and 32 future queries, charging both\n"
        "source-query compute and packet bits using the frozen config exchange rates.\n"
        "The conjunctive primary gate is the grouped lower-95 utility advantage at 16\n"
        "future queries. A one-query direct advantage is reported but is not a\n"
        "conjunctive requirement. Development tomography queries are disclosed\n"
        "separately and are never counted as the primary packet.\n",
        "The direct baseline asks the source separately for each target future. The\n"
        "comparison reports packet bits per query and source-query branches at 1, 4, 16,\n"
        "and 32 future queries. The historical Brier-equivalent exchange-rate\n"
        "scalarization is retained as an explicitly heuristic, non-promotional\n"
        "diagnostic and cannot change the machine decision. Development tomography\n"
        "queries are disclosed separately and are never counted as the primary packet.\n",
    )
    _replace_once(
        "docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md",
        "4. activation surfaces strictly beat the parameter-matched token sequence;\n",
        "4. activation surfaces strictly beat the independently selected fixed causal\n"
        "   token sketch within each source and after both frozen cross-family readers;\n",
    )
    _replace_once(
        "docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md",
        "7. both ordered cross-family transfers beat the target prior and retain at\n"
        "   least 70% of oracle-reader gain;\n"
        "8. exact rank four is better than ranks 1--3 and noninferior to 6 and 8;\n",
        "7. both ordered cross-family transfers beat the target prior, beat their\n"
        "   source-token packet controls, and retain at least 70% of oracle-reader gain;\n"
        "8. in each ordered cross-family endpoint, rank four strictly beats ranks 1--3\n"
        "   and is noninferior to 6 and 8 within the registered 0.002 Brier margin;\n",
    )
    _replace_once(
        "docs/25_SHARED_PREDICTIVE_QUOTIENT_SPQ0.md",
        "10. sender identity is at most 0.15 above chance;\n"
        "11. the 16-query amortized rate utility has strictly positive lower 95% bound.\n",
        "10. sender identity is at most 0.15 above chance.\n\n"
        "The rate frontier and pooled residual PCA remain non-promotional diagnostics;\n"
        "neither can rescue a failed semantic, activation, rank, or transfer gate.\n",
    )

    appendix = '''

## Independent review corrections before launch

The implementation review made four prospective corrections before any model
execution. First, the transcript comparison is named precisely: it is a fixed
causal token sketch with an independently selected, parameter-matched linear
readout, not a learned sequence model. Second, every cross-family reader now
evaluates both the activation-derived packet and the source-token-derived packet;
a positive activation-over-token lower bound is conjunctive. Third, predictive
rank is identified at the cross-family target-reader endpoint with paired
rank-four comparisons, rather than from source-local semantic point estimates.
Fourth, the Brier-equivalent rate scalarization is diagnostic only because its
exchange rates are conventional rather than empirically identified. The
non-promotional behavioral remainder is pooled residual PCA, not MAXVAR-GCCA.

These corrections do not alter the systems, histories, model roster, public test
registry, access boundary, or protected authorizations. They prevent a favorable
SPQ0 outcome from being interpreted more strongly than the executed controls
support.
'''
    for path in (
        "frank_eq_spq0_research_and_implementation_plan.md",
        "docs/26_SPQ0_OLIVIA_RUNBOOK.md",
    ):
        text = _read(path)
        if "## Independent review corrections before launch" not in text:
            _write(path, text.rstrip() + appendix + "\n")

    _replace_once(
        "tests/test_spq0.py",
        "def test_reduced_rank_map_and_parameter_matched_token_surface_are_deterministic() -> None:\n",
        "def test_truncated_ridge_map_and_fixed_token_surface_are_deterministic() -> None:\n",
    )
    _replace_once(
        "tests/test_spq0.py",
        '    assert all(\n'
        '        row["pair_specific_mapper"] is False\n'
        '        and row["target_reader_frozen_before_source_evaluation"] is True\n'
        '        for row in metrics["cross_family_composition"].values()\n'
        '    )\n',
        '    assert all(\n'
        '        row["pair_specific_mapper"] is False\n'
        '        and row["target_reader_frozen_before_source_evaluation"] is True\n'
        '        and "activation_over_token_packet_gain_ci" in row\n'
        '        and set(row["rank_transfer"]) == {"1", "2", "3", "4", "6", "8"}\n'
        '        and set(row["rank4_transfer_comparisons"])\n'
        '        == {"1", "2", "3", "6", "8"}\n'
        '        for row in metrics["cross_family_composition"].values()\n'
        '    )\n',
    )
    _replace_once(
        "tests/test_spq0.py",
        '    assert metrics["behavioral_residual_census"]["promotional"] is False\n',
        '    assert metrics["behavioral_residual_census"]["promotional"] is False\n'
        '    assert metrics["behavioral_residual_census"]["method"].startswith(\n'
        '        "pooled_residual_pca"\n'
        '    )\n',
    )
    _replace_once(
        "tests/test_spq0.py",
        '    assert training["pair_specific_mapper_count"] == 0\n',
        '    assert training["pair_specific_mapper_count"] == 0\n'
        '    assert all(\n'
        '        all("selected" in control for control in model["semantic_encoder"]["controls"].values())\n'
        '        for model in training["models"].values()\n'
        '    )\n',
    )
    _replace_once(
        "tests/test_spq0.py",
        '    assert decision["authorization"]["receiver_execution_authorized"] is False\n',
        '    assert decision["authorization"]["receiver_execution_authorized"] is False\n'
        '    assert "amortized_rate_utility" not in decision["checks"]\n'
        '    assert (\n'
        '        "heuristic_rate_scalarization_non_promotional"\n'
        '        in decision["check_details"]\n'
        '    )\n',
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _update_registration() -> None:
    registration_path = ROOT / "configs/spq0/registration.json"
    plan_path = ROOT / "configs/spq0/inspected_plan.json"
    registration = json.loads(registration_path.read_text())
    plan = json.loads(plan_path.read_text())
    for relative in sorted(registration["files"]):
        registration["files"][relative] = _sha256(ROOT / relative)
    registration["inspected_plan_sha256"] = plan["plan_sha256"]
    registration["active_checkpoint_revision_registry_sha256"] = plan[
        "active_checkpoint_revision_registry_sha256"
    ]
    registration["reserved_checkpoint_non_access_contract_sha256"] = plan[
        "reserved_checkpoint_non_access_contract_sha256"
    ]
    registration_path.write_text(
        json.dumps(registration, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("patch", "register"))
    args = parser.parse_args()
    if args.phase == "patch":
        _patch_code()
    else:
        _update_registration()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
