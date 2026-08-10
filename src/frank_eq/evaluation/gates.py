"""Fail-closed Stage-0 decision reducer."""

from __future__ import annotations

from frank_eq.config import GateConfig


def reduce_stage0(metrics: dict[str, object], gates: GateConfig) -> dict[str, object]:
    checks = {
        "heldout_signature_brier": {
            "required": f"upper_95 <= {gates.max_heldout_signature_brier}",
            "observed": metrics["heldout_signature_brier_ci"]["upper"],
            "passed": metrics["heldout_signature_brier_ci"]["upper"]
            <= gates.max_heldout_signature_brier,
        },
        "fact_accuracy": {
            "required": f"lower_95 >= {gates.min_fact_accuracy}",
            "observed": metrics["fact_accuracy_ci"]["lower"],
            "passed": metrics["fact_accuracy_ci"]["lower"] >= gates.min_fact_accuracy,
        },
        "renderer_invariance": {
            "required": f"mean >= {gates.min_renderer_cosine}",
            "observed": metrics["renderer_cosine"],
            "passed": metrics["renderer_cosine"] >= gates.min_renderer_cosine,
        },
        "cross_model_retrieval": {
            "required": f"lower_95 >= {gates.min_cross_model_retrieval_top1}",
            "observed": metrics["cross_model_retrieval_top1_ci"]["lower"],
            "passed": metrics["cross_model_retrieval_top1_ci"]["lower"]
            >= gates.min_cross_model_retrieval_top1,
        },
        "wrong_world_margin": {
            "required": f"lower_95 >= {gates.min_wrong_world_margin}",
            "observed": metrics["wrong_world_margin_ci"]["lower"],
            "passed": metrics["wrong_world_margin_ci"]["lower"]
            >= gates.min_wrong_world_margin,
        },
        "operational_residual": {
            "required": f"lower_95 >= {gates.min_residual_brier_gain}",
            "observed": metrics["residual_brier_gain_ci"]["lower"],
            "passed": metrics["residual_brier_gain_ci"]["lower"]
            >= gates.min_residual_brier_gain,
        },
        "quantization_retention": {
            "required": f">= {gates.min_quantization_retention}",
            "observed": metrics["quantization_retention"],
            "passed": metrics["quantization_retention"] >= gates.min_quantization_retention,
        },
        "held_model_retention": {
            "required": f">= {gates.min_held_model_retention}",
            "observed": metrics["held_model_retention"],
            "passed": metrics["held_model_retention"] >= gates.min_held_model_retention,
        },
        "model_identity_leakage": {
            "required": f"over_chance <= {gates.max_model_leakage_over_chance}",
            "observed": metrics["model_leakage_over_chance"],
            "passed": metrics["model_leakage_over_chance"]
            <= gates.max_model_leakage_over_chance,
        },
    }
    failures = [name for name, result in checks.items() if not result["passed"]]
    return {
        "schema": "frank_eq_stage0_decision_v1",
        "status": "pass" if not failures else "fail",
        "decision": "PROMOTE_REAL_MODEL_CANARY" if not failures else "STOP_OR_REVISE_STAGE0",
        "authorizes_real_model_canary": not failures,
        "authorizes_scientific_claim": False,
        "checks": checks,
        "failures": failures,
        "scope": "synthetic implementation and falsification gate only",
    }
