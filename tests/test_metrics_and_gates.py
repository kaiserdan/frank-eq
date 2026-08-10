import numpy as np

from frank_eq.config import GateConfig
from frank_eq.evaluation.gates import reduce_stage0
from frank_eq.evaluation.metrics import cross_model_retrieval


def test_cross_model_retrieval_perfect_for_shared_codes() -> None:
    base = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    code = np.concatenate([base, base], axis=0)
    worlds = np.asarray([0, 1, 2, 0, 1, 2])
    models = np.asarray([0, 0, 0, 1, 1, 1])
    outcomes, margins, outcome_worlds = cross_model_retrieval(code, worlds, models)
    assert outcomes.mean() == 1.0
    assert np.all(margins > 0)
    assert len(outcome_worlds) == len(outcomes)


def _failing_metrics(scope: str = "synthetic future-defined causal-state Stage 0") -> dict:
    return {
        "scope": scope,
        "heldout_signature_brier_ci": {"upper": 0.5},
        "fact_accuracy_ci": {"lower": 0.1},
        "renderer_cosine": 0.0,
        "cross_model_retrieval_top1_ci": {"lower": 0.0},
        "wrong_world_margin_ci": {"lower": -1.0},
        "residual_brier_gain_ci": {"lower": -1.0},
        "quantization_retention": 0.0,
        "held_model_retention": 0.0,
        "model_leakage_over_chance": 1.0,
    }


def test_gate_reducer_fails_closed() -> None:
    decision = reduce_stage0(_failing_metrics(), GateConfig())
    assert decision["status"] == "fail"
    assert decision["authorizes_real_model_canary"] is False
    assert decision["authorizes_scientific_claim"] is False
    assert decision["schema"] == "frank_eq_stage0_decision_v1"


def test_real_gate_uses_real_scope_and_never_reauthorizes_canary() -> None:
    decision = reduce_stage0(
        _failing_metrics("real frozen-LLM future-defined causal-state Stage A"),
        GateConfig(),
    )
    assert decision["schema"] == "frank_eq_real_stagea_decision_v2"
    assert decision["scope"] == "real frozen-LLM Stage-A representation gate"
    assert decision["authorizes_real_model_canary"] is False
    assert decision["authorizes_receiver_protocol_design"] is False
