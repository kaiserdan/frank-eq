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


def test_gate_reducer_fails_closed() -> None:
    metrics = {
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
    decision = reduce_stage0(metrics, GateConfig())
    assert decision["status"] == "fail"
    assert decision["authorizes_real_model_canary"] is False
    assert decision["authorizes_scientific_claim"] is False
