from frank_eq.rate_compute.records import (
    RateComputeResponse,
    _calibration_key,
    render_reasoning_query,
)
from frank_eq.schemas import OperationDefinition


def _response(item_id: int) -> RateComputeResponse:
    return RateComputeResponse(
        model_id="model-a",
        entity_count=4,
        world_id=4_000_001,
        local_world_id=1,
        renderer_id=0,
        split="train",
        kind="basis",
        item_id=item_id,
        family="edge",
        protocol="sequence",
        truth=0.98,
        raw_probability=0.75,
        raw_score=1.1,
        calibrated_probability=None,
        generated_tokens=0,
        prefix_sha256="prefix",
        query_sha256="query",
    )


def test_calibration_is_coordinate_specific() -> None:
    assert _calibration_key(_response(0)) != _calibration_key(_response(1))


def test_reasoning_query_reserves_semantic_answer_for_later_cue() -> None:
    operation = OperationDefinition(
        operation_id=0,
        family="compose",
        fact_args=(0, 3),
        residual_args=(1, 2),
        polarity=-1.0,
    )
    query = render_reasoning_query(operation, 4)

    assert "it is false that" in query
    assert "Do not give the final true/false answer" in query
    assert query.endswith("Reasoning:")
    assert "Reply with exactly" not in query
