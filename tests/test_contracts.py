import pytest

from frank_eq.contracts import (
    FutureBranchRecord,
    FutureSignatureRecord,
    StateCaptureRecord,
    validate_world_split_integrity,
)

DIGEST = "a" * 64


def valid_record(split: str = "train", world: str = "w0", state: str = "s0") -> FutureSignatureRecord:
    capture = StateCaptureRecord(
        state_id=state,
        world_id=world,
        model_id="model-a",
        renderer_id="renderer-0",
        split=split,
        prefix_sha256=DIGEST,
        hidden_artifact_sha256=DIGEST,
        captured_before_operation=True,
        capture_step=10,
    )
    branches = (
        FutureBranchRecord(
            state_id=state,
            operation_id="verify",
            operation_descriptor_sha256=DIGEST,
            outcome_probabilities=(0.25, 0.75),
            branch_seed=1,
            operation_reveal_step=11,
        ),
    )
    return FutureSignatureRecord(capture=capture, branches=branches)


def test_future_signature_contract_passes() -> None:
    record = valid_record()
    record.validate({"verify"})
    assert len(record.content_sha256()) == 64


def test_operation_must_be_revealed_after_capture() -> None:
    record = valid_record()
    bad_branch = FutureBranchRecord(
        state_id="s0",
        operation_id="verify",
        operation_descriptor_sha256=DIGEST,
        outcome_probabilities=(0.5, 0.5),
        branch_seed=1,
        operation_reveal_step=10,
    )
    with pytest.raises(ValueError, match="revealed"):
        FutureSignatureRecord(record.capture, (bad_branch,)).validate()


def test_world_cannot_cross_splits() -> None:
    with pytest.raises(ValueError, match="crosses splits"):
        validate_world_split_integrity(
            [valid_record("train", "w0", "s0"), valid_record("test", "w0", "s1")]
        )
